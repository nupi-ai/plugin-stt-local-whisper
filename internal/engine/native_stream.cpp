//go:build whispercpp

#include "native_stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "whisper.h"

static constexpr int kSampleRate = WHISPER_SAMPLE_RATE;
static constexpr float kPi = 3.14159265358979323846f;
static constexpr int kVadWindowMs = 2000;
static constexpr int kVadLastMs = 1000;

struct StreamDeleter {
    void operator()(whisper_context *ctx) const noexcept {
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
    }
};

struct whisper_stream {
    std::unique_ptr<whisper_context, StreamDeleter> ctx;
    whisper_full_params params;

    std::vector<float> pcmf32_new;
    std::vector<float> pcmf32;
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_vad;

    std::string language_hint;
    bool detect_language = true;

    std::string last_window;
    std::string transcript;
    float last_confidence = 0.0f;

    int n_samples_step = 0;
    int n_samples_len = 0;
    int n_samples_keep = 0;
    int vad_window_samples = 0;
    int vad_last_ms = kVadLastMs;

    // Iteration tracking for n_new_line mechanism
    int n_iter = 0;
    int n_new_line = 1;

    bool keep_context = false;
    bool use_vad = false;
    float vad_thold = 0.6f;
    float freq_thold = 100.0f;

    std::vector<whisper_token> prompt_tokens;
    std::vector<whisper_token> current_tokens;
    std::vector<whisper_token> current_text_tokens;
    std::vector<whisper_token> previous_text_tokens;
};

static int samples_from_ms(int32_t ms) {
    if (ms <= 0) {
        return 0;
    }
    return static_cast<int>(static_cast<int64_t>(kSampleRate) * ms / 1000);
}

static void high_pass_filter(std::vector<float> &data, float cutoff, float sample_rate) {
    if (data.empty()) {
        return;
    }

    const float rc = 1.0f / (2.0f * kPi * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); ++i) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

static bool vad_detect_silence(const std::vector<float> &pcm,
                               int sample_rate,
                               int last_ms,
                               float vad_thold,
                               float freq_thold) {
    const int n_samples = static_cast<int>(pcm.size());
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples == 0 || n_samples_last <= 0 || n_samples_last >= n_samples) {
        return false;
    }

    std::vector<float> data = pcm;
    if (freq_thold > 0.0f) {
        high_pass_filter(data, freq_thold, static_cast<float>(sample_rate));
    }

    float energy_all = 0.0f;
    float energy_last = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        energy_all += std::fabs(data[i]);
        if (i >= n_samples - n_samples_last) {
            energy_last += std::fabs(data[i]);
        }
    }

    energy_all /= static_cast<float>(n_samples);
    energy_last /= static_cast<float>(n_samples_last);

    return energy_last <= vad_thold * energy_all;
}

static std::string trim(const std::string &s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return std::string();
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string collect_text(whisper_context *ctx, float &confidence_out) {
    const int n_segments = whisper_full_n_segments(ctx);
    if (n_segments == 0) {
        confidence_out = 0.0f;
        return {};
    }

    std::string text;
    text.reserve(256);

    double prob_sum = 0.0;
    int prob_count = 0;

    for (int i = 0; i < n_segments; ++i) {
        const char *segment = whisper_full_get_segment_text(ctx, i);
        if (segment != nullptr) {
            if (!text.empty()) {
                text.push_back(' ');
            }
            text += segment;
        }

        const int tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < tokens; ++j) {
            const auto data = whisper_full_get_token_data(ctx, i, j);
            if (data.p > 0.0f) {
                prob_sum += data.p;
                prob_count += 1;
            }
        }
    }

    confidence_out = prob_count > 0 ? static_cast<float>(prob_sum / prob_count) : 0.0f;
    return trim(text);
}

static bool is_text_token(whisper_context *ctx, whisper_token token, const char *piece) {
    if (ctx == nullptr || piece == nullptr || piece[0] == '\0') {
        return false;
    }

    // Skip known special tokens emitted as bracketed identifiers, e.g. [_TT_150], [_BEG_], etc.
    if (std::strncmp(piece, "[_", 2) == 0) {
        return false;
    }

    // Guard against core control tokens.
    const whisper_token token_beg = whisper_token_beg(ctx);
    if (token == whisper_token_eot(ctx) ||
        token == whisper_token_sot(ctx) ||
        token == whisper_token_solm(ctx) ||
        token == whisper_token_prev(ctx) ||
        token == whisper_token_nosp(ctx) ||
        token == whisper_token_not(ctx) ||
        token == token_beg) {
        return false;
    }

    if (token_beg != -1 && token >= token_beg) {
        return false;
    }

    return true;
}

static void collect_tokens(whisper_stream *stream) {
    stream->current_tokens.clear();
    stream->current_text_tokens.clear();

    if (stream == nullptr || stream->ctx == nullptr) {
        return;
    }

    whisper_context *ctx = stream->ctx.get();
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const int token_count = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < token_count; ++j) {
            const whisper_token token = whisper_full_get_token_id(ctx, i, j);
            stream->current_tokens.push_back(token);

            const char *piece = whisper_token_to_str(ctx, token);
            if (is_text_token(ctx, token, piece)) {
                stream->current_text_tokens.push_back(token);
            }
        }
    }

    // Debug: print first few text tokens
#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG] collect_tokens: total=%zu, text=%zu\n",
            stream->current_tokens.size(), stream->current_text_tokens.size());
    if (!stream->current_text_tokens.empty()) {
        fprintf(stderr, "[DEBUG] First 10 text tokens: ");
        for (size_t i = 0; i < std::min(size_t(10), stream->current_text_tokens.size()); ++i) {
            const char *piece = whisper_token_to_str(ctx, stream->current_text_tokens[i]);
            fprintf(stderr, "'%s' ", piece ? piece : "?");
        }
        fprintf(stderr, "\n");
    }
#endif
}

static std::string tokens_to_text(whisper_stream *stream,
                                  const std::vector<whisper_token> &tokens,
                                  size_t start_index) {
    if (stream->ctx == nullptr || start_index >= tokens.size()) {
        return {};
    }

    std::string text;
    text.reserve(tokens.size() * 4);

    whisper_context *ctx = stream->ctx.get();
    for (size_t i = start_index; i < tokens.size(); ++i) {
        const char *piece = whisper_token_to_str(ctx, tokens[i]);
        if (piece != nullptr && is_text_token(ctx, tokens[i], piece)) {
            text += piece;
        }
    }

    return trim(text);
}

static bool has_repetition_loop(const std::vector<whisper_token> &tokens) {
    if (tokens.size() < 8) {
        return false;
    }

    const whisper_token last = tokens.back();
    int repetition = 0;
    for (size_t idx = tokens.size(); idx-- > 0 && repetition < 8;) {
        if (tokens[idx] == last) {
            repetition++;
        } else {
            break;
        }
    }

    return repetition >= 8;
}

// Find longest common prefix between previous and current token lists
// Returns the index in current_tokens where new content starts
static size_t find_common_prefix(const std::vector<whisper_token> &previous,
                                  const std::vector<whisper_token> &current) {
    if (previous.empty()) {
        return 0;  // No previous tokens, everything is new
    }

    // Try to find where previous tokens end in current tokens
    // This handles the case where Whisper "shifts" the window
    size_t best_match = 0;
    size_t best_match_len = 0;

#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG] find_common_prefix: searching for overlap\n");
#endif

    // Look for the longest suffix of previous that matches a prefix of current
    for (size_t i = 0; i < previous.size(); ++i) {
        size_t match_len = 0;
        while (i + match_len < previous.size() &&
               match_len < current.size() &&
               previous[i + match_len] == current[match_len]) {
            match_len++;
        }
        if (match_len > best_match_len) {
            best_match_len = match_len;
            best_match = match_len;
#ifdef WHISPER_DEBUG
            fprintf(stderr, "[DEBUG]   suffix at pos %zu: %zu tokens match\n", i, match_len);
#endif
        }
    }

#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG]   best match: %zu tokens (new content starts at %zu)\n",
            best_match_len, best_match);
#endif

    return best_match;
}

static std::string extract_new_text(whisper_stream *stream, bool &reset_transcript) {
    reset_transcript = false;

    const auto &current = stream->current_text_tokens;
    const auto &previous = stream->previous_text_tokens;

    if (current.empty()) {
        return {};
    }

#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG] extract_new_text: prev_tokens=%zu, curr_tokens=%zu\n",
            previous.size(), current.size());

    // Debug: show last few previous tokens and first few current tokens
    if (!previous.empty() && stream->ctx) {
        whisper_context *ctx = stream->ctx.get();
        fprintf(stderr, "[DEBUG]   Last 5 prev tokens: ");
        for (size_t i = previous.size() > 5 ? previous.size() - 5 : 0; i < previous.size(); ++i) {
            const char *piece = whisper_token_to_str(ctx, previous[i]);
            fprintf(stderr, "'%s' ", piece ? piece : "?");
        }
        fprintf(stderr, "\n");

        fprintf(stderr, "[DEBUG]   First 5 curr tokens: ");
        for (size_t i = 0; i < std::min(size_t(5), current.size()); ++i) {
            const char *piece = whisper_token_to_str(ctx, current[i]);
            fprintf(stderr, "'%s' ", piece ? piece : "?");
        }
        fprintf(stderr, "\n");
    }
#endif

    // Find where new content starts (after common prefix with previous window)
    const size_t new_start = find_common_prefix(previous, current);

#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG]   Result: common_prefix=%zu\n", new_start);
#endif

    if (new_start >= current.size()) {
        // Everything is a repeat, no new content
#ifdef WHISPER_DEBUG
        fprintf(stderr, "[DEBUG] No new content (entire window is repeat)\n");
#endif
        return {};
    }

    // Extract text from the new tokens
    const std::string text = tokens_to_text(stream, current, new_start);

#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG] New text: '%s'\n", text.c_str());
#endif

    // Remember current tokens for next comparison
    stream->previous_text_tokens = current;

    return text;
}
static whisper_full_params prepare_params(whisper_stream *stream) {
    whisper_full_params params = stream->params;
    if (stream->detect_language || stream->language_hint.empty()) {
        params.language = nullptr;
        params.detect_language = true;
    } else {
        params.language = stream->language_hint.c_str();
        params.detect_language = false;
    }

    // Pass prompt tokens from previous segment
    // Only if no_context is false
    if (params.no_context) {
        params.prompt_tokens = nullptr;
        params.prompt_n_tokens = 0;
    } else {
        params.prompt_tokens = stream->prompt_tokens.empty() ? nullptr : stream->prompt_tokens.data();
        params.prompt_n_tokens = stream->prompt_tokens.size();
    }

#ifdef WHISPER_DEBUG
    fprintf(stderr, "[DEBUG] prepare_params: passing %zu prompt tokens to whisper_full()\n",
            stream->prompt_tokens.size());
#endif

    return params;
}

static int run_inference(whisper_stream *stream,
                         const float *data,
                         int n_samples,
                         std::string &out_text,
                         float &out_conf) {
    whisper_full_params params = prepare_params(stream);
    if (whisper_full(stream->ctx.get(), params, data, n_samples) != 0) {
        return -2;
    }

    out_text = collect_text(stream->ctx.get(), out_conf);
    stream->last_confidence = out_conf;

    collect_tokens(stream);

    // NOTE: Prompt tokens are updated in whisper_stream_process(),
    // synchronized with buffer reset every n_new_line iterations

    return 0;
}

static bool should_trigger_vad(whisper_stream *stream) {
    if (stream->vad_window_samples <= 0 ||
        static_cast<int>(stream->pcmf32_vad.size()) < stream->vad_window_samples) {
        return false;
    }

    const auto begin = stream->pcmf32_vad.end() - stream->vad_window_samples;
    std::vector<float> chunk(begin, stream->pcmf32_vad.end());
    return vad_detect_silence(chunk,
                              kSampleRate,
                              stream->vad_last_ms,
                              stream->vad_thold,
                              stream->freq_thold);
}

static int transcribe_vad_buffer(whisper_stream *stream,
                                 char **out_text,
                                 float *out_confidence) {
    const int total_samples = static_cast<int>(stream->pcmf32_vad.size());
    const int take = stream->n_samples_len > 0 ?
        std::min(stream->n_samples_len, total_samples) : total_samples;

    if (take <= 0) {
        stream->pcmf32_vad.clear();
        return 0;
    }

    stream->pcmf32.assign(stream->pcmf32_vad.end() - take, stream->pcmf32_vad.end());
    stream->pcmf32_vad.clear();

    std::string full_text;
    float confidence = 0.0f;
    if (run_inference(stream,
                      stream->pcmf32.data(),
                      static_cast<int>(stream->pcmf32.size()),
                      full_text,
                      confidence) != 0) {
        return -2;
    }

    const std::string trimmed = trim(full_text);
    stream->last_window = trimmed;
    stream->last_confidence = confidence;

    if (trimmed.empty()) {
        return 0;
    }

    *out_confidence = confidence;
    *out_text = static_cast<char *>(std::malloc(trimmed.size() + 1));
    if (*out_text == nullptr) {
        return -3;
    }

    std::memcpy(*out_text, trimmed.c_str(), trimmed.size() + 1);

    stream->transcript.clear();
    stream->prompt_tokens.clear();
    stream->current_tokens.clear();
    stream->current_text_tokens.clear();
    stream->previous_text_tokens.clear();
    stream->pcmf32.clear();
    stream->pcmf32_old.clear();

    return 1;
}

extern "C" {

whisper_stream *whisper_stream_create(const char *model_path,
                                      int32_t step_ms,
                                      int32_t length_ms,
                                      int32_t keep_ms,
                                      int32_t threads,
                                      bool use_gpu,
                                      bool flash_attn,
                                      bool translate,
                                      float temperature_inc,
                                      bool disable_fallback,
                                      int32_t beam_size,
                                      int32_t audio_ctx,
                                      bool print_timestamps,
                                      bool print_special,
                                      bool keep_context,
                                      bool use_vad,
                                      float vad_thold,
                                      float freq_thold,
                                      int32_t max_tokens,
                                      bool tinydiarize) {
    if (model_path == nullptr) {
        return nullptr;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu;
    cparams.flash_attn = flash_attn;

    auto ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (ctx == nullptr) {
        return nullptr;
    }

    // Choose sampling strategy based on beam_size
    whisper_sampling_strategy strategy = (beam_size > 1) ?
        WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

    if (use_vad) {
        keep_context = false;
    }

    auto *stream = new whisper_stream();
    stream->ctx.reset(ctx);
    stream->params = whisper_full_default_params(strategy);
    stream->params.print_progress = false;
    stream->params.print_special = print_special;  // Configurable
    stream->params.print_realtime = false;
    stream->params.print_timestamps = print_timestamps;  // Configurable
    stream->params.translate = translate;
    stream->params.single_segment = !use_vad;
    stream->params.no_context = !keep_context;
    stream->params.max_tokens = max_tokens > 0 ? max_tokens : 0;
    stream->params.audio_ctx = audio_ctx;  // Encoder context size
    stream->params.temperature_inc = disable_fallback ? 0.0f : temperature_inc;
    stream->params.n_threads = threads > 0 ? threads : 1;
    stream->params.tdrz_enable = tinydiarize;

    // Configure beam search if enabled
    if (beam_size > 1) {
        stream->params.beam_search.beam_size = beam_size;
    }

    stream->keep_context = keep_context;
    stream->use_vad = use_vad;
    stream->vad_thold = vad_thold;
    stream->freq_thold = freq_thold;
    stream->vad_window_samples = samples_from_ms(kVadWindowMs);
    stream->vad_last_ms = kVadLastMs;

    if (use_vad) {
        stream->n_samples_step = 0;
        stream->n_samples_len = std::max(samples_from_ms(length_ms), 1);
        stream->n_samples_keep = 0;
    } else {
        stream->n_samples_step = std::max(samples_from_ms(step_ms), 1);
        stream->n_samples_len = std::max(samples_from_ms(length_ms), stream->n_samples_step);
        stream->n_samples_keep = std::min(samples_from_ms(keep_ms), stream->n_samples_step);
    }

    // Calculate n_new_line for buffer reset
    if (!use_vad && step_ms > 0) {
        stream->n_new_line = std::max(1, static_cast<int>(length_ms / step_ms) - 1);
    } else {
        stream->n_new_line = 1;
    }
    stream->n_iter = 0;

    return stream;
}

void whisper_stream_free(whisper_stream *stream) {
    if (stream != nullptr) {
        delete stream;
    }
}

int whisper_stream_process(whisper_stream *stream,
                           const float *samples,
                           int32_t sample_count,
                           char **out_text,
                           float *out_confidence) {
    if (stream == nullptr || samples == nullptr || sample_count <= 0 ||
        out_text == nullptr || out_confidence == nullptr) {
        return -1;
    }

    if (stream->use_vad) {
        stream->pcmf32_vad.insert(stream->pcmf32_vad.end(), samples, samples + sample_count);
        if (stream->n_samples_len > 0) {
            const size_t max_keep = static_cast<size_t>(stream->n_samples_len + stream->vad_window_samples);
            if (stream->pcmf32_vad.size() > max_keep) {
                const size_t drop = stream->pcmf32_vad.size() - max_keep;
                stream->pcmf32_vad.erase(stream->pcmf32_vad.begin(), stream->pcmf32_vad.begin() + drop);
            }
        }
        if (!should_trigger_vad(stream)) {
            return 0;
        }
        return transcribe_vad_buffer(stream, out_text, out_confidence);
    }

    stream->pcmf32_new.insert(stream->pcmf32_new.end(), samples, samples + sample_count);

    if (static_cast<int>(stream->pcmf32_new.size()) < stream->n_samples_step) {
        return 0;
    }

    const int n_samples_new = static_cast<int>(stream->pcmf32_new.size());
    const int n_samples_take = std::min(
        static_cast<int>(stream->pcmf32_old.size()),
        std::max(0, stream->n_samples_keep + stream->n_samples_len - n_samples_new));

    stream->pcmf32.resize(n_samples_new + n_samples_take);
    if (n_samples_take > 0) {
        std::copy(stream->pcmf32_old.end() - n_samples_take,
                  stream->pcmf32_old.end(),
                  stream->pcmf32.begin());
    }
    std::copy(stream->pcmf32_new.begin(),
              stream->pcmf32_new.end(),
              stream->pcmf32.begin() + n_samples_take);

    stream->pcmf32_new.clear();

    // Keep entire processed buffeer
    stream->pcmf32_old = stream->pcmf32;

    std::string full_text;
    float confidence = 0.0f;
    if (run_inference(stream, stream->pcmf32.data(),
                      static_cast<int>(stream->pcmf32.size()),
                      full_text, confidence) != 0) {
        return -2;
    }

    stream->last_window = full_text;

    bool reset_transcript = false;
    std::string delta = extract_new_text(stream, reset_transcript);

    if (reset_transcript) {
        stream->prompt_tokens.clear();
        stream->transcript.clear();
        stream->previous_text_tokens.clear();
    }

    // Increment iteration counter and reset buffer if needed
    stream->n_iter++;
    if ((stream->n_iter % stream->n_new_line) == 0) {
        // Reset pcmf32_old to only keep last n_samples_keep
        const int keep_size = std::min(stream->n_samples_keep,
                                       static_cast<int>(stream->pcmf32.size()));
        stream->pcmf32_old.assign(
            stream->pcmf32.end() - keep_size,
            stream->pcmf32.end()
        );

        // Update prompt tokens for next iteration
        // Only if no_context is false
        if (!stream->params.no_context) {
            stream->prompt_tokens.clear();
            stream->prompt_tokens = stream->current_tokens;
        }
    }

    if (delta.empty()) {
        stream->last_confidence = confidence;
        return 0;
    }

    if (!stream->transcript.empty()) {
        stream->transcript.push_back(' ');
    }
    stream->transcript += delta;

    stream->last_confidence = confidence;
    *out_confidence = confidence;
    *out_text = static_cast<char *>(std::malloc(delta.size() + 1));
    if (*out_text == nullptr) {
        return -3;
    }
    std::memcpy(*out_text, delta.c_str(), delta.size() + 1);
    return 1;
}

int whisper_stream_flush(whisper_stream *stream,
                         char **out_text,
                         float *out_confidence) {
    if (stream == nullptr || out_text == nullptr || out_confidence == nullptr) {
        return -1;
    }

    if (stream->use_vad) {
        if (!stream->pcmf32_vad.empty()) {
            int rc = transcribe_vad_buffer(stream, out_text, out_confidence);
            if (rc <= 0) {
                return rc;
            }
            return rc;
        }
        return 0;
    }

    if (!stream->pcmf32_new.empty()) {
        const int n_samples_new = static_cast<int>(stream->pcmf32_new.size());
        const int n_samples_take = std::min(
            static_cast<int>(stream->pcmf32_old.size()),
            std::max(0, stream->n_samples_keep + stream->n_samples_len - n_samples_new));

        stream->pcmf32.resize(n_samples_new + n_samples_take);
        if (n_samples_take > 0) {
            std::copy(stream->pcmf32_old.end() - n_samples_take,
                      stream->pcmf32_old.end(),
                      stream->pcmf32.begin());
        }
        std::copy(stream->pcmf32_new.begin(),
                  stream->pcmf32_new.end(),
                  stream->pcmf32.begin() + n_samples_take);
        stream->pcmf32_new.clear();
        stream->pcmf32_old.assign(
            stream->pcmf32.end() - std::min(stream->n_samples_keep,
                                            static_cast<int>(stream->pcmf32.size())),
            stream->pcmf32.end());
        std::string full_text;
        float confidence = 0.0f;
        if (run_inference(stream, stream->pcmf32.data(),
                          static_cast<int>(stream->pcmf32.size()),
                          full_text, confidence) != 0) {
            return -2;
        }
        stream->last_window = full_text;
        stream->last_confidence = confidence;
    }

    bool reset_transcript = false;
    std::string delta = extract_new_text(stream, reset_transcript);
    if (reset_transcript) {
        stream->prompt_tokens.clear();
        stream->transcript.clear();
        stream->previous_text_tokens.clear();
    }
    if (!delta.empty()) {
        if (!stream->transcript.empty()) {
            stream->transcript.push_back(' ');
        }
        stream->transcript += delta;
    }

    std::string final_text = trim(stream->transcript);
    if (final_text.empty()) {
        stream->pcmf32.clear();
        stream->pcmf32_old.clear();
        stream->pcmf32_new.clear();
        stream->last_window.clear();
        stream->transcript.clear();
        stream->prompt_tokens.clear();
        stream->current_tokens.clear();
        stream->current_text_tokens.clear();
        stream->previous_text_tokens.clear();
        stream->last_confidence = 0.0f;
        return 0;
    }

    *out_confidence = stream->last_confidence;
    *out_text = static_cast<char *>(std::malloc(final_text.size() + 1));
    if (*out_text == nullptr) {
        return -3;
    }
    std::memcpy(*out_text, final_text.c_str(), final_text.size() + 1);

    stream->pcmf32.clear();
    stream->pcmf32_old.clear();
    stream->pcmf32_new.clear();
    stream->last_window.clear();
    stream->transcript.clear();
    stream->prompt_tokens.clear();
    stream->current_tokens.clear();
    stream->current_text_tokens.clear();
    stream->previous_text_tokens.clear();

    return 1;
}

void whisper_stream_free_text(char *text) {
    if (text != nullptr) {
        std::free(text);
    }
}

int whisper_stream_set_language(whisper_stream *stream,
                                const char *language,
                                bool detect_language) {
    if (stream == nullptr) {
        return -1;
    }

    stream->language_hint.clear();
    stream->detect_language = detect_language;
    if (!detect_language && language != nullptr) {
        stream->language_hint = language;
    }
    return 0;
}

} // extern "C"

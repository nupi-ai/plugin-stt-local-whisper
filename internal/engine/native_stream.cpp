//go:build whispercpp

#include "native_stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "whisper.h"

static constexpr int kSampleRate = WHISPER_SAMPLE_RATE;

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
    std::vector<float> pcmf32_all;

    std::string language_hint;
    bool detect_language = true;

    std::string last_window;
    std::string transcript;
    float last_confidence = 0.0f;

    int n_samples_step = 0;
    int n_samples_len = 0;
    int n_samples_keep = 0;
};

static int samples_from_ms(int32_t ms) {
    if (ms <= 0) {
        return 0;
    }
    return static_cast<int>(static_cast<int64_t>(kSampleRate) * ms / 1000);
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

static std::string compute_delta(const std::string &previous, const std::string &current) {
    if (previous.empty()) {
        return trim(current);
    }

    const auto prev = previous;
    const auto curr = current;

    size_t prefix = 0;
    const size_t prefix_limit = std::min(prev.size(), curr.size());
    while (prefix < prefix_limit && prev[prefix] == curr[prefix]) {
        ++prefix;
    }

    size_t prev_suffix = prev.size();
    size_t curr_suffix = curr.size();
    while (prev_suffix > prefix && curr_suffix > prefix &&
           prev[prev_suffix - 1] == curr[curr_suffix - 1]) {
        --prev_suffix;
        --curr_suffix;
    }

    if (curr_suffix <= prefix) {
        return {};
    }

    return trim(curr.substr(prefix, curr_suffix - prefix));
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
    return 0;
}

extern "C" {

whisper_stream *whisper_stream_create(const char *model_path,
                                      int32_t step_ms,
                                      int32_t length_ms,
                                      int32_t keep_ms,
                                      int32_t threads,
                                      bool use_gpu,
                                      bool flash_attn) {
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

    auto *stream = new whisper_stream();
    stream->ctx.reset(ctx);
    stream->params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    stream->params.print_progress = false;
    stream->params.print_special = false;
    stream->params.print_realtime = false;
    stream->params.print_timestamps = false;
    stream->params.translate = false;
    stream->params.single_segment = true;
    stream->params.no_context = true;
    stream->params.temperature_inc = 0.0f;
    stream->params.n_threads = threads > 0 ? threads : 1;

    stream->n_samples_step = std::max(samples_from_ms(step_ms), 1);
    stream->n_samples_len = std::max(samples_from_ms(length_ms), stream->n_samples_step);
    stream->n_samples_keep = std::min(samples_from_ms(keep_ms), stream->n_samples_step);

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

    stream->pcmf32_new.insert(stream->pcmf32_new.end(), samples, samples + sample_count);
    stream->pcmf32_all.insert(stream->pcmf32_all.end(), samples, samples + sample_count);

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

    std::string delta = compute_delta(stream->last_window, full_text);
    stream->last_window = full_text;

    if (delta.empty()) {
        return 0;
    }

    if (!stream->transcript.empty()) {
        stream->transcript.push_back(' ');
    }
    stream->transcript += delta;

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
                          full_text, confidence) == 0) {
            std::string delta = compute_delta(stream->last_window, full_text);
            stream->last_window = full_text;
            if (!delta.empty()) {
                if (!stream->transcript.empty()) {
                    stream->transcript.push_back(' ');
                }
                stream->transcript += delta;
            }
        }
    }

    if (!stream->pcmf32_all.empty()) {
        whisper_full_params backup = prepare_params(stream);
        backup.single_segment = false;
        if (whisper_full(stream->ctx.get(), backup,
                         stream->pcmf32_all.data(),
                         static_cast<int>(stream->pcmf32_all.size())) == 0) {
            float confidence = 0.0f;
            std::string full = collect_text(stream->ctx.get(), confidence);
            stream->transcript = trim(full);
            stream->last_window = stream->transcript;
            stream->last_confidence = confidence;
        }
    }

    *out_confidence = stream->last_confidence;
    *out_text = static_cast<char *>(std::malloc(stream->transcript.size() + 1));
    if (*out_text == nullptr) {
        return -3;
    }
    std::memcpy(*out_text, stream->transcript.c_str(), stream->transcript.size() + 1);

    stream->pcmf32.clear();
    stream->pcmf32_old.clear();
    stream->pcmf32_new.clear();
    stream->pcmf32_all.clear();
    stream->last_window.clear();
    stream->transcript.clear();
    stream->last_confidence = 0.0f;

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

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct whisper_stream whisper_stream;

/// Creates a streaming Whisper context.
/// Returns NULL on failure.
whisper_stream *whisper_stream_create(const char *model_path,
                                      int32_t step_ms,
                                      int32_t length_ms,
                                      int32_t keep_ms,
                                      int32_t threads,
                                      bool use_gpu,
                                      bool flash_attn);

/// Releases all resources associated with the stream.
void whisper_stream_free(whisper_stream *stream);

/// Feeds new audio samples (mono PCM float32) into the stream.
/// On success:
///   - returns 1 when new text is available and sets out_text/confidence
///   - returns 0 when more audio is required (out_text is NULL)
/// On failure returns negative value.
int whisper_stream_process(whisper_stream *stream,
                           const float *samples,
                           int32_t sample_count,
                           char **out_text,
                           float *out_confidence);

/// Finalises the transcription and returns the full transcript.
/// Returns negative value on failure.
int whisper_stream_flush(whisper_stream *stream,
                         char **out_text,
                         float *out_confidence);

/// Frees strings returned by process / flush.
void whisper_stream_free_text(char *text);

/// Configures the language handling strategy.
/// When detect_language is true, the model will auto-detect language regardless of hint.
/// When detect_language is false and language is non-null, the provided hint is enforced.
/// Returns 0 on success, negative value on error.
int whisper_stream_set_language(whisper_stream *stream,
                                const char *language,
                                bool detect_language);

#ifdef __cplusplus
}
#endif

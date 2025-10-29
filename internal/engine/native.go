//go:build whispercpp

package engine

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp -I${SRCDIR}/../../third_party/whisper.cpp/include -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/../../third_party/whisper.cpp -I${SRCDIR}/../../third_party/whisper.cpp/include -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/whisper.cpp/build -L${SRCDIR}/../../third_party/whisper.cpp/build/src -Wl,-rpath,${SRCDIR}/../../third_party/whisper.cpp/build/src -lwhisper -lstdc++ -lm

#include "stdlib.h"
#include "include/whisper.h"
#include "ggml.h"

bool whisperGoAbort(void * user_data);
*/
import "C"

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"runtime/cgo"
	"strings"
	"sync"
	"unsafe"
)

const (
	whisperSampleRate   = 16000
	bytesPerSample      = 2
	windowSeconds       = 30    // maximum window for very long audio
	targetWindowSeconds = 10    // target window size matching stream.cpp length_ms
	minFrameMillis      = 3000  // batch process every ~3s (like stream.cpp step_ms)
	maxAudioBytes       = whisperSampleRate * bytesPerSample * windowSeconds
	targetWindowBytes   = whisperSampleRate * bytesPerSample * targetWindowSeconds
	minWhisperBytes     = whisperSampleRate * bytesPerSample * minFrameMillis / 1000
	audioTrimChunkSize  = whisperSampleRate * bytesPerSample // trim in ~1 s chunks
)

func NativeAvailable() bool { return true }

type NativeEngine struct {
	mu      sync.Mutex
	inferMu sync.Mutex

	ctx          *C.struct_whisper_context
	audio        []byte
	lastSegment  []byte // For overlapping between batches (like pcmf32_old in stream.cpp)
	promptTokens []C.whisper_token
	lastText     string
	language     string
	lastConf     float32
	defaultLang  string
}

func NewNativeEngine(modelPath string) (Engine, error) {
	if modelPath == "" {
		return nil, errors.New("whisper: model path required")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	cParams := C.whisper_context_default_params()
	cParams.use_gpu = C.bool(false)

	ctx := C.whisper_init_from_file_with_params(cPath, cParams)
	if ctx == nil {
		return nil, fmt.Errorf("whisper: failed to initialise context for %s", modelPath)
	}

	return &NativeEngine{
		ctx: ctx,
	}, nil
}

func (e *NativeEngine) TranscribeSegment(ctx context.Context, audio []byte, opts Options) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(audio) == 0 {
		return nil, nil
	}

	var (
		prevLang string
		forced   string
	)
	e.mu.Lock()
	prevLang = e.language
	forced = e.defaultLang
	e.mu.Unlock()

	lang := normaliseLanguage(opts.Language, prevLang, forced)

	e.mu.Lock()
	// Accumulate new audio
	e.audio = append(e.audio, audio...)

	// Check if we have enough audio to process (batch processing like stream.cpp)
	if len(e.audio) < minWhisperBytes {
		e.mu.Unlock()
		return nil, nil
	}

	// Create processing buffer: overlap from lastSegment + new audio
	// Following stream.cpp approach: use enough old audio to reach target window size (10s)
	var buffer []byte
	if len(e.lastSegment) > 0 {
		// Calculate how much old audio we need to reach targetWindowBytes
		// This matches stream.cpp: n_samples_take = min(old.size, max(0, keep + len - new))
		overlapBytesNeeded := targetWindowBytes - len(e.audio)
		if overlapBytesNeeded < 0 {
			overlapBytesNeeded = 0
		}
		if overlapBytesNeeded > len(e.lastSegment) {
			overlapBytesNeeded = len(e.lastSegment)
		}

		overlapStart := len(e.lastSegment) - overlapBytesNeeded
		overlap := e.lastSegment[overlapStart:]
		buffer = make([]byte, 0, len(overlap)+len(e.audio))
		buffer = append(buffer, overlap...)
		buffer = append(buffer, e.audio...)
	} else {
		buffer = append([]byte(nil), e.audio...)
	}

	// Trim if buffer exceeds maximum window size
	if len(buffer) > maxAudioBytes {
		buffer = buffer[len(buffer)-maxAudioBytes:]
	}

	previous := e.lastText
	e.mu.Unlock()

	agg, err := e.runInference(ctx, buffer, lang)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, nil
		}
		slog.Warn("native inference failed",
			"err", err,
			"ctx_err", ctx.Err(),
			"audio_len", len(buffer),
			"language", lang,
		)
		return nil, err
	}
	if agg.Text != "" {
		slog.Info("native inference aggregate",
			"stage", "segment",
			"audio_len", len(buffer),
			"language", lang,
			"text", agg.Text,
			"confidence", agg.Confidence,
		)
	}

	e.mu.Lock()
	e.language = lang
	// CRITICAL: Reset audio buffer after processing (batch processing)
	e.audio = nil
	// Save processed buffer as lastSegment for next batch's overlap
	e.lastSegment = buffer
	delta := diffTranscript(previous, agg.Text)
	e.lastText = agg.Text
	e.lastConf = agg.Confidence
	e.mu.Unlock()

	if delta == "" {
		return nil, nil
	}

	return []Result{
		{
			Text:       delta,
			Confidence: agg.Confidence,
			Final:      false,
		},
	}, nil
}

func (e *NativeEngine) Flush(ctx context.Context, opts Options) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var (
		prevLang string
		forced   string
	)
	e.mu.Lock()
	prevLang = e.language
	forced = e.defaultLang
	buffer := append([]byte(nil), e.audio...)
	previous := e.lastText
	e.mu.Unlock()

	lang := normaliseLanguage(opts.Language, prevLang, forced)

	var (
		combined   string
		confidence float32
		err        error
	)
	if len(buffer) > 0 {
		var agg transcriptAggregate
		agg, err = e.runInference(ctx, buffer, lang)
		if err == nil {
			combined = agg.Text
			confidence = agg.Confidence
		}
	} else {
		combined = previous
		confidence = e.lastConf
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, nil
		}
		slog.Warn("native flush inference failed",
			"err", err,
			"ctx_err", ctx.Err(),
			"audio_len", len(buffer),
			"language", lang,
		)
		e.mu.Lock()
		e.resetLocked()
		e.mu.Unlock()
		return nil, err
	}
	if combined != "" {
		slog.Info("native inference aggregate",
			"stage", "flush",
			"audio_len", len(buffer),
			"language", lang,
			"text", combined,
			"confidence", confidence,
		)
	}

	finalText := strings.TrimSpace(combined)

	e.mu.Lock()
	e.resetLocked()
	e.mu.Unlock()

	if finalText == "" {
		return nil, nil
	}

	return []Result{{
		Text:       finalText,
		Confidence: confidence,
		Final:      true,
	}}, nil
}

func (e *NativeEngine) Close() error {
	e.inferMu.Lock()
	defer e.inferMu.Unlock()

	e.mu.Lock()
	defer e.mu.Unlock()
	e.resetLocked()
	if e.ctx != nil {
		C.whisper_free(e.ctx)
		e.ctx = nil
	}
	return nil
}

func (e *NativeEngine) runInference(ctx context.Context, audio []byte, language string) (transcriptAggregate, error) {
	if err := ctx.Err(); err != nil {
		return transcriptAggregate{}, err
	}
	if len(audio) == 0 {
		return transcriptAggregate{}, nil
	}

	samples := pcmBytesToFloat32(audio)
	if len(samples) == 0 {
		return transcriptAggregate{}, nil
	}

	e.inferMu.Lock()
	defer e.inferMu.Unlock()

	state := C.whisper_init_state(e.ctx)
	if state == nil {
		return transcriptAggregate{}, errors.New("whisper: failed to initialise state")
	}
	defer C.whisper_free_state(state)

	cSamples := (*C.float)(unsafe.Pointer(&samples[0]))
	nSamples := C.int(len(samples))

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.print_progress = C.bool(false)
	params.print_realtime = C.bool(false)
	params.print_timestamps = C.bool(false)
	params.translate = C.bool(false)
	params.no_context = C.bool(false)
	params.single_segment = C.bool(false)

	lang := strings.TrimSpace(language)
	if lang == "" {
		lang = "auto"
	}
	cLang := C.CString(lang)
	params.language = cLang
	if strings.EqualFold(lang, "auto") {
		params.detect_language = C.bool(true)
	}
	defer C.free(unsafe.Pointer(cLang))

	handle := cgo.NewHandle(ctx)
	defer handle.Delete()
	params.abort_callback = (C.ggml_abort_callback)(C.whisperGoAbort)
	params.abort_callback_user_data = unsafe.Pointer(&handle)

	// Pass prompt tokens from previous segments to maintain context
	if len(e.promptTokens) > 0 {
		params.prompt_tokens = &e.promptTokens[0]
		params.prompt_n_tokens = C.int(len(e.promptTokens))
	}

	if ret := C.whisper_full_with_state(e.ctx, state, params, cSamples, nSamples); ret != 0 {
		if err := ctx.Err(); err != nil {
			return transcriptAggregate{}, err
		}
		return transcriptAggregate{}, fmt.Errorf("whisper: inference failed with code %d", int(ret))
	}

	// Collect tokens from this inference for next call's context
	e.collectPromptTokens(state)

	return collectTranscriptAggregate(state), nil
}

// collectPromptTokens gathers tokens from the current inference to use as context for the next call.
// This follows the approach from whisper-stream example in whisper.cpp repository.
func (e *NativeEngine) collectPromptTokens(state *C.struct_whisper_state) {
	const maxPromptTokens = 224 // whisper_n_text_ctx()/2, typically 448/2 = 224

	if state == nil {
		return
	}

	nSegments := int(C.whisper_full_n_segments_from_state(state))
	if nSegments == 0 {
		return
	}

	// Collect all tokens from all segments
	var tokens []C.whisper_token
	for i := 0; i < nSegments; i++ {
		nTokens := int(C.whisper_full_n_tokens_from_state(state, C.int(i)))
		for j := 0; j < nTokens; j++ {
			tokenID := C.whisper_full_get_token_id_from_state(state, C.int(i), C.int(j))
			tokens = append(tokens, tokenID)
		}
	}

	// Keep only the last maxPromptTokens tokens
	if len(tokens) > maxPromptTokens {
		tokens = tokens[len(tokens)-maxPromptTokens:]
	}

	e.promptTokens = tokens
}

func (e *NativeEngine) resetLocked() {
	e.audio = nil
	e.lastSegment = nil
	e.promptTokens = nil
	e.lastText = ""
	e.language = ""
	e.lastConf = 0
}

// SetDefaultLanguage configures the language hint to use when callers request auto detection.
func (e *NativeEngine) SetDefaultLanguage(lang string) {
	trimmed := preferLanguage(lang)
	e.mu.Lock()
	e.defaultLang = trimmed
	e.mu.Unlock()
}

func pcmBytesToFloat32(buf []byte) []float32 {
	n := len(buf) / 2
	if n == 0 {
		return nil
	}
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		u := binary.LittleEndian.Uint16(buf[2*i:])
		val := int16(u)
		samples[i] = float32(val) / 32768.0
	}
	return samples
}

//export whisperGoAbort
func whisperGoAbort(userData unsafe.Pointer) C.bool {
	if shouldAbort(userData) {
		return C.bool(true)
	}
	return C.bool(false)
}

type transcriptAggregate struct {
	Text       string
	Confidence float32
}

func collectTranscriptAggregate(state *C.struct_whisper_state) transcriptAggregate {
	if state == nil {
		return transcriptAggregate{}
	}
	count := int(C.whisper_full_n_segments_from_state(state))
	if count == 0 {
		return transcriptAggregate{}
	}
	var (
		builder      strings.Builder
		sumProb      float64
		tokenSamples int
	)
	for i := 0; i < count; i++ {
		text := strings.TrimSpace(C.GoString(C.whisper_full_get_segment_text_from_state(state, C.int(i))))
		if text != "" {
			if builder.Len() > 0 {
				builder.WriteByte(' ')
			}
			builder.WriteString(text)
		}
		tokenCount := int(C.whisper_full_n_tokens_from_state(state, C.int(i)))
		for j := 0; j < tokenCount; j++ {
			tokenData := C.whisper_full_get_token_data_from_state(state, C.int(i), C.int(j))
			if tokenData.p > 0 {
				sumProb += float64(tokenData.p)
				tokenSamples++
			}
		}
	}
	confidence := float32(0)
	if tokenSamples > 0 {
		confidence = float32(sumProb / float64(tokenSamples))
	}
	text := strings.TrimSpace(builder.String())
	if strings.EqualFold(text, "[BLANK_AUDIO]") {
		text = ""
	}
	return transcriptAggregate{
		Text:       text,
		Confidence: confidence,
	}
}

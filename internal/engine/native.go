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
	"runtime/cgo"
	"strings"
	"sync"
	"unsafe"
)

const maxAudioBytes = 10 * 1024 * 1024 // safety cap per stream

func NativeAvailable() bool { return true }

type NativeEngine struct {
	mu      sync.Mutex
	inferMu sync.Mutex

	ctx      *C.struct_whisper_context
	audio    []byte
	lastText string
	language string
	lastConf float32
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

	lang := normaliseLanguage(opts.Language, e.language)

	e.mu.Lock()
	if len(e.audio)+len(audio) > maxAudioBytes {
		e.mu.Unlock()
		return nil, fmt.Errorf("whisper: audio buffer overflow (max %d bytes)", maxAudioBytes)
	}
	e.audio = append(e.audio, audio...)
	buffer := append([]byte(nil), e.audio...)
	previous := e.lastText
	e.mu.Unlock()

	agg, err := e.runInference(ctx, buffer, lang)
	if err != nil {
		return nil, err
	}

	e.mu.Lock()
	e.language = lang
	e.audio = buffer
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

	lang := normaliseLanguage(opts.Language, e.language)

	e.mu.Lock()
	buffer := append([]byte(nil), e.audio...)
	previous := e.lastText
	e.mu.Unlock()

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
		e.mu.Lock()
		e.resetLocked()
		e.mu.Unlock()
		return nil, err
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
	params.no_context = C.bool(true)
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

	if ret := C.whisper_full_with_state(e.ctx, state, params, cSamples, nSamples); ret != 0 {
		if err := ctx.Err(); err != nil {
			return transcriptAggregate{}, err
		}
		return transcriptAggregate{}, fmt.Errorf("whisper: inference failed with code %d", int(ret))
	}

	return collectTranscriptAggregate(state), nil
}

func (e *NativeEngine) resetLocked() {
	e.audio = nil
	e.lastText = ""
	e.language = ""
	e.lastConf = 0
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
	return transcriptAggregate{
		Text:       strings.TrimSpace(builder.String()),
		Confidence: confidence,
	}
}

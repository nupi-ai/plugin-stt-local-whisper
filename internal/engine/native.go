//go:build whispercpp

package engine

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp -I${SRCDIR}/../../third_party/whisper.cpp/include -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/../../third_party/whisper.cpp -I${SRCDIR}/../../third_party/whisper.cpp/include -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/whisper.cpp/build -L${SRCDIR}/../../third_party/whisper.cpp/build/src -L${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src -L${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml-metal -L${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml-blas -Wl,-rpath,${SRCDIR}/../../third_party/whisper.cpp/build/src -Wl,-rpath,${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src -Wl,-rpath,${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml-metal -Wl,-rpath,${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml-blas -lwhisper -lggml -lggml-base -lggml-cpu -lm
#cgo !darwin LDFLAGS: -lstdc++
#cgo darwin LDFLAGS: -lggml-metal -lggml-blas -framework Accelerate -framework Metal -framework Foundation -framework CoreGraphics
#include <stdlib.h>
#include "native_stream.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

const (
	minFrameMillis      = 3000 // step_ms
	targetWindowMillis  = 10000
	keepMillis          = 200
	defaultFlashAttnEnv = "WHISPERCPP_FLASH_ATTENTION"
	useGPUEnv           = "WHISPERCPP_USE_GPU"
	threadsEnv          = "WHISPERCPP_THREADS"
)

func NativeAvailable() bool { return true }

type NativeEngine struct {
	mu sync.Mutex

	stream *C.whisper_stream

	defaultLang        string
	lastConf           float32
	lastLanguage       string
	lastDetectLanguage bool
	languageConfigured bool
}

func NewNativeEngine(modelPath string, opts NativeOptions) (Engine, error) {
	if strings.TrimSpace(modelPath) == "" {
		return nil, errors.New("whisper: model path required")
	}

	useGPU := true
	if opts.UseGPU != nil {
		useGPU = *opts.UseGPU
	} else if env := strings.TrimSpace(os.Getenv(useGPUEnv)); env != "" {
		if parsed, err := strconv.ParseBool(env); err == nil {
			useGPU = parsed
		}
	}

	flashAttn := true
	if opts.FlashAttention != nil {
		flashAttn = *opts.FlashAttention
	} else if env := strings.TrimSpace(os.Getenv(defaultFlashAttnEnv)); env != "" {
		if parsed, err := strconv.ParseBool(env); err == nil {
			flashAttn = parsed
		}
	}

	threads := runtime.NumCPU()
	if opts.Threads != nil && *opts.Threads > 0 {
		threads = *opts.Threads
	} else if env := strings.TrimSpace(os.Getenv(threadsEnv)); env != "" {
		if parsed, err := strconv.Atoi(env); err == nil && parsed > 0 {
			threads = parsed
		}
	}

	cModel := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModel))

	stream := C.whisper_stream_create(
		cModel,
		C.int32_t(minFrameMillis),
		C.int32_t(targetWindowMillis),
		C.int32_t(keepMillis),
		C.int32_t(threads),
		C.bool(useGPU),
		C.bool(flashAttn),
	)
	if stream == nil {
		return nil, fmt.Errorf("whisper: failed to initialise context for %s", modelPath)
	}

	return &NativeEngine{
		stream: stream,
	}, nil
}

func (e *NativeEngine) TranscribeSegment(ctx context.Context, audio []byte, opts Options) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(audio) == 0 {
		return nil, nil
	}

	samples := pcmBytesToFloat32(audio)
	if len(samples) == 0 {
		return nil, nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.applyLanguageLocked(opts.Language); err != nil {
		return nil, err
	}

	var outText *C.char
	var outConf C.float

	rc := C.whisper_stream_process(
		e.stream,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.int32_t(len(samples)),
		&outText,
		&outConf,
	)
	if rc < 0 {
		return nil, fmt.Errorf("whisper: process error (%d)", int(rc))
	}
	if rc == 0 {
		if outText != nil {
			C.whisper_stream_free_text(outText)
		}
		return nil, nil
	}

	text := C.GoString(outText)
	C.whisper_stream_free_text(outText)
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, nil
	}

	conf := float32(outConf)
	e.lastConf = conf

	return []Result{{
		Text:       text,
		Confidence: conf,
		Final:      false,
	}}, nil
}

func (e *NativeEngine) Flush(ctx context.Context, opts Options) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.applyLanguageLocked(opts.Language); err != nil {
		return nil, err
	}

	var outText *C.char
	var outConf C.float

	rc := C.whisper_stream_flush(e.stream, &outText, &outConf)
	if rc < 0 {
		return nil, fmt.Errorf("whisper: flush error (%d)", int(rc))
	}

	text := C.GoString(outText)
	C.whisper_stream_free_text(outText)
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, nil
	}

	conf := float32(outConf)
	e.lastConf = conf

	return []Result{{
		Text:       text,
		Confidence: conf,
		Final:      true,
	}}, nil
}

func (e *NativeEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stream != nil {
		C.whisper_stream_free(e.stream)
		e.stream = nil
	}
	return nil
}

func (e *NativeEngine) SetDefaultLanguage(lang string) {
	e.mu.Lock()
	e.defaultLang = normaliseLanguageCode(lang)
	e.languageConfigured = false
	e.mu.Unlock()
}

func (e *NativeEngine) applyLanguageLocked(lang string) error {
	hint := strings.TrimSpace(lang)
	detect := false

	switch {
	case hint == "":
		hint = e.defaultLang
	case strings.EqualFold(hint, "auto"):
		if e.defaultLang != "" {
			hint = e.defaultLang
		} else {
			hint = ""
		}
	default:
		hint = normaliseLanguageCode(hint)
	}

	if hint == "" {
		detect = true
	}

	if e.languageConfigured && e.lastLanguage == hint && e.lastDetectLanguage == detect {
		return nil
	}

	var cLang *C.char
	if !detect && hint != "" {
		cLang = C.CString(hint)
		defer C.free(unsafe.Pointer(cLang))
	}

	if rc := C.whisper_stream_set_language(e.stream, cLang, C.bool(detect)); rc != 0 {
		return fmt.Errorf("whisper: set language failed (%d)", int(rc))
	}

	e.lastLanguage = hint
	e.lastDetectLanguage = detect
	e.languageConfigured = true
	return nil
}

func normaliseLanguageCode(lang string) string {
	trimmed := strings.TrimSpace(lang)
	if trimmed == "" {
		return ""
	}
	if strings.EqualFold(trimmed, "auto") {
		return ""
	}
	return strings.ToLower(trimmed)
}

func pcmBytesToFloat32(buf []byte) []float32 {
	n := len(buf) / 2
	if n == 0 {
		return nil
	}
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		u := uint16(buf[2*i]) | uint16(buf[2*i+1])<<8
		val := int16(u)
		samples[i] = float32(val) / float32(math.MaxInt16)
	}
	return samples
}

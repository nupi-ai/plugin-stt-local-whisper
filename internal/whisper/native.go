//go:build whispercpp

package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp -I${SRCDIR}/../../third_party/whisper.cpp/include -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/../../third_party/whisper.cpp -I${SRCDIR}/../../third_party/whisper.cpp/include -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/whisper.cpp/build -L${SRCDIR}/../../third_party/whisper.cpp/build/src -Wl,-rpath,${SRCDIR}/../../third_party/whisper.cpp/build/src -lwhisper -lstdc++ -lm

#include "stdlib.h"
#include "include/whisper.h"
*/
import "C"

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"
)

func NativeAvailable() bool { return true }

type NativeEngine struct {
	ctx   *C.struct_whisper_context
	audio []byte
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
	return &NativeEngine{ctx: ctx, audio: make([]byte, 0, 1<<16)}, nil
}

func (e *NativeEngine) TranscribeSegment(ctx context.Context, audio []byte, opts Options) ([]Result, error) {
	if len(audio) == 0 {
		return nil, nil
	}
	e.audio = append(e.audio, audio...)
	return nil, nil
}

func (e *NativeEngine) Flush(ctx context.Context, opts Options) ([]Result, error) {
	if len(e.audio) == 0 {
		return nil, nil
	}
	samples := pcmBytesToFloat32(e.audio)
	if len(samples) == 0 {
		e.audio = e.audio[:0]
		return nil, nil
	}
	cSamples := (*C.float)(unsafe.Pointer(&samples[0]))
	nSamples := C.int(len(samples))
	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.print_progress = C.bool(false)
	params.print_realtime = C.bool(false)
	params.print_timestamps = C.bool(false)
	params.translate = C.bool(false)
	params.no_context = C.bool(true)
	params.single_segment = C.bool(false)
	lang := opts.Language
	if lang == "" {
		lang = "auto"
	}
	cLang := C.CString(lang)
	params.language = cLang
	defer C.free(unsafe.Pointer(cLang))
	if ret := C.whisper_full(e.ctx, params, cSamples, nSamples); ret != 0 {
		e.audio = e.audio[:0]
		return nil, fmt.Errorf("whisper: inference failed with code %d", int(ret))
	}
	results := collectResults(e.ctx)
	e.audio = e.audio[:0]
	return results, nil
}

func (e *NativeEngine) Close() error {
	if e.ctx != nil {
		C.whisper_free(e.ctx)
		e.ctx = nil
	}
	return nil
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

func collectResults(ctx *C.struct_whisper_context) []Result {
	count := int(C.whisper_full_n_segments(ctx))
	if count == 0 {
		return nil
	}
	out := make([]Result, 0, count)
	for i := 0; i < count; i++ {
		text := C.GoString(C.whisper_full_get_segment_text(ctx, C.int(i)))
		out = append(out, Result{Text: text, Confidence: 1.0, Final: true})
	}
	return out
}

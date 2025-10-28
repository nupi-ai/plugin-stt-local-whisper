//go:build !whispercpp

package engine

import "context"

// NativeAvailable reports whether the native whisper backend is compiled in.
func NativeAvailable() bool { return false }

// NewNativeEngine returns an error when the native backend is not built.
func NewNativeEngine(modelPath string) (Engine, error) {
	return nil, ErrNativeEngineUnavailable
}

// NativeEngine is a stub that satisfies the Engine interface when the native backend is absent.
type NativeEngine struct{}

func (e *NativeEngine) TranscribeSegment(ctx context.Context, audio []byte, opts Options) ([]Result, error) {
	return nil, ErrNativeEngineUnavailable
}

func (e *NativeEngine) Flush(ctx context.Context, opts Options) ([]Result, error) {
	return nil, ErrNativeEngineUnavailable
}

func (e *NativeEngine) Close() error { return nil }

func (e *NativeEngine) SetDefaultLanguage(string) {}

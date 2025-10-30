package engine

// NativeOptions configures the native Whisper backend. Fields are optional and
// fall back to environment variables or sensible defaults for auto-detection.
type NativeOptions struct {
	UseGPU         *bool
	FlashAttention *bool
	Threads        *int
}

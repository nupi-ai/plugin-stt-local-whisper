package engine

// NativeOptions configures the native Whisper backend. Fields are optional and
// fall back to environment variables or sensible defaults for auto-detection.
type NativeOptions struct {
	UseGPU         *bool
	FlashAttention *bool
	Threads        *int
	// StepMs configures the hop size for sliding-window inference.
	StepMs *int
	// LengthMs is the total window size before Whisper is invoked (--length).
	LengthMs *int
	// KeepMs is the overlap kept from the previous window (--keep).
	KeepMs *int
	// Translate enables translation from source language to English
	Translate *bool
	// TemperatureInc controls temperature fallback during decoding (0.0 to disable)
	TemperatureInc *float32
	// DisableFallback mirrors --no-fallback; when true temperature fallback is disabled regardless of TemperatureInc.
	DisableFallback *bool
	// BeamSize sets beam search size (1 for greedy sampling, >1 for beam search)
	BeamSize *int
	// AudioCtx sets encoder context size (0 = all audio)
	AudioCtx *int
	// PrintTimestamps enables timestamp output in transcription
	PrintTimestamps *bool
	// PrintSpecial enables special token output
	PrintSpecial *bool
	// KeepContext mirrors --keep-context and toggles prompt reuse between iterations.
	KeepContext *bool
	// UseVAD enables the VAD-driven mode (step_ms <= 0).
	UseVAD *bool
	// VADThreshold configures vad_thold.
	VADThreshold *float32
	// FreqThreshold configures freq_thold.
	FreqThreshold *float32
	// MaxTokens mirrors --max-tokens.
	MaxTokens *int
	// TinyDiarize enables the experimental TinyDiARize feature (--tinydiarize).
	TinyDiarize *bool
}

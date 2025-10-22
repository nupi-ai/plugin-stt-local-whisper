//go:build whispercpp

package engine

import (
	"context"
	"testing"
)

func BenchmarkNativeEngineTranscribeSegment(b *testing.B) {
	if !NativeAvailable() {
		b.Skip("native backend not available")
	}

	engine := openTestNativeEngine(b)
	audio, _ := loadTestAudio(b)
	if len(audio) > 3200 {
		audio = audio[:3200]
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.TranscribeSegment(ctx, audio, Options{Language: "en", Sequence: uint64(i)}); err != nil {
			b.Fatalf("TranscribeSegment failed: %v", err)
		}
	}
}

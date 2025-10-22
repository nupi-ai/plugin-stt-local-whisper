package engine

import (
	"context"
	"testing"
)

func BenchmarkStubEngineTranscribeSegment(b *testing.B) {
	eng := NewStubEngine(nil, "base")
	audio := make([]byte, 1600)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.TranscribeSegment(ctx, audio, Options{Sequence: uint64(i)}); err != nil {
			b.Fatalf("TranscribeSegment failed: %v", err)
		}
	}
	if _, err := eng.Flush(ctx, Options{}); err != nil {
		b.Fatalf("Flush failed: %v", err)
	}
}

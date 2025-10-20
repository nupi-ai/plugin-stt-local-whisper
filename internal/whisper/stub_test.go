package whisper

import (
	"context"
	"testing"
)

func TestStubEngineTranscribeAndFlush(t *testing.T) {
	engine := NewStubEngine(nil, "base")

	results, err := engine.TranscribeSegment(context.Background(), []byte("abcd"), Options{Language: "en", Sequence: 1})
	if err != nil {
		t.Fatalf("TranscribeSegment returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected one result, got %d", len(results))
	}
	if results[0].Final {
		t.Fatalf("expected non-final result")
	}
	if results[0].Text == "" {
		t.Fatalf("expected transcript text")
	}

	flushResults, err := engine.Flush(context.Background(), Options{Final: true})
	if err != nil {
		t.Fatalf("Flush returned error: %v", err)
	}
	if len(flushResults) != 1 {
		t.Fatalf("expected single flush result")
	}
	if !flushResults[0].Final {
		t.Fatalf("expected final result")
	}
	if flushResults[0].Text == "" {
		t.Fatalf("expected flush transcript text")
	}
}

package telemetry

import (
	"io"
	"log/slog"
	"testing"
	"time"
)

func TestRecorderSnapshot(t *testing.T) {
	recorder := NewRecorder(slog.New(slog.NewTextHandler(io.Discard, nil)))
	if snapshot := recorder.Snapshot(); snapshot.TotalStreams != 0 {
		t.Fatalf("expected empty snapshot, got %+v", snapshot)
	}

	stream := recorder.StartStream("session-1", "mic", map[string]string{"source": "test"})
	if stream == nil {
		t.Fatalf("expected stream metrics")
	}

	stream.RecordSegment(1, 160, false)
	stream.RecordTranscript(1, "hello", false)
	stream.RecordSegment(2, 80, true)
	stream.RecordTranscript(2, "hello world", true)
	stream.RecordFlush()

	time.Sleep(5 * time.Millisecond)
	stream.Finish(nil)

	snapshot := recorder.Snapshot()
	if snapshot.TotalStreams != 1 {
		t.Fatalf("unexpected TotalStreams: %d", snapshot.TotalStreams)
	}
	if snapshot.TotalSegments != 2 {
		t.Fatalf("unexpected TotalSegments: %d", snapshot.TotalSegments)
	}
	if snapshot.TotalBytes != 240 {
		t.Fatalf("unexpected TotalBytes: %d", snapshot.TotalBytes)
	}
	if snapshot.TotalTranscripts != 2 {
		t.Fatalf("unexpected TotalTranscripts: %d", snapshot.TotalTranscripts)
	}
	if snapshot.TotalFinalTranscripts != 1 {
		t.Fatalf("unexpected TotalFinalTranscripts: %d", snapshot.TotalFinalTranscripts)
	}
	if snapshot.TotalFlushes != 1 {
		t.Fatalf("unexpected TotalFlushes: %d", snapshot.TotalFlushes)
	}
	if snapshot.ActiveStreams != 0 {
		t.Fatalf("expected zero active streams, got %d", snapshot.ActiveStreams)
	}

	stream.Finish(nil)
	if snapshot2 := recorder.Snapshot(); snapshot2.TotalStreams != 1 {
		t.Fatalf("snapshot changed unexpectedly: %+v", snapshot2)
	}
}

func TestStreamFinishWithError(t *testing.T) {
	recorder := NewRecorder(slog.New(slog.NewTextHandler(io.Discard, nil)))
	stream := recorder.StartStream("s", "mic", nil)
	stream.RecordSegment(1, 10, false)
	stream.RecordFlush()
	stream.Finish(io.EOF)

	snapshot := recorder.Snapshot()
	if snapshot.TotalStreams != 1 {
		t.Fatalf("unexpected streams: %d", snapshot.TotalStreams)
	}
	if snapshot.ActiveStreams != 0 {
		t.Fatalf("expected zero active streams, got %d", snapshot.ActiveStreams)
	}
	if snapshot.TotalFlushes != 1 {
		t.Fatalf("unexpected flushes: %d", snapshot.TotalFlushes)
	}
}

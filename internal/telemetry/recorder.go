package telemetry

import (
	"log/slog"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// Recorder tracks adapter-level telemetry that can be forwarded to the daemon/event bus.
type Recorder struct {
	log *slog.Logger

	totalStreams          atomic.Uint64
	activeStreams         atomic.Int64
	totalSegments         atomic.Uint64
	totalBytes            atomic.Uint64
	totalTranscripts      atomic.Uint64
	totalFinalTranscripts atomic.Uint64
	totalFlushes          atomic.Uint64
}

// Snapshot captures cumulative metrics recorded so far.
type Snapshot struct {
	TotalStreams          uint64
	ActiveStreams         int64
	TotalSegments         uint64
	TotalBytes            uint64
	TotalTranscripts      uint64
	TotalFinalTranscripts uint64
	TotalFlushes          uint64
}

// NewRecorder constructs a Recorder using the provided logger.
func NewRecorder(logger *slog.Logger) *Recorder {
	if logger == nil {
		logger = slog.Default()
	}
	return &Recorder{
		log: logger.With("component", "telemetry.Recorder"),
	}
}

// Snapshot returns an immutable view of the recorder totals.
func (r *Recorder) Snapshot() Snapshot {
	if r == nil {
		return Snapshot{}
	}
	return Snapshot{
		TotalStreams:          r.totalStreams.Load(),
		ActiveStreams:         r.activeStreams.Load(),
		TotalSegments:         r.totalSegments.Load(),
		TotalBytes:            r.totalBytes.Load(),
		TotalTranscripts:      r.totalTranscripts.Load(),
		TotalFinalTranscripts: r.totalFinalTranscripts.Load(),
		TotalFlushes:          r.totalFlushes.Load(),
	}
}

// StreamMetrics accumulates statistics for a single transcription stream.
type StreamMetrics struct {
	recorder *Recorder
	log      *slog.Logger

	sessionID string
	streamID  string
	metadata  map[string]string

	started          time.Time
	segments         int
	bytes            int
	transcripts      int
	finalTranscripts int
	flushes          int
	lastSequence     uint64
	closed           atomic.Bool
}

// StartStream initialises a StreamMetrics instance bound to the recorder.
func (r *Recorder) StartStream(sessionID, streamID string, metadata map[string]string) *StreamMetrics {
	if r == nil {
		return nil
	}

	clonedMetadata := cloneMetadata(metadata)

	streamLogger := r.log.With(
		"session_id", sessionID,
		"stream_id", streamID,
	)
	if len(clonedMetadata) > 0 {
		streamLogger = streamLogger.With("metadata", clonedMetadata)
	}

	r.totalStreams.Add(1)
	r.activeStreams.Add(1)

	return &StreamMetrics{
		recorder: r,
		log:      streamLogger,

		sessionID: sessionID,
		streamID:  streamID,
		metadata:  clonedMetadata,

		started: time.Now(),
	}
}

// RecordSegment updates counters for an incoming audio segment.
func (s *StreamMetrics) RecordSegment(sequence uint64, size int, final bool) {
	if s == nil || size <= 0 {
		return
	}
	s.segments++
	s.bytes += size
	s.lastSequence = sequence
	s.recorder.totalSegments.Add(1)
	s.recorder.totalBytes.Add(uint64(size))

	s.log.Debug("segment received",
		"sequence", sequence,
		"bytes", size,
		"final", final,
	)
}

// RecordTranscript stores statistics for an emitted transcript.
func (s *StreamMetrics) RecordTranscript(sequence uint64, text string, final bool) {
	if s == nil {
		return
	}
	s.transcripts++
	if final {
		s.finalTranscripts++
		s.recorder.totalFinalTranscripts.Add(1)
	}
	s.recorder.totalTranscripts.Add(1)

	s.log.Debug("transcript emitted",
		"sequence", sequence,
		"final", final,
		"chars", len(text),
		"runes", utf8.RuneCountInString(text),
	)
}

// RecordFlush increments counters for a stream flush event.
func (s *StreamMetrics) RecordFlush() {
	if s == nil {
		return
	}
	s.flushes++
	s.recorder.totalFlushes.Add(1)
}

// Finish logs a summary and updates active stream counters.
func (s *StreamMetrics) Finish(err error) {
	if s == nil {
		return
	}
	if !s.closed.CompareAndSwap(false, true) {
		return
	}

	defer s.recorder.activeStreams.Add(-1)

	duration := time.Since(s.started)
	args := []any{
		"duration_ms", duration.Milliseconds(),
		"segments", s.segments,
		"bytes", s.bytes,
		"transcripts", s.transcripts,
		"final_transcripts", s.finalTranscripts,
		"flushes", s.flushes,
	}

	if err != nil {
		s.log.Error("stream completed with error", append(args, "error", err)...)
		return
	}

	s.log.Info("stream completed", args...)
}

func cloneMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

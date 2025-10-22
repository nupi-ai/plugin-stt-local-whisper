package engine

import "context"

// Engine exposes a streaming transcription interface backed by whisper.cpp or a stub implementation.
type Engine interface {
	// TranscribeSegment processes a chunk of audio and may emit zero or more transcripts.
	TranscribeSegment(ctx context.Context, audio []byte, opts Options) ([]Result, error)
	// Flush finalises the transcription session and emits any buffered transcripts.
	Flush(ctx context.Context, opts Options) ([]Result, error)
	// Close releases underlying resources.
	Close() error
}

// Options configures decoding for a segment or flush call.
type Options struct {
	Language string
	// Final indicates whether the segment corresponds to the end of the stream.
	Final bool
	// Sequence carries the original sequence number from the segment, when available.
	Sequence uint64
}

// Result represents a transcript produced by the engine.
type Result struct {
	Text       string
	Confidence float32
	Final      bool
}

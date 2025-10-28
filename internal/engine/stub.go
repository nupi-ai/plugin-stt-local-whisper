package engine

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/adapterinfo"
)

// StubEngine produces deterministic transcripts without invoking Whisper.
type StubEngine struct {
	log          *slog.Logger
	modelVariant string
	totalBytes   int
}

// NewStubEngine returns an Engine that generates placeholder transcripts.
func NewStubEngine(logger *slog.Logger, modelVariant string) *StubEngine {
	if logger == nil {
		logger = slog.Default()
	}
	return &StubEngine{
		log: logger.With(
			"component", "engine.stub",
			"adapter", adapterinfo.Info.Slug,
			"model_variant", modelVariant,
		),
		modelVariant: modelVariant,
	}
}

// Close implements the Engine interface.
func (e *StubEngine) Close() error {
	return nil
}

// TranscribeSegment implements the Engine interface.
func (e *StubEngine) TranscribeSegment(ctx context.Context, audio []byte, opts Options) ([]Result, error) {
	if len(audio) == 0 {
		return nil, nil
	}
	e.totalBytes += len(audio)
	text := fmt.Sprintf("[stub:%s] received %d bytes", e.modelVariant, len(audio))
	e.log.Debug("stub transcript", "bytes", len(audio), "sequence", opts.Sequence, "final", opts.Final)
	return []Result{
		{
			Text:       text,
			Confidence: 0.42,
			Final:      opts.Final,
		},
	}, nil
}

// Flush implements the Engine interface.
func (e *StubEngine) Flush(ctx context.Context, opts Options) ([]Result, error) {
	text := "[stub] stream closed"
	if e.totalBytes > 0 {
		text = fmt.Sprintf("[stub:%s] total bytes %d", e.modelVariant, e.totalBytes)
	}
	e.log.Debug("stub flush", "total_bytes", e.totalBytes)
	e.totalBytes = 0
	return []Result{
		{
			Text:       text,
			Confidence: 1.0,
			Final:      true,
		},
	}, nil
}

// SetDefaultLanguage satisfies the languageHintSetter interface; the stub ignores the hint.
func (e *StubEngine) SetDefaultLanguage(string) {}

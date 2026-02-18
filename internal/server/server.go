package server

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"time"

	napv1 "github.com/nupi-ai/nupi/api/nap/v1"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/adapterinfo"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/config"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/engine"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/telemetry"
)

// Server implements the SpeechToTextService and provides stubbed transcripts.
type Server struct {
	napv1.UnimplementedSpeechToTextServiceServer

	cfg     config.Config
	log     *slog.Logger
	engine  engine.Engine
	metrics *telemetry.Recorder
}

// New returns a new Server instance.
func New(cfg config.Config, logger *slog.Logger, engine engine.Engine, metrics *telemetry.Recorder) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	if engine == nil {
		panic("server: engine must not be nil")
	}
	if metrics == nil {
		metrics = telemetry.NewRecorder(logger)
	}
	return &Server{
		cfg: cfg,
		log: logger.With(
			"component", "server",
			"model_variant", cfg.ModelVariant,
			"language", cfg.Language,
		),
		engine:  engine,
		metrics: metrics,
	}
}

// StreamTranscription consumes PCM segments and emits stub transcripts that
// describe the received payload. This allows exercising the adapter runner and
// daemon integration before Whisper bindings are wired in.
func (s *Server) StreamTranscription(stream napv1.SpeechToTextService_StreamTranscriptionServer) (err error) {
	var (
		initLogged    bool
		streamMetrics *telemetry.StreamMetrics
		sessionID     string
		streamID      string
		lastSequence  uint64
		streamLang    string // effective language for the entire stream
	)
	ctx := stream.Context()
	defer func() {
		if streamMetrics != nil {
			streamMetrics.Finish(err)
		}
	}()

	for {
		req, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, context.Canceled) {
				if streamMetrics != nil {
					flushCtx := ctx
					if errors.Is(err, context.Canceled) {
						var cancel context.CancelFunc
						flushCtx, cancel = context.WithTimeout(context.Background(), 2*time.Second)
						defer cancel()
					}
					if flushErr := s.emitFlush(flushCtx, stream, sessionID, streamID, lastSequence, streamLang, streamMetrics, "stream closed"); flushErr != nil {
						return flushErr
					}
				}
				return nil
			}
			s.log.Error("failed to receive request", "error", err)
			return err
		}
		if req == nil {
			continue
		}

		if !initLogged {
			streamMetrics = s.metrics.StartStream(req.GetSessionId(), req.GetStreamId(), req.GetMetadata())
			streamLang = resolveLanguage(s.cfg.Language, req.GetMetadata())
			s.log.Info("stream opened",
				"session_id", req.GetSessionId(),
				"stream_id", req.GetStreamId(),
				"metadata", req.GetMetadata(),
				"resolved_language", streamLang,
			)
			sessionID = req.GetSessionId()
			streamID = req.GetStreamId()
			initLogged = true
		}

		segment := req.GetSegment()
		var sequence uint64
		if segment != nil {
			sequence = segment.GetSequence()
			lastSequence = sequence
		}

		if segment != nil && len(segment.GetAudio()) > 0 {
			logEntry := s.log.With(
				"session_id", req.GetSessionId(),
				"stream_id", req.GetStreamId(),
				"sequence", sequence,
				"final_requested", req.GetFlush() || segment.GetLast(),
			)
			if streamMetrics != nil {
				streamMetrics.RecordSegment(sequence, len(segment.GetAudio()), req.GetFlush() || segment.GetLast())
			}
			start := time.Now()
			results, err := s.engine.TranscribeSegment(ctx, segment.GetAudio(), engine.Options{
				Language: streamLang,
				Final:    req.GetFlush() || segment.GetLast(),
				Sequence: sequence,
			})
			if err != nil {
				logEntry.Error("engine segment failure", "error", err, "context_err", ctx.Err())
				return err
			}
			if streamMetrics != nil {
				streamMetrics.RecordInferenceDuration(time.Since(start))
			}
			for idx, res := range results {
				logEntry.Info("engine segment result",
					"index", idx,
					"text", res.Text,
					"confidence", res.Confidence,
					"final", res.Final,
				)
			}
			if err := s.sendResults(stream, sequence, results, streamLang, streamMetrics); err != nil {
				return err
			}
		}

		if req.GetFlush() {
			if err := s.emitFlush(ctx, stream, req.GetSessionId(), req.GetStreamId(), sequence, streamLang, streamMetrics, "stream flushed"); err != nil {
				return err
			}
			return nil
		}
	}
}

func (s *Server) emitFlush(
	ctx context.Context,
	stream napv1.SpeechToTextService_StreamTranscriptionServer,
	sessionID, streamID string,
	sequence uint64,
	lang string,
	metrics *telemetry.StreamMetrics,
	reason string,
) error {
	logEntry := s.log.With(
		"session_id", sessionID,
		"stream_id", streamID,
		"sequence", sequence,
	)
	if metrics != nil {
		metrics.RecordFlush()
	}
	start := time.Now()
	results, err := s.engine.Flush(ctx, engine.Options{Language: lang, Final: true})
	if err != nil {
		logEntry.Error("engine flush failure", "error", err, "context_err", ctx.Err())
		return err
	}
	if metrics != nil {
		metrics.RecordInferenceDuration(time.Since(start))
	}
	for idx, res := range results {
		logEntry.Info("engine flush result",
			"index", idx,
			"text", res.Text,
			"confidence", res.Confidence,
			"final", res.Final,
		)
	}
	if err := s.sendResults(stream, sequence, results, lang, metrics); err != nil {
		return err
	}
	logEntry.Info(reason)
	return nil
}

// resolveLanguage determines the effective language for a transcription stream
// based on the configured language mode and request metadata.
//
// The returned value is passed to whisper.cpp which requires an ISO 639-1 code
// (e.g. "pl", "en", "de") or "auto" for auto-detection.
//
//   - "client": use nupi.lang.iso1 from metadata; fall back to "auto" if absent.
//   - "auto": always auto-detect, ignore metadata.
//   - other: ISO 639-1 code from config, passed verbatim (ignore metadata).
func resolveLanguage(configLang string, metadata map[string]string) string {
	if configLang != "client" {
		return configLang
	}
	if code := strings.TrimSpace(metadata["nupi.lang.iso1"]); code != "" {
		return code
	}
	return "auto"
}

func (s *Server) sendResults(stream napv1.SpeechToTextService_StreamTranscriptionServer, sequence uint64, results []engine.Result, resolvedLang string, metrics *telemetry.StreamMetrics) error {
	for _, res := range results {
		if metrics != nil {
			metrics.RecordTranscript(sequence, res.Text, res.Final)
		}
		transcript := &napv1.Transcript{
			Sequence:   sequence,
			Text:       res.Text,
			Confidence: res.Confidence,
			Final:      res.Final,
			Metadata:   adapterinfo.TranscriptMetadata(s.cfg.ModelVariant, resolvedLang),
		}
		if err := stream.Send(transcript); err != nil {
			s.log.Error("failed to send transcript", "error", err)
			return err
		}
	}
	return nil
}

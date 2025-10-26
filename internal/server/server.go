package server

import (
	"io"
	"log/slog"
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
			if err == io.EOF {
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
			s.log.Info("stream opened",
				"session_id", req.GetSessionId(),
				"stream_id", req.GetStreamId(),
				"metadata", req.GetMetadata(),
			)
			initLogged = true
		}

		segment := req.GetSegment()
		var sequence uint64
		if segment != nil {
			sequence = segment.GetSequence()
		}

		if segment != nil && len(segment.GetAudio()) > 0 {
			if streamMetrics != nil {
				streamMetrics.RecordSegment(sequence, len(segment.GetAudio()), req.GetFlush() || segment.GetLast())
			}
			start := time.Now()
			results, err := s.engine.TranscribeSegment(ctx, segment.GetAudio(), engine.Options{
				Language: s.cfg.Language,
				Final:    req.GetFlush() || segment.GetLast(),
				Sequence: sequence,
			})
			if err != nil {
				s.log.Error("engine segment failure", "error", err)
				return err
			}
			if streamMetrics != nil {
				streamMetrics.RecordInferenceDuration(time.Since(start))
			}
			if err := s.sendResults(stream, sequence, results, streamMetrics); err != nil {
				return err
			}
		}

		if req.GetFlush() {
			if streamMetrics != nil {
				streamMetrics.RecordFlush()
			}
			start := time.Now()
			results, err := s.engine.Flush(ctx, engine.Options{Language: s.cfg.Language, Final: true})
			if err != nil {
				s.log.Error("engine flush failure", "error", err)
				return err
			}
			if streamMetrics != nil {
				streamMetrics.RecordInferenceDuration(time.Since(start))
			}
			if err := s.sendResults(stream, sequence, results, streamMetrics); err != nil {
				return err
			}
			s.log.Info("stream flushed",
				"session_id", req.GetSessionId(),
				"stream_id", req.GetStreamId(),
			)
			return nil
		}
	}
}

func (s *Server) sendResults(stream napv1.SpeechToTextService_StreamTranscriptionServer, sequence uint64, results []engine.Result, metrics *telemetry.StreamMetrics) error {
	for _, res := range results {
		if metrics != nil {
			metrics.RecordTranscript(sequence, res.Text, res.Final)
		}
		transcript := &napv1.Transcript{
			Sequence:   sequence,
			Text:       res.Text,
			Confidence: res.Confidence,
			Final:      res.Final,
			Metadata:   adapterinfo.TranscriptMetadata(s.cfg.ModelVariant, s.cfg.Language),
		}
		if err := stream.Send(transcript); err != nil {
			s.log.Error("failed to send transcript", "error", err)
			return err
		}
	}
	return nil
}

package server

import (
	"io"
	"log/slog"

	napv1 "github.com/nupi-ai/nupi/api/nap/v1"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/config"
	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/whisper"
)

// Server implements the SpeechToTextService and provides stubbed transcripts.
type Server struct {
	napv1.UnimplementedSpeechToTextServiceServer

	cfg    config.Config
	log    *slog.Logger
	engine whisper.Engine
}

// New returns a new Server instance.
func New(cfg config.Config, logger *slog.Logger, engine whisper.Engine) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	if engine == nil {
		panic("server: engine must not be nil")
	}
	return &Server{
		cfg: cfg,
		log: logger.With(
			"component", "server",
			"model_variant", cfg.ModelVariant,
			"language", cfg.Language,
		),
		engine: engine,
	}
}

// StreamTranscription consumes PCM segments and emits stub transcripts that
// describe the received payload. This allows exercising the adapter runner and
// daemon integration before Whisper bindings are wired in.
func (s *Server) StreamTranscription(stream napv1.SpeechToTextService_StreamTranscriptionServer) error {
	var (
		initLogged bool
	)
	ctx := stream.Context()

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
			results, err := s.engine.TranscribeSegment(ctx, segment.GetAudio(), whisper.Options{
				Language: s.cfg.Language,
				Final:    req.GetFlush() || segment.GetLast(),
				Sequence: sequence,
			})
			if err != nil {
				s.log.Error("engine segment failure", "error", err)
				return err
			}
			if err := s.sendResults(stream, sequence, results); err != nil {
				return err
			}
		}

		if req.GetFlush() {
			results, err := s.engine.Flush(ctx, whisper.Options{Language: s.cfg.Language, Final: true})
			if err != nil {
				s.log.Error("engine flush failure", "error", err)
				return err
			}
			if err := s.sendResults(stream, sequence, results); err != nil {
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

func (s *Server) sendResults(stream napv1.SpeechToTextService_StreamTranscriptionServer, sequence uint64, results []whisper.Result) error {
	for _, res := range results {
		transcript := &napv1.Transcript{
			Sequence:   sequence,
			Text:       res.Text,
			Confidence: res.Confidence,
			Final:      res.Final,
			Metadata: map[string]string{
				"generator":     "nupi-whisper-local-stt",
				"model_variant": s.cfg.ModelVariant,
				"language":      s.cfg.Language,
			},
		}
		if err := stream.Send(transcript); err != nil {
			s.log.Error("failed to send transcript", "error", err)
			return err
		}
	}
	return nil
}

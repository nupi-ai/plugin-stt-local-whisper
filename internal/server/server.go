package server

import (
	"fmt"
	"io"
	"log/slog"

	napv1 "github.com/nupi-ai/nupi/api/nap/v1"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/config"
)

// Server implements the SpeechToTextService and provides stubbed transcripts.
type Server struct {
	napv1.UnimplementedSpeechToTextServiceServer

	cfg config.Config
	log *slog.Logger
}

// New returns a new Server instance.
func New(cfg config.Config, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{
		cfg: cfg,
		log: logger.With(
			"component", "server",
			"model_variant", cfg.ModelVariant,
			"language", cfg.Language,
		),
	}
}

// StreamTranscription consumes PCM segments and emits stub transcripts that
// describe the received payload. This allows exercising the adapter runner and
// daemon integration before Whisper bindings are wired in.
func (s *Server) StreamTranscription(stream napv1.SpeechToTextService_StreamTranscriptionServer) error {
	var (
		initLogged bool
	)

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
			text := fmt.Sprintf("[stub:%s] received %d bytes", s.cfg.ModelVariant, len(segment.GetAudio()))
			resp := &napv1.Transcript{
				Sequence:   sequence,
				Text:       text,
				Confidence: 0.42,
				Final:      req.GetFlush() || segment.GetLast(),
				Metadata: map[string]string{
					"generator":     "nupi-whisper-local-stt-stub",
					"model_variant": s.cfg.ModelVariant,
					"language":      s.cfg.Language,
				},
			}
			if err := stream.Send(resp); err != nil {
				s.log.Error("failed to send transcript", "error", err)
				return err
			}
		}

		if req.GetFlush() {
			if segment == nil || len(segment.GetAudio()) == 0 {
				resp := &napv1.Transcript{
					Sequence:   sequence,
					Text:       "[stub] stream closed",
					Confidence: 1.0,
					Final:      true,
					Metadata: map[string]string{
						"generator": "nupi-whisper-local-stt-stub",
					},
				}
				if err := stream.Send(resp); err != nil {
					s.log.Error("failed to send flush transcript", "error", err)
					return err
				}
			}
			s.log.Info("stream flushed",
				"session_id", req.GetSessionId(),
				"stream_id", req.GetStreamId(),
			)
			return nil
		}
	}
}

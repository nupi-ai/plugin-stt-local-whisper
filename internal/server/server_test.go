package server_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	napv1 "github.com/nupi-ai/nupi/api/nap/v1"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/config"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/engine"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/server"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/telemetry"
)

const bufSize = 1024 * 1024

func TestStreamTranscriptionStub(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	lis := bufconn.Listen(bufSize)
	defer lis.Close()

	grpcServer := grpc.NewServer()
	t.Cleanup(grpcServer.Stop)

	cfg := config.Config{
		ListenAddr:   "bufconn",
		ModelVariant: "small",
		Language:     "pl",
		LogLevel:     "debug",
	}
	eng := engine.NewStubEngine(slog.New(slog.NewTextHandler(io.Discard, nil)), cfg.ModelVariant)
	recorder := telemetry.NewRecorder(slog.New(slog.NewTextHandler(io.Discard, nil)))
	napv1.RegisterSpeechToTextServiceServer(grpcServer, server.New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), eng, recorder))

	go func() {
		if err := grpcServer.Serve(lis); err != nil &&
			!errors.Is(err, grpc.ErrServerStopped) &&
			!errors.Is(err, net.ErrClosed) &&
			err.Error() != "closed" {
			t.Errorf("Serve() error: %v", err)
		}
	}()

	conn, err := grpc.DialContext(ctx, "bufconn",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("DialContext error: %v", err)
	}
	t.Cleanup(func() { conn.Close() })

	client := napv1.NewSpeechToTextServiceClient(conn)
	stream, err := client.StreamTranscription(ctx)
	if err != nil {
		t.Fatalf("StreamTranscription error: %v", err)
	}

	if err := stream.Send(&napv1.StreamTranscriptionRequest{
		SessionId: "session-1",
		StreamId:  "mic",
		Format: &napv1.AudioFormat{
			Encoding:   "pcm_s16le",
			SampleRate: 16000,
			Channels:   1,
		},
	}); err != nil {
		t.Fatalf("Send init error: %v", err)
	}

	segment := []byte("test")
	if err := stream.Send(&napv1.StreamTranscriptionRequest{
		SessionId: "session-1",
		StreamId:  "mic",
		Segment: &napv1.Segment{
			Sequence: 1,
			Audio:    segment,
		},
	}); err != nil {
		t.Fatalf("Send segment error: %v", err)
	}

	if err := stream.Send(&napv1.StreamTranscriptionRequest{
		SessionId: "session-1",
		StreamId:  "mic",
		Flush:     true,
	}); err != nil {
		t.Fatalf("Send flush error: %v", err)
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("CloseSend error: %v", err)
	}

	resp1, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv 1 error: %v", err)
	}
	if got, want := resp1.GetSequence(), uint64(1); got != want {
		t.Fatalf("unexpected sequence: got %d, want %d", got, want)
	}
	if !strings.Contains(resp1.GetText(), "received 4 bytes") {
		t.Fatalf("unexpected transcript text: %q", resp1.GetText())
	}
	if resp1.GetFinal() {
		t.Fatalf("expected first transcript to be non-final")
	}

	resp2, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv 2 error: %v", err)
	}
	if !resp2.GetFinal() {
		t.Fatalf("expected second transcript to be final")
	}
	if resp2.GetText() != "[stub:small] total bytes 4" {
		t.Fatalf("unexpected final transcript: %q", resp2.GetText())
	}

	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("expected EOF after flush, got %v", err)
	}

	snapshot := recorder.Snapshot()
	if snapshot.TotalStreams != 1 {
		t.Fatalf("unexpected TotalStreams: %d", snapshot.TotalStreams)
	}
	if snapshot.TotalSegments != 1 {
		t.Fatalf("unexpected TotalSegments: %d", snapshot.TotalSegments)
	}
	if snapshot.TotalTranscripts != 2 {
		t.Fatalf("unexpected TotalTranscripts: %d", snapshot.TotalTranscripts)
	}
	if snapshot.TotalFinalTranscripts != 1 {
		t.Fatalf("unexpected TotalFinalTranscripts: %d", snapshot.TotalFinalTranscripts)
	}
}

func TestStreamTranscriptionFlushOnCancel(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	lis := bufconn.Listen(bufSize)
	defer lis.Close()

	grpcServer := grpc.NewServer()
	t.Cleanup(grpcServer.Stop)

	cfg := config.Config{
		ListenAddr:   "bufconn",
		ModelVariant: "small",
		Language:     "pl",
		LogLevel:     "debug",
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	eng := engine.NewStubEngine(logger, cfg.ModelVariant)
	recorder := telemetry.NewRecorder(logger)
	napv1.RegisterSpeechToTextServiceServer(grpcServer, server.New(cfg, logger, eng, recorder))

	go func() {
		if err := grpcServer.Serve(lis); err != nil &&
			!errors.Is(err, grpc.ErrServerStopped) &&
			!errors.Is(err, net.ErrClosed) &&
			err.Error() != "closed" {
			t.Errorf("Serve() error: %v", err)
		}
	}()

	conn, err := grpc.DialContext(ctx, "bufconn",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("DialContext error: %v", err)
	}
	t.Cleanup(func() { conn.Close() })

	client := napv1.NewSpeechToTextServiceClient(conn)
	stream, err := client.StreamTranscription(ctx)
	if err != nil {
		t.Fatalf("StreamTranscription error: %v", err)
	}

	if err := stream.Send(&napv1.StreamTranscriptionRequest{
		SessionId: "session-42",
		StreamId:  "mic",
		Format: &napv1.AudioFormat{
			Encoding:   "pcm_s16le",
			SampleRate: 16000,
			Channels:   1,
		},
	}); err != nil {
		t.Fatalf("Send init error: %v", err)
	}

	if err := stream.Send(&napv1.StreamTranscriptionRequest{
		SessionId: "session-42",
		StreamId:  "mic",
		Segment: &napv1.Segment{
			Sequence: 1,
			Audio:    []byte("abcd"),
		},
	}); err != nil {
		t.Fatalf("Send segment error: %v", err)
	}

	if err := stream.CloseSend(); err != nil {
		t.Fatalf("CloseSend error: %v", err)
	}

	first, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv first transcript: %v", err)
	}
	if first.GetFinal() {
		t.Fatalf("expected non-final transcript, got final: %v", first)
	}

	final, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv final transcript: %v", err)
	}
	if !final.GetFinal() {
		t.Fatalf("expected final transcript")
	}
	if final.GetText() != "[stub:small] total bytes 4" {
		t.Fatalf("unexpected final text: %q", final.GetText())
	}

	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("expected EOF after final transcript, got %v", err)
	}
}

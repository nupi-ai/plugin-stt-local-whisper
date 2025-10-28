package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthgrpc "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	napv1 "github.com/nupi-ai/nupi/api/nap/v1"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/adapterinfo"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/config"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/engine"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/models"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/server"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/telemetry"
)

// lazySTTServer wraps a SpeechToTextServiceServer and allows deferred initialization.
// It returns Unavailable errors until the underlying server is set via setServer.
type lazySTTServer struct {
	napv1.UnimplementedSpeechToTextServiceServer
	server atomic.Pointer[napv1.SpeechToTextServiceServer]
}

func (l *lazySTTServer) setServer(srv napv1.SpeechToTextServiceServer) {
	l.server.Store(&srv)
}

func (l *lazySTTServer) StreamTranscription(stream napv1.SpeechToTextService_StreamTranscriptionServer) error {
	srv := l.server.Load()
	if srv == nil {
		return status.Error(codes.Unavailable, "STT service is initializing, please retry in a moment")
	}
	return (*srv).StreamTranscription(stream)
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	cfg, err := config.Loader{}.Load()
	if err != nil {
		slog.Error("failed to load configuration", "error", err)
		os.Exit(1)
	}

	logger := newLogger(cfg.LogLevel)
	logger.Info("starting adapter",
		"adapter", adapterinfo.Info.Name,
		"adapter_slug", adapterinfo.Info.Slug,
		"listen_addr", cfg.ListenAddr,
		"model_variant", cfg.ModelVariant,
		"language", cfg.Language,
		"data_dir", cfg.DataDir,
	)

	recorder := telemetry.NewRecorder(logger)

	// STEP 1: Bind port IMMEDIATELY (before loading model)
	// This allows the manager's readiness check to succeed while model loads in background.
	lis, err := net.Listen("tcp", cfg.ListenAddr)
	if err != nil {
		logger.Error("failed to bind listener", "error", err)
		os.Exit(1)
	}
	defer lis.Close()
	logger.Info("listener bound, port ready", "addr", lis.Addr().String())

	// STEP 2: Setup gRPC server with lazy STT service wrapper
	grpcServer := grpc.NewServer()
	healthServer := health.NewServer()
	healthgrpc.RegisterHealthServer(grpcServer, healthServer)

	serviceName := napv1.SpeechToTextService_ServiceDesc.ServiceName
	healthServer.SetServingStatus("", healthgrpc.HealthCheckResponse_NOT_SERVING)
	healthServer.SetServingStatus(serviceName, healthgrpc.HealthCheckResponse_NOT_SERVING)

	lazyService := &lazySTTServer{}
	napv1.RegisterSpeechToTextServiceServer(grpcServer, lazyService)

	// STEP 3: Start gRPC server in background (port is already bound)
	serverErr := make(chan error, 1)
	go func() {
		if err := grpcServer.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			serverErr <- err
		}
	}()
	logger.Info("gRPC server started (NOT_SERVING while initializing)")

	// STEP 4: Load model (can take 10+ seconds, but port is already available for readiness checks!)
	manager, err := models.NewManager(cfg.DataDir, logger)
	if err != nil {
		logger.Error("failed to initialise model manager", "error", err)
		grpcServer.Stop()
		os.Exit(1)
	}

	eng, modelPath, engineErr := engine.New(cfg, manager, logger)
	if engineErr != nil {
		logger.Warn("engine initialised with warnings", "error", engineErr)
	}
	if modelPath != "" {
		logger.Info("resolved model path", "path", modelPath)
	}
	defer func() {
		if err := eng.Close(); err != nil {
			logger.Warn("failed to close engine", "error", err)
		}
	}()

	// STEP 5: Activate the real STT service now that engine is ready
	realService := server.New(cfg, logger, eng, recorder)
	lazyService.setServer(realService)

	healthServer.SetServingStatus("", healthgrpc.HealthCheckResponse_SERVING)
	healthServer.SetServingStatus(serviceName, healthgrpc.HealthCheckResponse_SERVING)
	logger.Info("adapter ready to serve requests")

	// STEP 6: Setup graceful shutdown
	go func() {
		<-ctx.Done()
		logger.Info("shutdown requested, stopping gRPC server")
		healthServer.SetServingStatus(serviceName, healthgrpc.HealthCheckResponse_NOT_SERVING)
		healthServer.SetServingStatus("", healthgrpc.HealthCheckResponse_NOT_SERVING)

		stopped := make(chan struct{})
		go func() {
			grpcServer.GracefulStop()
			close(stopped)
		}()

		select {
		case <-stopped:
		case <-time.After(5 * time.Second):
			logger.Warn("graceful stop timed out, forcing stop")
			grpcServer.Stop()
		}
	}()

	// STEP 7: Wait for server to finish or error
	select {
	case err := <-serverErr:
		logger.Error("gRPC server terminated with error", "error", err)
		os.Exit(1)
	case <-ctx.Done():
		// Normal shutdown via signal
	}

	if snapshot := recorder.Snapshot(); snapshot.TotalStreams > 0 {
		logger.Info("telemetry totals",
			"total_streams", snapshot.TotalStreams,
			"total_segments", snapshot.TotalSegments,
			"total_transcripts", snapshot.TotalTranscripts,
			"total_final_transcripts", snapshot.TotalFinalTranscripts,
			"total_bytes", snapshot.TotalBytes,
			"total_flushes", snapshot.TotalFlushes,
		)
	}

	logger.Info("adapter stopped")
}

func newLogger(level string) *slog.Logger {
	handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: parseLevel(level),
	})
	return slog.New(handler)
}

func parseLevel(value string) slog.Leveler {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "debug":
		return slog.LevelDebug
	case "info", "":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"

	napv1 "github.com/nupi-ai/nupi/api/nap/v1"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/config"
	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/server"
)

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
		"listen_addr", cfg.ListenAddr,
		"model_variant", cfg.ModelVariant,
		"language", cfg.Language,
	)

	lis, err := net.Listen("tcp", cfg.ListenAddr)
	if err != nil {
		logger.Error("failed to bind listener", "error", err)
		os.Exit(1)
	}
	defer lis.Close()

	grpcServer := grpc.NewServer()
	napv1.RegisterSpeechToTextServiceServer(grpcServer, server.New(cfg, logger))

	go func() {
		<-ctx.Done()
		logger.Info("shutdown requested, stopping gRPC server")

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

	if err := grpcServer.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
		logger.Error("gRPC server terminated with error", "error", err)
		os.Exit(1)
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

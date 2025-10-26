package engine

import (
	"context"
	"errors"
	"strings"

	"log/slog"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/config"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/models"
)

// ErrNativeEngineUnavailable indicates that a real native backend is not yet wired in.
var ErrNativeEngineUnavailable = errors.New("engine: native backend unavailable")

// New resolves the desired model and returns an Engine instance.
// Currently the implementation falls back to the stub engine when the native backend
// is unavailable or model artefacts cannot be ensured locally.
func New(cfg config.Config, manager *models.Manager, logger *slog.Logger) (Engine, string, error) {
	manifest, err := models.DefaultManifest()
	if err != nil {
		return newEngineWithOptions(cfg, manager, logger, engineOptions{})
	}

	return newEngineWithOptions(cfg, manager, logger, engineOptions{
		manifest: manifest,
		ensure: models.EnsureOptions{
			Manifest: manifest,
			Override: cfg.ModelPath,
		},
	})
}

type engineOptions struct {
	manifest models.Manifest
	ensure   models.EnsureOptions
}

func newEngineWithOptions(cfg config.Config, manager *models.Manager, logger *slog.Logger, opts engineOptions) (Engine, string, error) {
	if logger == nil {
		logger = slog.Default()
	}

	if cfg.UseStubEngine {
		path := ""
		if manager != nil && strings.TrimSpace(cfg.ModelPath) != "" {
			resolved, err := manager.Resolve(cfg.ModelVariant, cfg.ModelPath)
			if err != nil {
				return NewStubEngine(logger, cfg.ModelVariant), "", err
			}
			path = resolved
		}
		logger.Warn("stub engine forced by configuration")
		return NewStubEngine(logger, cfg.ModelVariant), path, nil
	}

	if manager == nil {
		logger.Warn("model manager unavailable; using stub engine")
		return NewStubEngine(logger, cfg.ModelVariant), "", ErrNativeEngineUnavailable
	}

	if opts.ensure.Manifest.Variants == nil || len(opts.ensure.Manifest.Variants) == 0 {
		return NewStubEngine(logger, cfg.ModelVariant), "", errors.New("models: manifest is empty")
	}

	modelPath, err := manager.EnsureVariant(context.Background(), cfg.ModelVariant, opts.ensure)
	if err != nil {
		logger.Warn("model ensure failed; using stub engine", "error", err)
		return NewStubEngine(logger, cfg.ModelVariant), "", err
	}

	if NativeAvailable() {
		native, nativeErr := NewNativeEngine(modelPath)
		if nativeErr != nil {
			logger.Error("native engine initialisation failed; using stub", "error", nativeErr, "model_path", modelPath)
			return NewStubEngine(logger, cfg.ModelVariant), modelPath, nativeErr
		}
		logger.Info("native engine ready", "model_path", modelPath)
		return native, modelPath, nil
	}

	logger.Warn("native backend disabled at build time; using stub engine", "model_path", modelPath)
	return NewStubEngine(logger, cfg.ModelVariant), modelPath, ErrNativeEngineUnavailable
}

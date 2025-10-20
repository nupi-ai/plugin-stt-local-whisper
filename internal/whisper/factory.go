package whisper

import (
	"errors"
	"log/slog"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/config"
	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/models"
)

// ErrNativeEngineUnavailable indicates that a real whisper.cpp backend is not yet wired in.
var ErrNativeEngineUnavailable = errors.New("whisper: native engine unavailable")

// NewEngine resolves the desired model and returns an Engine instance.
// Currently the implementation falls back to the stub engine; once whisper.cpp bindings
// land, this function will instantiate the real backend.
func NewEngine(cfg config.Config, manager *models.Manager, logger *slog.Logger) (Engine, string, error) {
	if logger == nil {
		logger = slog.Default()
	}

	if cfg.UseStubEngine {
		logger.Warn("stub engine forced by configuration")
		return NewStubEngine(logger, cfg.ModelVariant), "", nil
	}

	if manager == nil {
		logger.Warn("model manager unavailable; using stub engine")
		return NewStubEngine(logger, cfg.ModelVariant), "", ErrNativeEngineUnavailable
	}

	modelPath, err := manager.Resolve(cfg.ModelVariant, cfg.ModelPath)
	if err != nil {
		logger.Warn("model resolution failed; using stub engine", "error", err)
		return NewStubEngine(logger, cfg.ModelVariant), "", err
	}

	logger.Warn("whisper.cpp backend not yet implemented; using stub engine", "model_path", modelPath)
	return NewStubEngine(logger, cfg.ModelVariant), modelPath, ErrNativeEngineUnavailable
}

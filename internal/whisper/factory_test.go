package whisper

import (
	"errors"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/config"
	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/models"
)

func TestNewEngineUsesStubWhenForced(t *testing.T) {
	cfg := config.Config{ModelVariant: "base", UseStubEngine: true}
	engine, modelPath, err := NewEngine(cfg, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if modelPath != "" {
		t.Fatalf("expected empty model path, got %q", modelPath)
	}
	if _, ok := engine.(*StubEngine); !ok {
		t.Fatalf("expected stub engine")
	}
}

func TestNewEngineFallsBackWhenModelMissing(t *testing.T) {
	tempDir := t.TempDir()
	manager, err := models.NewManager(tempDir, nil)
	if err != nil {
		t.Fatalf("NewManager error: %v", err)
	}

	cfg := config.Config{ModelVariant: "base"}
	engine, modelPath, err := NewEngine(cfg, manager, nil)
	if err == nil {
		t.Fatalf("expected error due to missing model")
	}
	if modelPath != "" {
		t.Fatalf("expected empty model path")
	}
	if _, ok := engine.(*StubEngine); !ok {
		t.Fatalf("expected stub engine")
	}
}

func TestNewEngineResolvesModel(t *testing.T) {
	tempDir := t.TempDir()
	manager, err := models.NewManager(tempDir, nil)
	if err != nil {
		t.Fatalf("NewManager error: %v", err)
	}

	path := manager.ModelsDir() + "/base.gguf"
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}

	cfg := config.Config{ModelVariant: "base"}
	engine, modelPath, err := NewEngine(cfg, manager, nil)
	if !errors.Is(err, ErrNativeEngineUnavailable) {
		t.Fatalf("expected ErrNativeEngineUnavailable, got %v", err)
	}
	if modelPath != path {
		t.Fatalf("unexpected model path: want %s, got %s", path, modelPath)
	}
	if _, ok := engine.(*StubEngine); !ok {
		t.Fatalf("expected stub engine")
	}
}

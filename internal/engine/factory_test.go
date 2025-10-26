package engine

import (
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/config"
	"github.com/nupi-ai/plugin-stt-local-whisper/internal/models"
)

func TestNewUsesStubWhenForced(t *testing.T) {
	cfg := config.Config{ModelVariant: "base", UseStubEngine: true}
	engine, modelPath, err := New(cfg, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
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

func TestNewFallsBackWhenModelMissing(t *testing.T) {
	tempDir := t.TempDir()
	manager, err := models.NewManager(tempDir, nil)
	if err != nil {
		t.Fatalf("NewManager error: %v", err)
	}

	cfg := config.Config{
		ModelVariant:  "base",
		ModelPath:     filepath.Join(tempDir, "missing.gguf"),
		UseStubEngine: true,
	}
	emptyManifest := models.Manifest{Variants: map[string]models.Variant{}}
	engine, modelPath, err := newEngineWithOptions(cfg, manager, nil, engineOptions{
		manifest: emptyManifest,
		ensure: models.EnsureOptions{
			Manifest: emptyManifest,
			Override: cfg.ModelPath,
		},
	})
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

func TestNewResolvesModel(t *testing.T) {
	tempDir := t.TempDir()
	manager, err := models.NewManager(tempDir, nil)
	if err != nil {
		t.Fatalf("NewManager error: %v", err)
	}

	path := manager.ModelsDir() + "/ggml-base.en.bin"
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}

	cfg := config.Config{ModelVariant: "base"}
	manifest := models.Manifest{Variants: map[string]models.Variant{
		"base": {
			DisplayName: "Base",
			Filename:    "ggml-base.en.bin",
			URL:         "",
		},
	}}
	if NativeAvailable() {
		cfg.UseStubEngine = true
	}
	engine, modelPath, err := newEngineWithOptions(cfg, manager, nil, engineOptions{
		manifest: manifest,
		ensure: models.EnsureOptions{
			Manifest: manifest,
			Override: cfg.ModelPath,
		},
	})
	if NativeAvailable() && cfg.UseStubEngine {
		if err != nil {
			t.Fatalf("expected stub initialisation, got %v", err)
		}
		if modelPath != "" {
			t.Fatalf("expected empty model path when stub forced, got %s", modelPath)
		}
	} else if NativeAvailable() {
		if err != nil {
			t.Fatalf("expected native engine initialisation, got %v", err)
		}
		if modelPath != path {
			t.Fatalf("unexpected model path: want %s, got %s", path, modelPath)
		}
	} else {
		if !errors.Is(err, ErrNativeEngineUnavailable) && err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, ok := engine.(*StubEngine); !ok {
			t.Fatalf("expected stub engine when native unavailable")
		}
	}
}

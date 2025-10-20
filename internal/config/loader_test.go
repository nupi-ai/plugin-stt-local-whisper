package config_test

import (
	"testing"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/config"
)

func TestLoaderDefaults(t *testing.T) {
	loader := config.Loader{}
	cfg, err := loader.Load()
	if err != nil {
		t.Fatalf("Load() returned error: %v", err)
	}
	if cfg.ListenAddr != config.DefaultListenAddr {
		t.Fatalf("expected listen addr %q, got %q", config.DefaultListenAddr, cfg.ListenAddr)
	}
	if cfg.ModelVariant != config.DefaultModel {
		t.Fatalf("expected model variant %q, got %q", config.DefaultModel, cfg.ModelVariant)
	}
	if cfg.Language != config.DefaultLanguage {
		t.Fatalf("expected language %q, got %q", config.DefaultLanguage, cfg.Language)
	}
	if cfg.LogLevel != config.DefaultLogLevel {
		t.Fatalf("expected log level %q, got %q", config.DefaultLogLevel, cfg.LogLevel)
	}
}

func TestLoaderOverrides(t *testing.T) {
	env := map[string]string{
		"NUPI_MODULE_CONFIG":       `{"model_variant":"small","language":"pl","log_level":"debug"}`,
		"NUPI_ADAPTER_LISTEN_ADDR": "0.0.0.0:6000",
		"NUPI_LOG_LEVEL":           "warn",
		"NUPI_MODEL_VARIANT":       "medium",
		"NUPI_LANGUAGE_HINT":       "en",
	}

	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			value, ok := env[key]
			return value, ok
		},
	}

	cfg, err := loader.Load()
	if err != nil {
		t.Fatalf("Load() returned error: %v", err)
	}

	assertEqual(t, "0.0.0.0:6000", cfg.ListenAddr, "listen addr")
	assertEqual(t, "medium", cfg.ModelVariant, "model variant")
	assertEqual(t, "en", cfg.Language, "language")
	assertEqual(t, "warn", cfg.LogLevel, "log level")
}

func TestLoaderInvalidJSON(t *testing.T) {
	env := map[string]string{
		"NUPI_MODULE_CONFIG": "{invalid-json",
	}
	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			value, ok := env[key]
			return value, ok
		},
	}

	if _, err := loader.Load(); err == nil {
		t.Fatalf("expected error for invalid JSON")
	}
}

func assertEqual(t *testing.T, want, got, label string) {
	t.Helper()
	if want != got {
		t.Fatalf("unexpected %s: want %q, got %q", label, want, got)
	}
}

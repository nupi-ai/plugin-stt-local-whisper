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
	if cfg.DataDir != config.DefaultDataDir {
		t.Fatalf("expected data dir %q, got %q", config.DefaultDataDir, cfg.DataDir)
	}
	if cfg.ModelPath != "" {
		t.Fatalf("expected empty model path, got %q", cfg.ModelPath)
	}
	if cfg.UseStubEngine {
		t.Fatalf("expected stub engine disabled by default")
	}
}

func TestLoaderOverrides(t *testing.T) {
	env := map[string]string{
		"NUPI_MODULE_CONFIG":       `{"model_variant":"small","language":"pl","log_level":"debug","data_dir":"/tmp/data","model_path":"/tmp/models/custom.gguf","use_stub_engine":false}`,
		"NUPI_ADAPTER_LISTEN_ADDR": "0.0.0.0:6000",
		"NUPI_LOG_LEVEL":           "warn",
		"NUPI_MODEL_VARIANT":       "medium",
		"NUPI_LANGUAGE_HINT":       "en",
		"NUPI_MODULE_DATA_DIR":     "/var/lib/nupi",
		"NUPI_MODEL_PATH":          "/var/lib/nupi/models/medium.gguf",
		"NUPI_WHISPER_STUB":        "true",
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
	assertEqual(t, "/var/lib/nupi", cfg.DataDir, "data dir")
	assertEqual(t, "/var/lib/nupi/models/medium.gguf", cfg.ModelPath, "model path")
	assertBool(t, true, cfg.UseStubEngine, "use stub engine")
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

func assertBool(t *testing.T, want, got bool, label string) {
	t.Helper()
	if want != got {
		t.Fatalf("unexpected %s: want %v, got %v", label, want, got)
	}
}

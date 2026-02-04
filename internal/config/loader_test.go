package config_test

import (
	"strings"
	"testing"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/config"
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
	if cfg.UseGPU != nil {
		t.Fatalf("expected use_gpu default (nil), got %v", cfg.UseGPU)
	}
	if cfg.FlashAttention != nil {
		t.Fatalf("expected flash_attention default (nil), got %v", cfg.FlashAttention)
	}
	if cfg.Threads != nil {
		t.Fatalf("expected threads default (nil), got %v", *cfg.Threads)
	}
}

func TestLoaderOverrides(t *testing.T) {
	env := map[string]string{
		"NUPI_ADAPTER_CONFIG":          `{"model_variant":"small","language":"pl","log_level":"debug","data_dir":"/tmp/data","model_path":"/tmp/models/custom.gguf","use_stub_engine":false,"use_gpu":false,"flash_attention":true,"threads":4}`,
		"NUPI_ADAPTER_LISTEN_ADDR":     "0.0.0.0:6000",
		"NUPI_LOG_LEVEL":               "warn",
		"NUPI_MODEL_VARIANT":           "medium",
		"NUPI_LANGUAGE_HINT":           "en",
		"NUPI_ADAPTER_DATA_DIR":        "/var/lib/nupi",
		"NUPI_MODEL_PATH":              "/var/lib/nupi/models/medium.gguf",
		"NUPI_ADAPTER_USE_STUB_ENGINE": "true",
		"WHISPERCPP_USE_GPU":           "true",
		"WHISPERCPP_FLASH_ATTENTION":   "false",
		"WHISPERCPP_THREADS":           "6",
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
	assertBoolPtr(t, true, cfg.UseGPU, "use gpu")
	assertBoolPtr(t, false, cfg.FlashAttention, "flash attention")
	assertIntPtr(t, 6, cfg.Threads, "threads")
}

func TestLoaderThreadsAuto(t *testing.T) {
	env := map[string]string{
		"NUPI_ADAPTER_CONFIG": `{"threads":0}`,
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

	if cfg.Threads != nil {
		t.Fatalf("expected threads nil when configured as 0, got %v", *cfg.Threads)
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

func assertBoolPtr(t *testing.T, want bool, got *bool, label string) {
	t.Helper()
	if got == nil {
		t.Fatalf("unexpected %s: want %v, got nil", label, want)
	}
	if *got != want {
		t.Fatalf("unexpected %s: want %v, got %v", label, want, *got)
	}
}

func assertIntPtr(t *testing.T, want int, got *int, label string) {
	t.Helper()
	if got == nil {
		t.Fatalf("unexpected %s: want %d, got nil", label, want)
	}
	if *got != want {
		t.Fatalf("unexpected %s: want %d, got %d", label, want, *got)
	}
}

func TestLoaderInvalidEnvBool(t *testing.T) {
	env := map[string]string{
		"WHISPERCPP_USE_GPU": "not_bool",
	}
	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			v, ok := env[key]
			return v, ok
		},
	}
	_, err := loader.Load()
	if err == nil {
		t.Fatal("expected error for invalid bool env value")
	}
	if !strings.Contains(err.Error(), "WHISPERCPP_USE_GPU") {
		t.Errorf("error should mention env var name, got: %v", err)
	}
}

func TestLoaderInvalidEnvFlashAttention(t *testing.T) {
	env := map[string]string{
		"WHISPERCPP_FLASH_ATTENTION": "abc",
	}
	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			v, ok := env[key]
			return v, ok
		},
	}
	_, err := loader.Load()
	if err == nil {
		t.Fatal("expected error for invalid flash_attention env value")
	}
	if !strings.Contains(err.Error(), "WHISPERCPP_FLASH_ATTENTION") {
		t.Errorf("error should mention env var name, got: %v", err)
	}
}

func TestLoaderInvalidEnvThreads(t *testing.T) {
	env := map[string]string{
		"WHISPERCPP_THREADS": "xyz",
	}
	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			v, ok := env[key]
			return v, ok
		},
	}
	_, err := loader.Load()
	if err == nil {
		t.Fatal("expected error for invalid threads env value")
	}
	if !strings.Contains(err.Error(), "WHISPERCPP_THREADS") {
		t.Errorf("error should mention env var name, got: %v", err)
	}
}

func TestLoaderInvalidStubEngineBool(t *testing.T) {
	env := map[string]string{
		"NUPI_ADAPTER_USE_STUB_ENGINE": "not_a_bool",
	}
	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			v, ok := env[key]
			return v, ok
		},
	}
	_, err := loader.Load()
	if err == nil {
		t.Fatal("expected error for invalid stub engine bool value")
	}
	if !strings.Contains(err.Error(), "NUPI_ADAPTER_USE_STUB_ENGINE") {
		t.Errorf("error should mention env var name, got: %v", err)
	}
}

func TestLoaderEmptyEnvVarsSkipped(t *testing.T) {
	env := map[string]string{
		"NUPI_ADAPTER_USE_STUB_ENGINE": "",
		"WHISPERCPP_USE_GPU":           "",
		"WHISPERCPP_FLASH_ATTENTION":   "  ",
		"WHISPERCPP_THREADS":           "",
	}
	loader := config.Loader{
		Lookup: func(key string) (string, bool) {
			v, ok := env[key]
			return v, ok
		},
	}
	cfg, err := loader.Load()
	if err != nil {
		t.Fatalf("empty env vars should be skipped, got: %v", err)
	}
	if cfg.UseStubEngine {
		t.Errorf("UseStubEngine should remain false for empty env")
	}
	if cfg.UseGPU != nil {
		t.Errorf("UseGPU should remain nil for empty env, got %v", *cfg.UseGPU)
	}
	if cfg.FlashAttention != nil {
		t.Errorf("FlashAttention should remain nil for empty env, got %v", *cfg.FlashAttention)
	}
	if cfg.Threads != nil {
		t.Errorf("Threads should remain nil for empty env, got %v", *cfg.Threads)
	}
}

func TestValidateNegativeThreads(t *testing.T) {
	// assignIntPtr normalizes <=0 to nil during Load, so we test Validate
	// directly to ensure defense-in-depth works if a future code path
	// bypasses assignIntPtr.
	negOne := -1
	cfg := config.Config{
		ListenAddr: "localhost:0",
		Threads:    &negOne,
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for negative threads")
	}
	if !strings.Contains(err.Error(), "threads") {
		t.Errorf("error should mention threads, got: %v", err)
	}
}

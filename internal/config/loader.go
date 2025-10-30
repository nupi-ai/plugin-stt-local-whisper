package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Loader loads configuration from environment variables. Tests can override
// Lookup to inject deterministic maps.
type Loader struct {
	Lookup func(string) (string, bool)
}

// Load retrieves the adapter configuration from environment variables and
// validates it.
func (l Loader) Load() (Config, error) {
	if l.Lookup == nil {
		l.Lookup = os.LookupEnv
	}

	cfg := Config{
		ListenAddr: DefaultListenAddr,
		DataDir:    DefaultDataDir,
	}

	if raw, ok := l.Lookup("NUPI_ADAPTER_CONFIG"); ok && strings.TrimSpace(raw) != "" {
		if err := applyJSON(raw, &cfg); err != nil {
			return Config{}, err
		}
	}

	overrideString(l.Lookup, "NUPI_ADAPTER_LISTEN_ADDR", &cfg.ListenAddr)
	overrideString(l.Lookup, "NUPI_LOG_LEVEL", &cfg.LogLevel)
	overrideString(l.Lookup, "NUPI_MODEL_VARIANT", &cfg.ModelVariant)
	overrideString(l.Lookup, "NUPI_LANGUAGE_HINT", &cfg.Language)
	overrideString(l.Lookup, "NUPI_ADAPTER_DATA_DIR", &cfg.DataDir)
	overrideString(l.Lookup, "NUPI_MODEL_PATH", &cfg.ModelPath)
	overrideBool(l.Lookup, "NUPI_ADAPTER_USE_STUB_ENGINE", &cfg.UseStubEngine)
	if value, ok := l.Lookup("WHISPERCPP_USE_GPU"); ok {
		if parsed, err := parseBool(value); err == nil {
			assignBoolPtr(&cfg.UseGPU, parsed)
		}
	}
	if value, ok := l.Lookup("WHISPERCPP_FLASH_ATTENTION"); ok {
		if parsed, err := parseBool(value); err == nil {
			assignBoolPtr(&cfg.FlashAttention, parsed)
		}
	}
	if value, ok := l.Lookup("WHISPERCPP_THREADS"); ok {
		if parsed, err := parseInt(value); err == nil {
			assignIntPtr(&cfg.Threads, parsed)
		}
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func applyJSON(raw string, cfg *Config) error {
	type jsonConfig struct {
		ListenAddr     string `json:"listen_addr"`
		ModelVariant   string `json:"model_variant"`
		Language       string `json:"language"`
		LogLevel       string `json:"log_level"`
		DataDir        string `json:"data_dir"`
		ModelPath      string `json:"model_path"`
		UseStubEngine  *bool  `json:"use_stub_engine"`
		UseGPU         *bool  `json:"use_gpu"`
		FlashAttention *bool  `json:"flash_attention"`
		Threads        *int   `json:"threads"`
	}
	var payload jsonConfig
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return fmt.Errorf("config: decode NUPI_ADAPTER_CONFIG: %w", err)
	}
	if payload.ListenAddr != "" {
		cfg.ListenAddr = payload.ListenAddr
	}
	if payload.ModelVariant != "" {
		cfg.ModelVariant = payload.ModelVariant
	}
	if payload.Language != "" {
		cfg.Language = payload.Language
	}
	if payload.LogLevel != "" {
		cfg.LogLevel = payload.LogLevel
	}
	if payload.DataDir != "" {
		cfg.DataDir = payload.DataDir
	}
	if payload.ModelPath != "" {
		cfg.ModelPath = payload.ModelPath
	}
	if payload.UseStubEngine != nil {
		cfg.UseStubEngine = *payload.UseStubEngine
	}
	if payload.UseGPU != nil {
		assignBoolPtr(&cfg.UseGPU, *payload.UseGPU)
	}
	if payload.FlashAttention != nil {
		assignBoolPtr(&cfg.FlashAttention, *payload.FlashAttention)
	}
	if payload.Threads != nil {
		assignIntPtr(&cfg.Threads, *payload.Threads)
	}
	return nil
}

func overrideString(lookup func(string) (string, bool), key string, target *string) {
	if lookup == nil || target == nil {
		return
	}
	if value, ok := lookup(key); ok && strings.TrimSpace(value) != "" {
		*target = strings.TrimSpace(value)
	}
}

func overrideBool(lookup func(string) (string, bool), key string, target *bool) {
	if lookup == nil || target == nil {
		return
	}
	if value, ok := lookup(key); ok {
		if parsed, err := parseBool(value); err == nil {
			*target = parsed
		}
	}
}

func parseBool(value string) (bool, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return false, fmt.Errorf("empty")
	}
	parsed, err := strconv.ParseBool(trimmed)
	if err != nil {
		return false, err
	}
	return parsed, nil
}

func parseInt(value string) (int, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return 0, fmt.Errorf("empty")
	}
	parsed, err := strconv.Atoi(trimmed)
	if err != nil {
		return 0, err
	}
	return parsed, nil
}

func assignBoolPtr(target **bool, value bool) {
	v := value
	*target = &v
}

func assignIntPtr(target **int, value int) {
	if value <= 0 {
		*target = nil
		return
	}
	v := value
	*target = &v
}

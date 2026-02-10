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
	if err := overrideBool(l.Lookup, "NUPI_ADAPTER_USE_STUB_ENGINE", &cfg.UseStubEngine); err != nil {
		return Config{}, err
	}
	if value, ok := l.Lookup("WHISPERCPP_USE_GPU"); ok && strings.TrimSpace(value) != "" {
		parsed, err := parseBool(value)
		if err != nil {
			return Config{}, fmt.Errorf("config: invalid value for WHISPERCPP_USE_GPU: %w", err)
		}
		assignBoolPtr(&cfg.UseGPU, parsed)
	}
	if value, ok := l.Lookup("WHISPERCPP_FLASH_ATTENTION"); ok && strings.TrimSpace(value) != "" {
		parsed, err := parseBool(value)
		if err != nil {
			return Config{}, fmt.Errorf("config: invalid value for WHISPERCPP_FLASH_ATTENTION: %w", err)
		}
		assignBoolPtr(&cfg.FlashAttention, parsed)
	}
	if value, ok := l.Lookup("WHISPERCPP_THREADS"); ok && strings.TrimSpace(value) != "" {
		parsed, err := parseInt(value)
		if err != nil {
			return Config{}, fmt.Errorf("config: invalid value for WHISPERCPP_THREADS: %w", err)
		}
		assignIntPtr(&cfg.Threads, parsed)
	}
	if value, ok := l.Lookup("WHISPERCPP_BEAM_SIZE"); ok && strings.TrimSpace(value) != "" {
		parsed, err := parseInt(value)
		if err != nil {
			return Config{}, fmt.Errorf("config: invalid value for WHISPERCPP_BEAM_SIZE: %w", err)
		}
		assignIntPtr(&cfg.BeamSize, parsed)
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
		BeamSize       *int   `json:"beam_size"`
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
	if payload.BeamSize != nil {
		assignIntPtr(&cfg.BeamSize, *payload.BeamSize)
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

func overrideBool(lookup func(string) (string, bool), key string, target *bool) error {
	if lookup == nil || target == nil {
		return nil
	}
	if value, ok := lookup(key); ok && strings.TrimSpace(value) != "" {
		parsed, err := parseBool(value)
		if err != nil {
			return fmt.Errorf("config: invalid value for %s: %w", key, err)
		}
		*target = parsed
	}
	return nil
}

func parseBool(value string) (bool, error) {
	return strconv.ParseBool(strings.TrimSpace(value))
}

func parseInt(value string) (int, error) {
	return strconv.Atoi(strings.TrimSpace(value))
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

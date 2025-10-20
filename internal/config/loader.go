package config

import (
	"encoding/json"
	"fmt"
	"os"
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
	}

	if raw, ok := l.Lookup("NUPI_MODULE_CONFIG"); ok && strings.TrimSpace(raw) != "" {
		if err := applyJSON(raw, &cfg); err != nil {
			return Config{}, err
		}
	}

	overrideString(l.Lookup, "NUPI_ADAPTER_LISTEN_ADDR", &cfg.ListenAddr)
	overrideString(l.Lookup, "NUPI_LOG_LEVEL", &cfg.LogLevel)
	overrideString(l.Lookup, "NUPI_MODEL_VARIANT", &cfg.ModelVariant)
	overrideString(l.Lookup, "NUPI_LANGUAGE_HINT", &cfg.Language)

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func applyJSON(raw string, cfg *Config) error {
	type jsonConfig struct {
		ListenAddr   string `json:"listen_addr"`
		ModelVariant string `json:"model_variant"`
		Language     string `json:"language"`
		LogLevel     string `json:"log_level"`
	}
	var payload jsonConfig
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return fmt.Errorf("config: decode NUPI_MODULE_CONFIG: %w", err)
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

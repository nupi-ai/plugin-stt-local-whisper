package config

import (
	"fmt"
	"strings"
)

const (
	// DefaultListenAddr is used when the adapter runner does not inject an explicit address.
	DefaultListenAddr = "127.0.0.1:50051"
	DefaultModel      = "base"
	DefaultLanguage   = "client"
	DefaultLogLevel   = "info"
	DefaultDataDir    = "data"
)

// Config captures bootstrap configuration extracted from environment variables
// or injected JSON payload (`NUPI_ADAPTER_CONFIG`).
type Config struct {
	ListenAddr     string
	ModelVariant string
	// Language mode: "client" (default), "auto", or ISO 639-1 code
	// (e.g. "pl", "en"). Passed to whisper.cpp for transcription hints.
	Language string
	LogLevel       string
	DataDir        string
	ModelPath      string
	UseStubEngine  bool
	UseGPU         *bool
	FlashAttention *bool
	Threads        *int
	BeamSize       *int
}

// Validate applies defaults, checks required fields, and rejects out-of-range
// values.
func (c *Config) Validate() error {
	if c.ListenAddr == "" {
		return fmt.Errorf("config: listen address is required")
	}
	if c.ModelVariant == "" {
		c.ModelVariant = DefaultModel
	}
	c.Language = strings.ToLower(strings.TrimSpace(c.Language))
	if c.Language == "" {
		c.Language = DefaultLanguage
	}
	if c.Language != "client" && c.Language != "auto" && len(c.Language) > 8 {
		return fmt.Errorf("config: language must be 'client', 'auto', or a short ISO 639-1 code, got %q", c.Language)
	}
	if c.LogLevel == "" {
		c.LogLevel = DefaultLogLevel
	}
	if c.DataDir == "" {
		c.DataDir = DefaultDataDir
	}
	if c.Threads != nil && *c.Threads < 0 {
		return fmt.Errorf("config: threads must be >= 0, got %d", *c.Threads)
	}
	if c.BeamSize != nil && *c.BeamSize < 1 {
		return fmt.Errorf("config: beam_size must be >= 1, got %d", *c.BeamSize)
	}
	return nil
}

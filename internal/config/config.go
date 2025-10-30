package config

import "fmt"

const (
	// DefaultListenAddr is used when the adapter runner does not inject an explicit address.
	DefaultListenAddr = "127.0.0.1:50051"
	DefaultModel      = "base"
	DefaultLanguage   = "auto"
	DefaultLogLevel   = "info"
	DefaultDataDir    = "data"
)

// Config captures bootstrap configuration extracted from environment variables
// or injected JSON payload (`NUPI_ADAPTER_CONFIG`).
type Config struct {
	ListenAddr     string
	ModelVariant   string
	Language       string
	LogLevel       string
	DataDir        string
	ModelPath      string
	UseStubEngine  bool
	UseGPU         *bool
	FlashAttention *bool
	Threads        *int
}

// Validate applies defaults and raises an error when required fields are
// missing. Iteration 0 keeps validation minimal; future work will extend it.
func (c *Config) Validate() error {
	if c.ListenAddr == "" {
		return fmt.Errorf("config: listen address is required")
	}
	if c.ModelVariant == "" {
		c.ModelVariant = DefaultModel
	}
	if c.Language == "" {
		c.Language = DefaultLanguage
	}
	if c.LogLevel == "" {
		c.LogLevel = DefaultLogLevel
	}
	if c.DataDir == "" {
		c.DataDir = DefaultDataDir
	}
	return nil
}

package adapterinfo

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"gopkg.in/yaml.v3"
)

// Metadata captures static identifiers for the adapter. Centralising the values
// makes it easy to clone this repository for new adapters.
type Metadata struct {
	Name        string
	BinaryName  string
	Slug        string
	Description string
	GeneratorID string
	Version     string
}

// Info describes the current adapter.
var Info = mustLoadMetadata()

// TranscriptMetadata produces the standard metadata payload attached
// to emitted transcripts.
func TranscriptMetadata(modelVariant, language string) map[string]string {
	return map[string]string{
		"generator":     Info.GeneratorID,
		"model_variant": modelVariant,
		"language":      language,
	}
}

// Version returns the adapter semantic version.
func Version() string {
	return Info.Version
}

func mustLoadMetadata() Metadata {
	data, err := loadManifest()
	if err != nil {
		panic(err)
	}
	meta, err := parseManifest(data)
	if err != nil {
		panic(err)
	}
	return meta
}

func loadManifest() ([]byte, error) {
	var candidates []string
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, wd)
	}
	if _, file, _, ok := runtime.Caller(0); ok {
		srcRoot := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
		candidates = append(candidates, srcRoot)
	}

	seen := make(map[string]struct{})
	for _, base := range candidates {
		base = filepath.Clean(base)
		if _, ok := seen[base]; ok {
			continue
		}
		seen[base] = struct{}{}

		candidate := filepath.Join(base, "plugin.yaml")
		if data, err := os.ReadFile(candidate); err == nil {
			return data, nil
		}
	}
	return nil, errors.New("adapterinfo: plugin.yaml not found next to binary or source tree")
}

type manifestDocument struct {
	Metadata struct {
		Name        string `yaml:"name"`
		Slug        string `yaml:"slug"`
		Description string `yaml:"description"`
		Version     string `yaml:"version"`
		Generator   string `yaml:"generator"`
	} `yaml:"metadata"`
	Spec struct {
		Entrypoint struct {
			Command string `yaml:"command"`
		} `yaml:"entrypoint"`
	} `yaml:"spec"`
}

func parseManifest(data []byte) (Metadata, error) {
	var doc manifestDocument
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return Metadata{}, fmt.Errorf("adapterinfo: decode manifest: %w", err)
	}

	meta := Metadata{
		Name:        strings.TrimSpace(doc.Metadata.Name),
		Slug:        strings.TrimSpace(doc.Metadata.Slug),
		Description: strings.TrimSpace(doc.Metadata.Description),
		Version:     strings.TrimSpace(doc.Metadata.Version),
		GeneratorID: strings.TrimSpace(doc.Metadata.Generator),
	}

	if meta.Version == "" {
		return Metadata{}, fmt.Errorf("adapterinfo: metadata.version missing in manifest")
	}
	if meta.Slug == "" {
		return Metadata{}, fmt.Errorf("adapterinfo: metadata.slug missing in manifest")
	}
	if meta.Name == "" {
		meta.Name = meta.Slug
	}
	if meta.Description == "" {
		meta.Description = meta.Name
	}

	entrypoint := strings.TrimSpace(doc.Spec.Entrypoint.Command)
	if entrypoint != "" {
		cmd := entrypoint
		if strings.HasPrefix(cmd, "./") {
			cmd = cmd[2:]
		}
		meta.BinaryName = cmd
	}
	if meta.BinaryName == "" {
		meta.BinaryName = meta.Slug
	}
	if meta.GeneratorID == "" {
		meta.GeneratorID = meta.Slug
	}

	return meta, nil
}

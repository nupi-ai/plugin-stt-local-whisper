package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/models"
)

func main() {
	var (
		variant = flag.String("variant", "base", "model variant defined in internal/models/embedded_manifest.json")
		output  = flag.String("dir", "testdata", "base directory where models/<file> will be stored")
	)
	flag.Parse()

	if strings.TrimSpace(*output) == "" {
		fmt.Fprintln(os.Stderr, "download_model: --dir must not be empty")
		os.Exit(2)
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	baseDir := filepath.Clean(*output)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	manager, err := models.NewManager(baseDir, logger)
	if err != nil {
		fmt.Fprintf(os.Stderr, "download_model: init manager: %v\n", err)
		os.Exit(1)
	}

	manifest, err := models.DefaultManifest()
	if err != nil {
		fmt.Fprintf(os.Stderr, "download_model: load manifest: %v\n", err)
		os.Exit(1)
	}

	path, err := manager.EnsureVariant(ctx, *variant, models.EnsureOptions{
		Manifest: manifest,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "download_model: ensure variant %q: %v\n", *variant, err)
		os.Exit(1)
	}

	fmt.Printf("Model %q ready at %s\n", *variant, path)
}

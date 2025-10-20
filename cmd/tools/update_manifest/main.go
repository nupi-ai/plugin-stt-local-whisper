package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/nupi-ai/module-nupi-whisper-local-stt/internal/models"
)

func main() {
	manifestPath := flag.String("manifest", "internal/models/embedded_manifest.json", "Path to manifest JSON to update")
	flag.Parse()

	file, err := os.Open(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open manifest: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	manifest, err := models.LoadManifest(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse manifest: %v\n", err)
		os.Exit(1)
	}

	client := &http.Client{Timeout: 10 * time.Minute}

	for name, variant := range manifest.Variants {
		if variant.URL == "" {
			fmt.Printf("%s: skipping (no URL)\n", name)
			continue
		}

		fmt.Printf("%s: downloading %s...\n", name, variant.URL)
		resp, err := client.Get(variant.URL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: download error: %v\n", name, err)
			continue
		}
		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "%s: unexpected status %s\n", name, resp.Status)
			resp.Body.Close()
			continue
		}

		hasher := sha256.New()
		written, err := io.Copy(hasher, resp.Body)
		resp.Body.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: read error: %v\n", name, err)
			continue
		}

		sum := hex.EncodeToString(hasher.Sum(nil))
		variant.SHA256 = sum
		variant.SizeBytes = written
		manifest.Variants[name] = variant

		fmt.Printf("%s: size=%d sha256=%s\n", name, written, sum)
	}

	out, err := os.Create(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "write manifest: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	encoder := json.NewEncoder(out)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(manifest); err != nil {
		fmt.Fprintf(os.Stderr, "encode manifest: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Updated manifest written to %s\n", *manifestPath)
}

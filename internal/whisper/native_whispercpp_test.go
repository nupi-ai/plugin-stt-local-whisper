//go:build whispercpp

package whisper

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeEngineTranscribesFixture(t *testing.T) {
	if !NativeAvailable() {
		t.Skip("native backend not available")
	}
	modelRel := filepath.Join("testdata", "models", "ggml-base.en.bin")
	modelPath := locateFixture(t, modelRel, "run `go run ./cmd/tools/download_model --variant base --dir testdata`")

	engine, err := NewNativeEngine(modelPath)
	if err != nil {
		t.Fatalf("NewNativeEngine: %v", err)
	}
	t.Cleanup(func() {
		if cerr := engine.Close(); cerr != nil {
			t.Errorf("engine.Close: %v", cerr)
		}
	})

	audioPath := locateFixture(t, filepath.Join("testdata", "test.wav"), "")
	audio, sampleRate, err := loadPCM16LE(audioPath)
	if err != nil {
		t.Fatalf("loadPCM16LE: %v", err)
	}
	if sampleRate != 16000 {
		t.Fatalf("unexpected sample rate: got %d, want 16000", sampleRate)
	}
	if len(audio) == 0 {
		t.Fatal("empty PCM payload")
	}

	ctx := context.Background()

	const chunkSize = 3200 // ~100ms of audio at 16kHz mono PCM16
	var (
		offset       int
		seq          uint64
		partialCount int
		finalText    string
	)

	for offset < len(audio) {
		end := offset + chunkSize
		if end > len(audio) {
			end = len(audio)
		}

		results, err := engine.TranscribeSegment(ctx, audio[offset:end], Options{
			Language: "en",
			Sequence: seq,
		})
		if err != nil {
			t.Fatalf("TranscribeSegment(seq=%d): %v", seq, err)
		}
		for _, res := range results {
			if res.Final {
				finalText = res.Text
			} else if strings.TrimSpace(res.Text) != "" {
				partialCount++
			}
		}

		offset = end
		seq++
	}

	results, err := engine.Flush(ctx, Options{Language: "en"})
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	for _, res := range results {
		if res.Final {
			finalText = res.Text
		}
	}

	if finalText == "" {
		t.Fatalf("final transcript empty (partials=%d)", partialCount)
	}

	lower := strings.ToLower(finalText)
	expectedPhrases := []string{
		"hi nupi",
		"show me what you can do",
	}
	for _, phrase := range expectedPhrases {
		if !strings.Contains(lower, phrase) {
			t.Fatalf("final transcript %q missing phrase %q", finalText, phrase)
		}
	}
	if partialCount == 0 {
		t.Fatalf("expected at least one partial transcript, got 0")
	}
}

func locateFixture(t *testing.T, relativePath string, suggestion string) string {
	t.Helper()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}

	visited := make([]string, 0, 4)
	for {
		candidate := filepath.Join(wd, relativePath)
		visited = append(visited, candidate)

		info, err := os.Stat(candidate)
		if err == nil && !info.IsDir() {
			return candidate
		}
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("stat %s: %v", candidate, err)
		}

		parent := filepath.Dir(wd)
		if parent == wd {
			msg := fmt.Sprintf("fixture %s not found (checked: %s)", relativePath, strings.Join(visited, ", "))
			if suggestion != "" {
				msg = fmt.Sprintf("%s; %s", msg, suggestion)
			}
			t.Skip(msg)
		}
		wd = parent
	}
}

func loadPCM16LE(path string) ([]byte, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("read wav: %w", err)
	}
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("invalid wav header")
	}

	offset := 12
	var (
		sampleRate    int
		audioFormat   uint16
		channels      uint16
		bitsPerSample uint16
		audioData     []byte
	)

	for offset+8 <= len(data) {
		chunkID := string(data[offset : offset+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		chunkStart := offset + 8
		chunkEnd := chunkStart + chunkSize
		if chunkEnd > len(data) {
			return nil, 0, fmt.Errorf("chunk %s out of range", chunkID)
		}
		switch chunkID {
		case "fmt ":
			if chunkSize < 16 {
				return nil, 0, fmt.Errorf("fmt chunk too small")
			}
			audioFormat = binary.LittleEndian.Uint16(data[chunkStart : chunkStart+2])
			channels = binary.LittleEndian.Uint16(data[chunkStart+2 : chunkStart+4])
			sampleRate = int(binary.LittleEndian.Uint32(data[chunkStart+4 : chunkStart+8]))
			bitsPerSample = binary.LittleEndian.Uint16(data[chunkStart+14 : chunkStart+16])
		case "data":
			audioData = data[chunkStart:chunkEnd]
		}
		// Chunks are word aligned.
		offset = chunkEnd
		if chunkSize%2 == 1 {
			offset++
		}
	}

	if audioFormat != 1 {
		return nil, 0, fmt.Errorf("unsupported audio format %d", audioFormat)
	}
	if channels != 1 {
		return nil, 0, fmt.Errorf("expected mono audio, got %d channels", channels)
	}
	if bitsPerSample != 16 {
		return nil, 0, fmt.Errorf("expected 16-bit PCM, got %d", bitsPerSample)
	}
	if len(audioData) == 0 {
		return nil, 0, fmt.Errorf("no data chunk found")
	}
	return audioData, sampleRate, nil
}

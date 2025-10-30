//go:build whispercpp

package engine

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

	engine := openTestNativeEngine(t)
	audio, sampleRate := loadTestAudio(t)
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
	if len(results) == 0 {
		t.Fatal("flush returned no transcripts")
	}
	finalResult := results[len(results)-1]
	if !finalResult.Final {
		t.Fatalf("expected final transcript flag, got Final=%v", finalResult.Final)
	}
	if finalResult.Text != "" {
		finalText = finalResult.Text
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

func TestNativeEngineAutoLanguageFallsBack(t *testing.T) {
	if !NativeAvailable() {
		t.Skip("native backend not available")
	}

	engine := openTestNativeEngine(t)
	engine.SetDefaultLanguage("en")
	audio, _ := loadTestAudio(t)
	ctx := context.Background()

	const chunkSize = 9600
	var final string

	for seq, offset := uint64(0), 0; offset < len(audio); seq++ {
		end := offset + chunkSize
		if end > len(audio) {
			end = len(audio)
		}
		results, err := engine.TranscribeSegment(ctx, audio[offset:end], Options{Language: "auto", Sequence: seq})
		if err != nil {
			t.Fatalf("TranscribeSegment(auto) returned error: %v", err)
		}
		for _, res := range results {
			if strings.TrimSpace(res.Text) != "" {
				final = res.Text
			}
		}
		offset = end
	}

	results, err := engine.Flush(ctx, Options{Language: "auto"})
	if err != nil {
		t.Fatalf("Flush(auto) returned error: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected final transcript")
	}
	final = strings.TrimSpace(results[len(results)-1].Text)
	if final == "" {
		t.Fatal("final transcript empty with auto language")
	}
}

func TestNativeEngineTranscribeSegmentRespectsContextCancellation(t *testing.T) {
	if !NativeAvailable() {
		t.Skip("native backend not available")
	}

	engine := openTestNativeEngine(t)
	audio, _ := loadTestAudio(t)
	if len(audio) > 3200 {
		audio = audio[:3200]
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := engine.TranscribeSegment(ctx, audio, Options{Language: "en"}); err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation, got %v", err)
	}
}

func TestNativeEngineTrimsOversizedAudio(t *testing.T) {
	if !NativeAvailable() {
		t.Skip("native backend not available")
	}

	const (
		bytesPerSample = 2
		sampleRate     = 16000
	)

	engine := openTestNativeEngine(t)
	windowSamples := (targetWindowMillis * sampleRate) / 1000
	stepSamples := (minFrameMillis * sampleRate) / 1000
	big := make([]byte, (windowSamples+stepSamples)*bytesPerSample)

	if _, err := engine.TranscribeSegment(context.Background(), big, Options{Language: "en"}); err != nil {
		t.Fatalf("unexpected error for oversized audio: %v", err)
	}
	if _, err := engine.Flush(context.Background(), Options{Language: "en"}); err != nil {
		t.Fatalf("flush after oversized segment failed: %v", err)
	}
}

func TestNewNativeEngineRejectsEmptyPath(t *testing.T) {
	if _, err := NewNativeEngine("", NativeOptions{}); err == nil {
		t.Fatal("expected error for empty model path")
	}
}

func openTestNativeEngine(tb testing.TB) *NativeEngine {
	tb.Helper()

	modelRel := filepath.Join("testdata", "models", "ggml-base.en.bin")
	modelPath := locateFixture(tb, modelRel, "run `go run ./cmd/tools/models/download --variant base --dir testdata`")
	eng, err := NewNativeEngine(modelPath, NativeOptions{})
	if err != nil {
		tb.Fatalf("NewNativeEngine: %v", err)
	}
	native, ok := eng.(*NativeEngine)
	if !ok {
		tb.Fatalf("unexpected engine type %T", eng)
	}
	tb.Cleanup(func() {
		if cerr := native.Close(); cerr != nil {
			tb.Errorf("engine.Close: %v", cerr)
		}
	})
	return native
}

func loadTestAudio(tb testing.TB) ([]byte, int) {
	tb.Helper()
	audioPath := locateFixture(tb, filepath.Join("testdata", "test.wav"), "")
	audio, sampleRate, err := loadPCM16LE(audioPath)
	if err != nil {
		tb.Fatalf("loadPCM16LE: %v", err)
	}
	return audio, sampleRate
}

func locateFixture(tb testing.TB, relativePath string, suggestion string) string {
	tb.Helper()

	wd, err := os.Getwd()
	if err != nil {
		tb.Fatalf("getwd: %v", err)
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
			tb.Fatalf("stat %s: %v", candidate, err)
		}

		parent := filepath.Dir(wd)
		if parent == wd {
			msg := fmt.Sprintf("fixture %s not found (checked: %s)", relativePath, strings.Join(visited, ", "))
			if suggestion != "" {
				msg = fmt.Sprintf("%s; %s", msg, suggestion)
			}
			tb.Skip(msg)
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

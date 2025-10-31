package adapterinfo

import "testing"

func TestMetadataLoadedFromManifest(t *testing.T) {
	t.Run("version", func(t *testing.T) {
		if Version() == "" {
			t.Fatal("Version() returned empty string")
		}
		if Version() != Info.Version {
			t.Fatalf("Version() mismatch: got %q want %q", Version(), Info.Version)
		}
	})

	expect := Metadata{
		Name:        "Nupi Whisper Local STT",
		BinaryName:  "stt-local-whisper",
		Slug:        "stt-local-whisper",
		Description: "Local speech-to-text adapter backed by Whisper.",
		GeneratorID: "stt-local-whisper",
		Version:     "1.0.0",
	}

	if Info != expect {
		t.Fatalf("unexpected Info metadata: %+v", Info)
	}
}

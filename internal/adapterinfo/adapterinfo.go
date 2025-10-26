package adapterinfo

// Metadata captures static identifiers for the adapter. Centralising the values
// makes it easy to clone this repository for new adapters.
type Metadata struct {
	Name        string
	BinaryName  string
	Slug        string
	Description string
	GeneratorID string
}

// Info describes the current adapter.
var Info = Metadata{
	Name:        "Nupi Whisper Local STT",
	BinaryName:  "plugin-stt-local-whisper",
	Slug:        "stt-local-whisper",
	Description: "Local speech-to-text adapter backed by Whisper.",
	GeneratorID: "stt-local-whisper",
}

// TranscriptMetadata produces the standard metadata payload attached
// to emitted transcripts.
func TranscriptMetadata(modelVariant, language string) map[string]string {
	return map[string]string{
		"generator":     Info.GeneratorID,
		"model_variant": modelVariant,
		"language":      language,
	}
}

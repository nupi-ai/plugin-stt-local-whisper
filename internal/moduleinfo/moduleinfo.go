package moduleinfo

// Metadata captures static identifiers for the module. Centralising the values
// makes it easy to clone this repository for new adapters.
type Metadata struct {
	Name        string
	BinaryName  string
	Slug        string
	Description string
	GeneratorID string
}

// Info describes the current module.
var Info = Metadata{
	Name:        "Nupi Whisper Local STT",
	BinaryName:  "module-nupi-whisper-local-stt",
	Slug:        "nupi-whisper-local-stt",
	Description: "Local speech-to-text adapter backed by Whisper.",
	GeneratorID: "nupi-whisper-local-stt",
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

package whisper

import "testing"

func TestDiffTranscript(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		previous string
		current  string
		want     string
	}{
		{
			name:     "initial text",
			previous: "",
			current:  "hello world",
			want:     "hello world",
		},
		{
			name:     "same text",
			previous: "hello world",
			current:  "hello world",
			want:     "",
		},
		{
			name:     "append text",
			previous: "hello",
			current:  "hello world",
			want:     "world",
		},
		{
			name:     "append whitespace",
			previous: "hello",
			current:  "hello   world",
			want:     "world",
		},
		{
			name:     "prefix mismatch",
			previous: "hello world",
			current:  "hola mundo",
			want:     "hola mundo",
		},
		{
			name:     "multibyte characters",
			previous: "cześć",
			current:  "cześć świecie",
			want:     "świecie",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := diffTranscript(tc.previous, tc.current)
			if got != tc.want {
				t.Fatalf("diffTranscript(%q, %q) = %q, want %q", tc.previous, tc.current, got, tc.want)
			}
		})
	}
}

func TestNormaliseLanguage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		candidate string
		fallback  string
		want      string
	}{
		{"candidate wins", "en", "pl", "en"},
		{"fallback used", "", "pl", "pl"},
		{"defaults to auto", "  ", " ", "auto"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := normaliseLanguage(tc.candidate, tc.fallback); got != tc.want {
				t.Fatalf("normaliseLanguage(%q, %q) = %q, want %q", tc.candidate, tc.fallback, got, tc.want)
			}
		})
	}
}

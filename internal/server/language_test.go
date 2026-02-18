package server

import "testing"

func TestResolveLanguage_ClientWithMetadata(t *testing.T) {
	meta := map[string]string{"nupi.lang.iso1": "pl"}
	got := resolveLanguage("client", meta)
	if got != "pl" {
		t.Errorf("resolveLanguage(client, iso1=pl): got %q, want %q", got, "pl")
	}
}

func TestResolveLanguage_ClientWithoutMetadata(t *testing.T) {
	got := resolveLanguage("client", nil)
	if got != "auto" {
		t.Errorf("resolveLanguage(client, nil): got %q, want %q", got, "auto")
	}
}

func TestResolveLanguage_ClientWithEmptyMetadata(t *testing.T) {
	meta := map[string]string{}
	got := resolveLanguage("client", meta)
	if got != "auto" {
		t.Errorf("resolveLanguage(client, empty): got %q, want %q", got, "auto")
	}
}

func TestResolveLanguage_ClientWithEmptyISO1(t *testing.T) {
	meta := map[string]string{"nupi.lang.iso1": ""}
	got := resolveLanguage("client", meta)
	if got != "auto" {
		t.Errorf("resolveLanguage(client, iso1=empty): got %q, want %q", got, "auto")
	}
}

func TestResolveLanguage_AutoIgnoresMetadata(t *testing.T) {
	meta := map[string]string{"nupi.lang.iso1": "pl"}
	got := resolveLanguage("auto", meta)
	if got != "auto" {
		t.Errorf("resolveLanguage(auto, iso1=pl): got %q, want %q", got, "auto")
	}
}

func TestResolveLanguage_SpecificIgnoresMetadata(t *testing.T) {
	meta := map[string]string{"nupi.lang.iso1": "pl"}
	got := resolveLanguage("de", meta)
	if got != "de" {
		t.Errorf("resolveLanguage(de, iso1=pl): got %q, want %q", got, "de")
	}
}

func TestResolveLanguage_SpecificWithoutMetadata(t *testing.T) {
	got := resolveLanguage("en", nil)
	if got != "en" {
		t.Errorf("resolveLanguage(en, nil): got %q, want %q", got, "en")
	}
}

func TestResolveLanguage_AutoWithoutMetadata(t *testing.T) {
	got := resolveLanguage("auto", nil)
	if got != "auto" {
		t.Errorf("resolveLanguage(auto, nil): got %q, want %q", got, "auto")
	}
}

func TestResolveLanguage_ClientWithWhitespaceISO1(t *testing.T) {
	meta := map[string]string{"nupi.lang.iso1": "  pl  "}
	got := resolveLanguage("client", meta)
	if got != "pl" {
		t.Errorf("resolveLanguage(client, iso1='  pl  '): got %q, want %q", got, "pl")
	}
}

func TestResolveLanguage_ClientWithWhitespaceOnlyISO1(t *testing.T) {
	meta := map[string]string{"nupi.lang.iso1": "   "}
	got := resolveLanguage("client", meta)
	if got != "auto" {
		t.Errorf("resolveLanguage(client, iso1='   '): got %q, want %q", got, "auto")
	}
}

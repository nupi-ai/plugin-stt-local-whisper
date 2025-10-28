package engine

import "strings"

func diffTranscript(previous, current string) string {
	prevTrimmed := strings.TrimSpace(previous)
	currTrimmed := strings.TrimSpace(current)

	if prevTrimmed == "" {
		return currTrimmed
	}
	if prevTrimmed == currTrimmed {
		return ""
	}

	prevRunes := []rune(prevTrimmed)
	currRunes := []rune(currTrimmed)

	if len(prevRunes) > len(currRunes) {
		return currTrimmed
	}
	for i := range prevRunes {
		if currRunes[i] != prevRunes[i] {
			return currTrimmed
		}
	}

	delta := string(currRunes[len(prevRunes):])
	return strings.TrimLeft(delta, " \t\r\n")
}

func normaliseLanguage(candidate, previous, forced string) string {
	if lang := preferLanguage(candidate); lang != "" {
		return lang
	}
	if lang := preferLanguage(previous); lang != "" {
		return lang
	}
	if lang := preferLanguage(forced); lang != "" {
		return lang
	}
	if strings.TrimSpace(candidate) != "" {
		return "auto"
	}
	if strings.TrimSpace(previous) != "" {
		return "auto"
	}
	if trimmed := strings.TrimSpace(forced); trimmed != "" {
		return trimmed
	}
	return "auto"
}

func preferLanguage(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" || strings.EqualFold(trimmed, "auto") {
		return ""
	}
	return strings.ToLower(trimmed)
}

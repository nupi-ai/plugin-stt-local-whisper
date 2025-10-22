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

func normaliseLanguage(candidate, fallback string) string {
	if trimmed := strings.TrimSpace(candidate); trimmed != "" {
		return trimmed
	}
	if trimmed := strings.TrimSpace(fallback); trimmed != "" {
		return trimmed
	}
	return "auto"
}

package main

import (
	"fmt"

	"github.com/nupi-ai/plugin-stt-local-whisper/internal/adapterinfo"
)

func main() {
	fmt.Println(adapterinfo.Version())
}

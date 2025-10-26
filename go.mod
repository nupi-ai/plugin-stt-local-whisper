module github.com/nupi-ai/plugin-stt-local-whisper

go 1.24.0

toolchain go1.24.7

require (
	github.com/nupi-ai/nupi v0.0.0
	google.golang.org/grpc v1.64.0
)

require (
	golang.org/x/net v0.43.0 // indirect
	golang.org/x/sys v0.36.0 // indirect
	golang.org/x/text v0.29.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240318140521-94a12d6c2237 // indirect
	google.golang.org/protobuf v1.33.0 // indirect
)

replace github.com/nupi-ai/nupi => ../nupi

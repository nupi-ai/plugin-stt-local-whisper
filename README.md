# Nupi Whisper Local STT

This repository contains the reference **Nupi Whisper Local STT** adapter. It delivers speech-to-text capabilities to the Nupi platform, implements the `SpeechToTextService` gRPC contract from `github.com/nupi-ai/nupi/api/nap/v1`, and documents the patterns we use across adapters.

## Quickstart

```bash
git clone git@github.com:nupi-ai/plugin-stt-local-whisper.git
cd plugin-stt-local-whisper
go run ./cmd/adapter
```

`cmd/adapter` boots the gRPC server using the stub backend. Configuration is controlled entirely via environment variables (see `docs/architecture/adapter_architecture_plan.md`). The Makefile and CI workflow derive binary and artefact names from `go.mod`, so renaming the module path automatically updates build outputs.

### Switch to the native backend (Whisper)

Unit tests rely on the stub backend, but production transcription needs the native path.
The commands below compile `whisper.cpp` and start the adapter with CGO enabled:

```bash
git submodule update --init --recursive
make native-lib
DYLD_LIBRARY_PATH=$(pwd)/third_party/whisper.cpp/build/src \
    LD_LIBRARY_PATH=$(pwd)/third_party/whisper.cpp/build/src \
    CGO_ENABLED=1 GOFLAGS="-tags=whispercpp" go run ./cmd/adapter
```

If you experiment with another engine, override `NATIVE_CMAKE_FLAGS`,
`NATIVE_BUILD_TARGET`, or `NATIVE_LIB_PREFIX` rather than editing the Makefile.

### Testing and artefacts

```bash
make test          # race-tested suite against the stub backend
make build-native  # CGO build
make test-native   # go test with the native backend
make dist-native   # optional: emit native binary under dist/
make release-native VERSION=1.0.0  # package binary + native libs + manifest into dist/*.tar.gz
```

#### Native integration test

1. Build the native library (`make native-lib`).
2. Cache a reference model:

   ```bash
   go run ./cmd/tools/models/download --variant base --dir testdata
   ```

3. Run the tagged fixture:

   ```bash
   DYLD_LIBRARY_PATH=$(pwd)/third_party/whisper.cpp/build/src \
   LD_LIBRARY_PATH=$(pwd)/third_party/whisper.cpp/build/src \
   CGO_ENABLED=1 GOFLAGS="-tags=whispercpp" \
   go test ./internal/engine -run TestNativeEngineTranscribesFixture -tags whispercpp
   ```

The fixture skips automatically when the GGUF model is unavailable.

### Model manifest helpers

`internal/models/embedded_manifest.json` captures downloadable variants. Refresh the file
after tweaking URLs or upstream artefacts:

```bash
go run ./cmd/tools/models/manifest/update --manifest internal/models/embedded_manifest.json
```

Both helper commands live under `cmd/tools/models/` to encourage reuse across adapters.

> **Note**  
> Protobuf definitions are imported via a local `replace` directive pointing to
> `../nupi`. Keep both repositories side-by-side (or adjust `go.work`) before running
> builds elsewhere.

> **CI note**  
> Cross-platform artefacts are produced only for tagged releases or manual dispatch
> (`workflow_dispatch`). Routine pushes/pull requests run the lint/test job only.

## Repository structure

- `cmd/adapter`: release entrypoint.
- `internal/server`: gRPC implementation of `SpeechToTextService`.
- `internal/config`: configuration loader (env + JSON payloads).
- `internal/models`: model cache manager and manifest utilities.
- `internal/engine`: backend abstraction with stub + Whisper implementations.
- `internal/adapterinfo`: centralised adapter metadata (name, slug, generator id).
- `docs/`: architecture plan, operations guide, research notes.
- `plugin.yaml`: manifest consumed by the adapter registry/runner.
- `.github/workflows`: CI definitions (lint/test + optional release matrix).
- `third_party/whisper.cpp`: git submodule containing the Whisper sources.
- `Makefile`: convenience targets (`build`, `build-native`, `test`, `test-native`, `dist`, `dist-native`).

### Naming conventions

- Repository path and binary follow the `plugin-stt-…` prefix to match the new
  Nupi plugin catalog.
- The adapter slug exposed to the registry/daemon is intentionally short:
  `stt-local-whisper`. This mirrors the convention described in the platform
  TODO (`Podetap K`) and is centralised in `internal/adapterinfo` and
  `plugin.yaml`.
- Module path updates in `go.mod` automatically propagate to the Makefile and CI so the
  binary/artifact names stay in sync.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `NUPI_ADAPTER_CONFIG` | – | JSON payload injected by the adapter runner. |
| `NUPI_ADAPTER_LISTEN_ADDR` | `127.0.0.1:50051` | gRPC bind address. |
| `NUPI_MODEL_VARIANT` | `base` | Manifest-based model lookup. |
| `NUPI_LANGUAGE_HINT` | `auto` | ISO-639 language hint forwarded to the engine. |
| `NUPI_ADAPTER_DATA_DIR` | `./data` | Working directory for cached models. |
| `NUPI_MODEL_PATH` | – | Absolute GGUF path (skips manifest resolution). |
| `NUPI_ADAPTER_USE_STUB_ENGINE` | `false` | Forces the stub backend (e.g. CI). |
| `NUPI_LOG_LEVEL` | `info` | Logger verbosity (`debug`, `info`, `warn`, `error`). |

Drop GGUF artefacts under `${NUPI_ADAPTER_DATA_DIR}/models/<variant>.gguf` or point
`NUPI_MODEL_PATH` at a specific file.

When the native backend is enabled, the following optional environment toggles are also recognised:

| Env var | Default | Purpose |
| --- | --- | --- |
| `WHISPERCPP_USE_GPU` | `true` | Enable Whisper's GPU kernels when available. |
| `WHISPERCPP_FLASH_ATTENTION` | `true` | Toggle FlashAttention kernels within whisper.cpp. |
| `WHISPERCPP_THREADS` | host CPU cores | Override the inference thread count |

The manifest exposes matching adapter options (`use_gpu`, `flash_attention`, `threads`). Defaults mirror the table above; `threads: 0` means auto-detect.

Operational expectations—health, telemetry, timeouts—are described in
[`docs/operations/runtime.md`](docs/operations/runtime.md).

## Implementation notes (Whisper-specific)

- The native backend uses `whisper.cpp` behind the `whispercpp` build tag and streams
  partial transcripts by diffing the aggregate transcript.
- Model downloads default to the official `ggml` artefacts; checksums are verified before
  caching.
- Telemetry captures per-stream metrics and shutdown totals so the adapter runner can
  emit structured events.

See `todo.md` for the iteration log and upcoming work (configuration hardening, release
packaging, incremental inference improvements).

## Releasing

1. Bump the adapter metadata in `plugin.yaml` (version, description, slug). Go code reads them directly from the manifest.
2. Run tests (`make test` and, when available, `make test-native` with the native toolchain).
3. Build the native artefact:

   ```bash
   make release-native VERSION=1.0.0
   ```

   The Makefile verifies that the Go code and manifest agree on the version and emits
   `dist/stt-local-whisper_<version>_<os>_<arch>.tar.gz` alongside a `.sha256` checksum.
4. Upload the tarball via your preferred release channel (GitHub Release, plugin registry,
   etc.) and update the changelog/upgrade notes as needed.

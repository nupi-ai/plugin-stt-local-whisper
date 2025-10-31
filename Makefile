.PHONY: all build build-native dist dist-native test test-native native-lib clean release-native

# === Adapter identity ===
ADAPTER_NAME ?= $(shell go list -m)
ADAPTER_BINARY ?= stt-local-whisper
NATIVE_TAG := whispercpp
NATIVE_DIR := third_party/whisper.cpp

GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
VERSION ?= $(shell go run ./cmd/tools/version)
MANIFEST_VERSION := $(shell awk '/^[[:space:]]*version:/{print $$2; exit}' plugin.yaml)
ARTIFACT_BASENAME := $(ADAPTER_BINARY)_$(VERSION)_$(GOOS)_$(GOARCH)
ARTIFACT := dist/$(ARTIFACT_BASENAME).tar.gz
PACKAGE_DIR := dist/.package

# === Native library build (Whisper-specific defaults) ===
NATIVE_BUILD := $(NATIVE_DIR)/build
NATIVE_LIB_DIR := $(NATIVE_BUILD)/src
NATIVE_CMAKE_FLAGS ?= -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF
NATIVE_BUILD_TARGET ?= whisper
NATIVE_LIB_PREFIX ?= libwhisper

all: build

build:
	go build ./...

build-native: native-lib
	CGO_ENABLED=1 GOFLAGS="-tags=$(NATIVE_TAG)" go build ./...


dist: build
	cmake -E make_directory dist
	go build -o dist/$(ADAPTER_BINARY) ./cmd/adapter

dist-native: native-lib
	cmake -E make_directory dist
	DYLD_LIBRARY_PATH=$(NATIVE_LIB_DIR) LD_LIBRARY_PATH=$(NATIVE_LIB_DIR) \
		CGO_ENABLED=1 GOFLAGS="-tags=$(NATIVE_TAG)" go build -o dist/$(ADAPTER_BINARY) ./cmd/adapter

release-native: dist-native
	@if [ "$(VERSION)" != "$(MANIFEST_VERSION)" ]; then \
		echo "ERROR: version mismatch – adapterinfo reports $(VERSION) but plugin.yaml declares $(MANIFEST_VERSION)"; \
		exit 1; \
	fi
	@echo "Preparing release $(VERSION) for $(GOOS)/$(GOARCH)…"
	rm -rf $(PACKAGE_DIR)
	cmake -E make_directory $(PACKAGE_DIR)
	cp dist/$(ADAPTER_BINARY) $(PACKAGE_DIR)/
	cp plugin.yaml $(PACKAGE_DIR)/
	cp LICENSE $(PACKAGE_DIR)/
	@if ls $(NATIVE_LIB_DIR)/$(NATIVE_LIB_PREFIX)* >/dev/null 2>&1; then \
		cp $(NATIVE_LIB_DIR)/$(NATIVE_LIB_PREFIX)* $(PACKAGE_DIR)/; \
	else \
		echo "warning: no native libraries found in $(NATIVE_LIB_DIR)"; \
	fi
	tar -C $(PACKAGE_DIR) -czf $(ARTIFACT) .
	python3 -c 'import hashlib, pathlib, sys; p = pathlib.Path(sys.argv[1]); print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")' $(ARTIFACT) > $(ARTIFACT).sha256
	rm -rf $(PACKAGE_DIR)
	@echo ""
	@echo "Release artefact created:"
	@echo "  $(ARTIFACT)"
	@echo "Checksum:"
	@cat $(ARTIFACT).sha256

test:
	GOCACHE=$(PWD)/.gocache go test -race ./...

native-lib:
	cmake -E make_directory $(NATIVE_BUILD)
	cmake -S $(NATIVE_DIR) -B $(NATIVE_BUILD) -Wno-dev \
		-DCMAKE_BUILD_TYPE=Release \
		$(NATIVE_CMAKE_FLAGS)
	cmake --build $(NATIVE_BUILD) --target $(NATIVE_BUILD_TARGET)

test-native: native-lib
	DYLD_LIBRARY_PATH=$(NATIVE_LIB_DIR) LD_LIBRARY_PATH=$(NATIVE_LIB_DIR) \
		CGO_ENABLED=1 GOFLAGS="-tags=$(NATIVE_TAG)" go test ./...

clean:
	cmake -E rm -f $(NATIVE_BUILD)/$(NATIVE_LIB_PREFIX).*
	cmake -E rm -rf $(NATIVE_BUILD)
	go clean ./...

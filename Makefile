.PHONY: all build build-native dist dist-native test test-native native-lib clean

# === Adapter identity ===
ADAPTER_NAME ?= $(shell go list -m)
ADAPTER_BINARY ?= $(notdir $(ADAPTER_NAME))
NATIVE_TAG := whispercpp
NATIVE_DIR := third_party/whisper.cpp

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

test:
	GOCACHE=$(PWD)/.gocache go test -race ./...

native-lib:
	cmake -E make_directory $(NATIVE_BUILD)
	cmake -S $(NATIVE_DIR) -B $(NATIVE_BUILD) \
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

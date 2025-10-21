.PHONY: all build build-whisper dist-whispercpp test whispercpp clean

WHISPER_DIR := third_party/whisper.cpp
WHISPER_BUILD := $(WHISPER_DIR)/build
WHISPER_LIB_DIR := $(WHISPER_BUILD)/src

all: build

build:
	go build ./...

build-whisper:
	CGO_ENABLED=1 GOFLAGS="-tags=whispercpp" go build ./...

dist-whispercpp: whispercpp
	cmake -E make_directory dist
	DYLD_LIBRARY_PATH=$(WHISPER_LIB_DIR) LD_LIBRARY_PATH=$(WHISPER_LIB_DIR) \
		CGO_ENABLED=1 GOFLAGS="-tags=whispercpp" go build -o dist/nupi-whisper-local-stt ./cmd/adapter

test:
	GOCACHE=$(PWD)/.gocache go test -race ./...

whispercpp:
	cmake -E make_directory $(WHISPER_BUILD)
	cmake -S $(WHISPER_DIR) -B $(WHISPER_BUILD) \
		-DCMAKE_BUILD_TYPE=Release \
		-DWHISPER_BUILD_TESTS=OFF \
		-DWHISPER_BUILD_EXAMPLES=OFF
	cmake --build $(WHISPER_BUILD) --target whisper

test-whisper: whispercpp
	DYLD_LIBRARY_PATH=$(WHISPER_LIB_DIR) LD_LIBRARY_PATH=$(WHISPER_LIB_DIR) \
		CGO_ENABLED=1 GOFLAGS="-tags=whispercpp" go test ./...

clean:
	cmake -E rm -f $(WHISPER_BUILD)/libwhisper.*
	cmake -E rm -rf $(WHISPER_BUILD)
	go clean ./...

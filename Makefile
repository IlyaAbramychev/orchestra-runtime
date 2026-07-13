BINARY = orchestra-runtime
WORKER_BINARY = orchestra-worker
# Version is injected into the binary via -ldflags (see GO_BUILD below) and
# surfaced to clients by GET /api/system. Leave unset for dev builds — the
# default picks a short git SHA so mismatches between builds are obvious.
VERSION ?= dev-$(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
LLAMA_CPP_COMMIT ?= $(shell git -C $(LLAMA_DIR) rev-parse HEAD 2>/dev/null || echo unknown)
BUILD_DIR = bin
INSTALL_DIR ?= $(HOME)/.orchestra/bin
LLAMA_DIR = llama.cpp
LLAMA_BUILD_DIR = build/llama

# Default target
.PHONY: all
all: build-metal

# --- llama.cpp build targets ---

# -DLLAMA_CURL=OFF: recent llama.cpp requires libcurl for its own model-download
# helpers (llama_model_load_from_url) which we don't use — we do all HTTP from
# Go. Disabling avoids the libcurl4-openssl-dev dep on Linux.

.PHONY: llama-prepare
llama-prepare:
	@if [ -f "$(LLAMA_DIR)/build/CMakeCache.txt" ]; then \
		cache_home=$$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$(LLAMA_DIR)/build/CMakeCache.txt"); \
		current_home=$$(cd "$(LLAMA_DIR)" && pwd -P); \
		if [ -n "$$cache_home" ] && [ "$$cache_home" != "$$current_home" ]; then \
			echo "Resetting relocated llama.cpp CMake cache ($$cache_home -> $$current_home)..."; \
			rm -f "$(LLAMA_DIR)/build/CMakeCache.txt"; \
			rm -rf "$(LLAMA_DIR)/build/CMakeFiles"; \
		fi; \
	fi

.PHONY: llama-metal
llama-metal: llama-prepare
	@echo "Building llama.cpp with Metal backend..."
	cd $(LLAMA_DIR) && cmake -B build \
		-DGGML_METAL=ON \
		-DGGML_ACCELERATE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_BUILD_APP=OFF \
		-DLLAMA_BUILD_MTMD=ON \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_CURL=OFF \
		-DLLAMA_OPENSSL=OFF \
		-DCMAKE_BUILD_TYPE=Release
	cd $(LLAMA_DIR) && cmake --build build --config Release -j$(shell sysctl -n hw.ncpu)
	@echo "llama.cpp built successfully (Metal)"

.PHONY: llama-cuda
llama-cuda: llama-prepare
	@echo "Building llama.cpp with CUDA backend..."
	cd $(LLAMA_DIR) && cmake -B build \
		-DGGML_CUDA=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_BUILD_APP=OFF \
		-DLLAMA_BUILD_MTMD=ON \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_CURL=OFF \
		-DLLAMA_OPENSSL=OFF \
		-DCMAKE_BUILD_TYPE=Release
	cd $(LLAMA_DIR) && cmake --build build --config Release -j$$(nproc)
	@echo "llama.cpp built successfully (CUDA)"

.PHONY: llama-cpu
llama-cpu: llama-prepare
	@echo "Building llama.cpp with CPU backend..."
	cd $(LLAMA_DIR) && cmake -B build \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_BUILD_APP=OFF \
		-DLLAMA_BUILD_MTMD=ON \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_CURL=OFF \
		-DLLAMA_OPENSSL=OFF \
		-DCMAKE_BUILD_TYPE=Release
	cd $(LLAMA_DIR) && cmake --build build --config Release -j4
	@echo "llama.cpp built successfully (CPU)"

# --- Go build targets ---

# Detect llama.cpp library paths (cmake output structure varies)
LLAMA_LIB_DIR = $(LLAMA_DIR)/build/src
GGML_LIB_DIR = $(LLAMA_DIR)/build/ggml/src
COMMON_LIB_DIR = $(LLAMA_DIR)/build/common
GGML_METAL_DIR = $(LLAMA_DIR)/build/ggml/src/ggml-metal
GGML_BLAS_DIR = $(LLAMA_DIR)/build/ggml/src/ggml-blas

LLAMA_INCLUDE = $(shell pwd)/$(LLAMA_DIR)/include
LLAMA_LIB = $(shell pwd)/$(LLAMA_LIB_DIR)
COMMON_LIB = $(shell pwd)/$(COMMON_LIB_DIR)
GGML_LIB = $(shell pwd)/$(GGML_LIB_DIR)
GGML_METAL = $(shell pwd)/$(GGML_METAL_DIR)
GGML_BLAS = $(shell pwd)/$(GGML_BLAS_DIR)

GGML_INCLUDE = $(shell pwd)/$(LLAMA_DIR)/ggml/include
COMMON_INCLUDE = $(shell pwd)/$(LLAMA_DIR)/common
VENDOR_INCLUDE = $(shell pwd)/$(LLAMA_DIR)/vendor
BASE_CGO = CGO_ENABLED=1 CGO_CFLAGS="-I$(LLAMA_INCLUDE) -I$(GGML_INCLUDE)" CGO_CXXFLAGS="-I$(LLAMA_INCLUDE) -I$(GGML_INCLUDE) -I$(COMMON_INCLUDE) -I$(VENDOR_INCLUDE)"
BASE_LDFLAGS = -L$(COMMON_LIB) -L$(LLAMA_LIB) -L$(GGML_LIB) -lllama-common -lllama-common-base -lllama -lggml -lstdc++ -lm
METAL_LDFLAGS = -L$(COMMON_LIB) -L$(LLAMA_LIB) -L$(GGML_LIB) -L$(GGML_METAL) -L$(GGML_BLAS) -lllama-common -lllama-common-base -lllama -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas -lstdc++ -lm -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
TEST_LDFLAGS = $(BASE_LDFLAGS)
ifeq ($(shell uname -s),Darwin)
TEST_LDFLAGS = $(METAL_LDFLAGS)
endif
GO_LDFLAGS = -s -w -X main.version=$(VERSION) -X main.buildCommit=$(BUILD_COMMIT) -X main.llamaCppCommit=$(LLAMA_CPP_COMMIT)
GO_BUILD = go build -ldflags="$(GO_LDFLAGS)" -o $(BUILD_DIR)/$(BINARY) ./cmd/server
GO_BUILD_WORKER = go build -ldflags="$(GO_LDFLAGS)" -o $(BUILD_DIR)/$(WORKER_BINARY) ./cmd/worker

# HOST binary (./cmd/server) does NOT need CGo in subprocess mode. We still
# link against llama.cpp for in-process mode (the default) so users keep the
# existing single-binary workflow. Worker always needs CGo.
#
# On a subprocess-only future we could split: `make build-host-nocgo` that
# compiles the server without CGo and only ships with the worker — ~10 MB
# smaller and the host survives corrupt CGo libs.

.PHONY: build-metal
build-metal: llama-metal
	@echo "Building $(BINARY) + $(WORKER_BINARY) (Metal)..."
	mkdir -p $(BUILD_DIR)
	$(BASE_CGO) CGO_LDFLAGS="$(METAL_LDFLAGS)" $(GO_BUILD)
	$(BASE_CGO) CGO_LDFLAGS="$(METAL_LDFLAGS)" $(GO_BUILD_WORKER)
	@echo "Built: $(BUILD_DIR)/$(BINARY) + $(BUILD_DIR)/$(WORKER_BINARY)"

.PHONY: build-cuda
build-cuda: llama-cuda
	@echo "Building $(BINARY) + $(WORKER_BINARY) (CUDA)..."
	mkdir -p $(BUILD_DIR)
	$(BASE_CGO) CGO_LDFLAGS="$(BASE_LDFLAGS) -lcuda -lcudart" $(GO_BUILD)
	$(BASE_CGO) CGO_LDFLAGS="$(BASE_LDFLAGS) -lcuda -lcudart" $(GO_BUILD_WORKER)
	@echo "Built: $(BUILD_DIR)/$(BINARY) + $(BUILD_DIR)/$(WORKER_BINARY)"

.PHONY: build-cpu
build-cpu: llama-cpu
	@echo "Building $(BINARY) + $(WORKER_BINARY) (CPU)..."
	mkdir -p $(BUILD_DIR)
	$(BASE_CGO) CGO_LDFLAGS="$(BASE_LDFLAGS)" $(GO_BUILD)
	$(BASE_CGO) CGO_LDFLAGS="$(BASE_LDFLAGS)" $(GO_BUILD_WORKER)
	@echo "Built: $(BUILD_DIR)/$(BINARY) + $(BUILD_DIR)/$(WORKER_BINARY)"

# --- Development ---

.PHONY: run
run:
	$(BASE_CGO) CGO_LDFLAGS="$(METAL_LDFLAGS)" go run ./cmd/server

.PHONY: test
test:
	$(BASE_CGO) CGO_LDFLAGS="$(TEST_LDFLAGS)" go test ./... -v

.PHONY: lint
lint:
	go vet ./...

.PHONY: parity
parity:
	@test -n "$(ORCHESTRA_PARITY_GGUF)" || (echo "Set ORCHESTRA_PARITY_GGUF=/absolute/path/model.gguf" && exit 1)
	ORCHESTRA_PARITY_GGUF="$(ORCHESTRA_PARITY_GGUF)" scripts/parity-ollama-runtime.sh

.PHONY: install
install:
	@test -x "$(BUILD_DIR)/$(BINARY)" || (echo "Missing $(BUILD_DIR)/$(BINARY); run make build-metal, build-cuda, or build-cpu first" && exit 1)
	@test -x "$(BUILD_DIR)/$(WORKER_BINARY)" || (echo "Missing $(BUILD_DIR)/$(WORKER_BINARY); build both runtime binaries first" && exit 1)
	mkdir -p "$(INSTALL_DIR)"
	cp "$(BUILD_DIR)/$(BINARY)" "$(INSTALL_DIR)/$(BINARY)"
	cp "$(BUILD_DIR)/$(WORKER_BINARY)" "$(INSTALL_DIR)/$(WORKER_BINARY)"
	chmod 755 "$(INSTALL_DIR)/$(BINARY)" "$(INSTALL_DIR)/$(WORKER_BINARY)"
ifeq ($(shell uname -s),Darwin)
	# Re-sign after copying. macOS can reject linker-signed Go/CGo pages when
	# the installed file metadata differs from the build artifact.
	codesign --force --sign - "$(INSTALL_DIR)/$(BINARY)"
	codesign --force --sign - "$(INSTALL_DIR)/$(WORKER_BINARY)"
endif
	@echo "Installed: $(INSTALL_DIR)/$(BINARY) + $(INSTALL_DIR)/$(WORKER_BINARY)"

# --- Submodule ---

.PHONY: submodule-init
submodule-init:
	git submodule update --init --recursive $(LLAMA_DIR)

# --- Cleanup ---

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/
	cd $(LLAMA_DIR) && rm -rf build/ 2>/dev/null || true

.PHONY: clean-models
clean-models:
	rm -rf ~/.orchestra/models/*

.PHONY: help
help:
	@echo "Orchestra Runtime - Build targets:"
	@echo ""
	@echo "  make build-metal   Build with Metal backend (macOS Apple Silicon)"
	@echo "  make build-cuda    Build with CUDA backend (NVIDIA GPU)"
	@echo "  make build-cpu     Build with CPU-only backend"
	@echo "  make run           Run in development mode"
	@echo "  make install       Install both built binaries to ~/.orchestra/bin"
	@echo "  make test          Run tests"
	@echo "  make parity        Compare Ollama and runtime on one source GGUF"
	@echo "  make clean         Clean build artifacts"
	@echo "  make submodule-init  Initialize llama.cpp submodule"
	@echo "  make help          Show this help"

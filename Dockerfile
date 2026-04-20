# Orchestra Runtime - CPU build
# For GPU builds, build natively with `make build-metal` or `make build-cuda`

FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates cmake make g++ linux-headers

WORKDIR /build

# Copy llama.cpp and build it
COPY llama.cpp/ llama.cpp/
RUN cd llama.cpp && cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

# Copy Go source and build
COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=1 \
    CGO_CFLAGS="-I/build/llama.cpp/include" \
    CGO_LDFLAGS="-L/build/llama.cpp/build/src -L/build/llama.cpp/build/ggml/src -lllama -lggml -lstdc++ -lm" \
    go build -ldflags="-s -w" -o /orchestra-runtime ./cmd/server

# Runtime
FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata libstdc++ libgcc

WORKDIR /app
COPY --from=builder /orchestra-runtime /app/orchestra-runtime

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:8100/health || exit 1

ENTRYPOINT ["/app/orchestra-runtime"]

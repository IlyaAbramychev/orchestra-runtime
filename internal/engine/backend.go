package engine

import (
	"context"
	"time"
)

// Backend is the inference surface used by services/handlers. We have two
// implementations:
//
//   - *Engine          — in-process, direct CGo into llama.cpp (this package).
//   - *supervisor.Remote — out-of-process proxy to a worker subprocess
//     (internal/supervisor package). Protects the host from C crashes.
//
// Both are selected at startup based on ORCHESTRA_USE_SUBPROCESS=1.
// Method signatures mirror *Engine so the refactor is mechanical.
//
// Adding a new method?
//  1. Put it on this interface.
//  2. Implement on *Engine (package engine).
//  3. Implement on *supervisor.Remote (forwards over RPC).
//  4. Add the RPC method name in internal/rpc/protocol.go.
//  5. Wire it through in cmd/worker/main.go.
type Backend interface {
	// Lifecycle
	InitBackend()
	FreeBackend()
	Close() error

	// Model management
	LoadModel(modelID, path string, opts LoadOptions) error
	UnloadModel()
	IsLoaded() bool
	LoadedModelID() string
	State() string
	ModelDesc() string

	// Inference
	Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (*CompletionResult, error)
	CompleteStream(ctx context.Context, messages []ChatMessage, params CompletionParams) (<-chan CompletionChunk, error)
	Embed(ctx context.Context, text string, normalize bool) (*EmbeddingResult, error)

	// Idle/keep-alive management
	SetIdleTimeout(d time.Duration)
	IdleTimeout() time.Duration
	ApplyKeepAlive(seconds *int64)
	MarkUsed()
}

// Compile-time check: *Engine satisfies Backend.
var _ Backend = (*Engine)(nil)

// Close is provided so Backend has uniform teardown. For the in-process
// Engine, closing frees the llama.cpp backend state.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.unloadLocked()
	// llama_backend_free is not called — it's process-global and fine to
	// leave until exit. Calling it here would race with any other *Engine
	// (we have none today but want to be future-safe).
	return nil
}

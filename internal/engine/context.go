package engine

import (
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// Engine manages the loaded model and context.
type Engine struct {
	mu      sync.RWMutex
	model   *llamaModel
	ctx     *llamaContext
	vocab   *llamaVocab
	sampler *llamaSampler

	modelID   string
	modelPath string
	state     string // idle, loading, ready, error

	// Idle auto-unload (inspired by Ollama's OLLAMA_KEEP_ALIVE):
	// release GPU/RAM after the model hasn't been used for `idleTimeout`.
	// Zero disables auto-unload.
	idleTimeout time.Duration
	lastUsedAt  time.Time
	idleStop    chan struct{}
}

const (
	StateIdle    = "idle"
	StateLoading = "loading"
	StateReady   = "ready"
	StateError   = "error"
)

func New() *Engine {
	return &Engine{state: StateIdle}
}

// SetIdleTimeout enables automatic model unload after `d` of inactivity.
// Call after `InitBackend` but before serving requests. Pass 0 to disable.
func (e *Engine) SetIdleTimeout(d time.Duration) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.idleTimeout = d
}

// IdleTimeout returns the currently configured idle timeout (0 if disabled).
func (e *Engine) IdleTimeout() time.Duration {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.idleTimeout
}

// MarkUsed records that an inference request just completed. Streaming and
// non-streaming completions should both call this — extends the idle deadline.
func (e *Engine) MarkUsed() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.lastUsedAt = time.Now()
}

// ApplyKeepAlive applies a per-request keep-alive hint. Semantics (Ollama):
//   seconds == nil   — no override, default idle timeout remains
//   seconds == 0     — unload the model right now
//   seconds  > 0     — stay loaded for this many seconds after last use
//   seconds  < 0     — stay loaded forever (disables auto-unload this session)
// Safe to call from handler goroutines after the stream finishes.
func (e *Engine) ApplyKeepAlive(seconds *int64) {
	if seconds == nil {
		return
	}
	if *seconds == 0 {
		e.UnloadModel()
		return
	}
	e.mu.Lock()
	if *seconds < 0 {
		// "forever" — disable watcher by stopping it; it can be re-enabled by
		// the next LoadModel or a server-side SetIdleTimeout call.
		if e.idleStop != nil {
			close(e.idleStop)
			e.idleStop = nil
		}
		e.idleTimeout = 0
	} else {
		e.idleTimeout = time.Duration(*seconds) * time.Second
		// Restart watcher so the new timeout takes effect immediately.
		if e.state == StateReady && e.model != nil {
			e.startIdleWatcherLocked()
		}
	}
	e.lastUsedAt = time.Now()
	e.mu.Unlock()
}

// startIdleWatcher launches a background goroutine that unloads the model
// once `idleTimeout` elapses without a MarkUsed call. Must be invoked with
// e.mu held and only when a model is actually loaded.
func (e *Engine) startIdleWatcherLocked() {
	if e.idleTimeout <= 0 {
		return
	}
	// Stop any previous watcher (defensive — LoadModel already unloads first).
	if e.idleStop != nil {
		close(e.idleStop)
	}
	stop := make(chan struct{})
	e.idleStop = stop
	timeout := e.idleTimeout
	go e.idleWatcher(stop, timeout)
}

func (e *Engine) idleWatcher(stop chan struct{}, timeout time.Duration) {
	// Check a bit more often than the timeout so the latency of auto-unload
	// is at most ~10% of the window.
	tick := timeout / 10
	if tick < 30*time.Second {
		tick = 30 * time.Second
	}
	t := time.NewTicker(tick)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			e.mu.Lock()
			if e.state != StateReady || e.model == nil {
				e.mu.Unlock()
				return
			}
			if time.Since(e.lastUsedAt) >= timeout {
				slog.Info("idle auto-unload triggered",
					"model_id", e.modelID,
					"idle_for", time.Since(e.lastUsedAt).Round(time.Second))
				e.unloadLocked()
				e.mu.Unlock()
				return
			}
			e.mu.Unlock()
		}
	}
}

func (e *Engine) InitBackend() {
	llamaBackendInit()
	slog.Info("llama.cpp backend initialized")
}

func (e *Engine) FreeBackend() {
	llamaBackendFree()
	slog.Info("llama.cpp backend freed")
}

// LoadOptions bundles every tunable we expose to callers. Missing fields fall
// back to sensible defaults via `normalize()`.
//
// Callers should set explicitly:
//   GPULayers    number of layers on GPU (-1 = all, 0 = CPU only)
//   CtxSize      n_ctx; ≤ model's trained ctx to avoid RoPE extrapolation
//   Threads      CPU threads for inference; 0 = auto
//
// Advanced (leave zero to inherit llama.cpp's default):
//   BatchSize       n_batch, ≥ NCtx is fine but rarely needed
//   RopeFreqBase    override RoPE base (0 = from GGUF)
//   RopeFreqScale   override RoPE scale (0 = from GGUF)
//   FlashAttn       -1 auto (default), 0 off, 1 on
//   OffloadKQV      move KV cache to GPU (true on GPU systems)
//   UseMmap         mmap the GGUF (true by default; false = full RAM copy)
//   UseMlock        pin pages in RAM (guarantees no swap)
//   TypeK/TypeV     KV cache quant: "" (f16 default), "q8_0", "q4_0", …
type LoadOptions struct {
	GPULayers     int
	CtxSize       int
	Threads       int
	BatchSize     int
	RopeFreqBase  float32
	RopeFreqScale float32
	FlashAttn     int
	OffloadKQV    bool
	UseMmap       bool
	UseMlock      bool
	TypeK         string
	TypeV         string
}

// DefaultLoadOptions returns the set of values we use when callers pass zero
// or a legacy signature. Matches the old behaviour prior to the expanded API.
func DefaultLoadOptions() LoadOptions {
	return LoadOptions{
		GPULayers:  -1,
		CtxSize:    4096,
		Threads:    0,
		BatchSize:  512,
		FlashAttn:  -1, // auto
		OffloadKQV: true,
		UseMmap:    true,
		UseMlock:   false,
	}
}

// normalize fills in anything the caller left as a zero value with a sensible
// default. Keeps external callers concise without losing ergonomics.
func (o *LoadOptions) normalize() {
	if o.CtxSize == 0 {
		o.CtxSize = 4096
	}
	if o.BatchSize == 0 {
		o.BatchSize = 512
	}
	if o.GPULayers == 0 {
		// 0 is ambiguous with "CPU only"; we pick -1 (auto) here to match
		// llama.cpp convention. Caller wanting CPU-only must pass a negative-
		// number-in-disguise via ModelParams directly.
		o.GPULayers = -1
	}
}

// LoadModel loads a GGUF model into memory using `opts`.
func (e *Engine) LoadModel(modelID, path string, opts LoadOptions) error {
	opts.normalize()

	e.mu.Lock()
	defer e.mu.Unlock()

	// Unload any existing model
	e.unloadLocked()

	e.state = StateLoading
	slog.Info("loading model",
		"path", path,
		"gpu_layers", opts.GPULayers,
		"ctx_size", opts.CtxSize,
		"batch", opts.BatchSize,
		"flash_attn", opts.FlashAttn,
		"offload_kqv", opts.OffloadKQV,
		"use_mmap", opts.UseMmap,
		"use_mlock", opts.UseMlock,
		"type_k", opts.TypeK,
		"type_v", opts.TypeV,
	)

	model, err := llamaModelLoad(path, ModelParams{
		NGPULayers: opts.GPULayers,
		UseMmap:    opts.UseMmap,
		UseMlock:   opts.UseMlock,
	})
	if err != nil {
		e.state = StateError
		return fmt.Errorf("load model: %w", err)
	}

	ctx, err := llamaNewContext(model, ContextParams{
		NCtx:          opts.CtxSize,
		NBatch:        opts.BatchSize,
		NThreads:      opts.Threads,
		NSeqMax:       1,
		RopeFreqBase:  opts.RopeFreqBase,
		RopeFreqScale: opts.RopeFreqScale,
		FlashAttn:     opts.FlashAttn,
		OffloadKQV:    opts.OffloadKQV,
		TypeK:         opts.TypeK,
		TypeV:         opts.TypeV,
	})
	if err != nil {
		model.Free()
		e.state = StateError
		return fmt.Errorf("create context: %w", err)
	}

	e.model = model
	e.ctx = ctx
	e.vocab = model.Vocab()
	e.modelID = modelID
	e.modelPath = path
	e.state = StateReady
	e.lastUsedAt = time.Now()

	// Kick off the idle-unload watcher for this load session.
	e.startIdleWatcherLocked()

	slog.Info("model loaded",
		"id", modelID,
		"desc", model.Desc(),
		"size_bytes", model.Size(),
		"n_params", model.NParams(),
		"n_ctx_train", model.NCtxTrain(),
		"n_ctx", ctx.NCtx(),
		"idle_timeout", e.idleTimeout,
	)
	return nil
}

// UnloadModel frees the current model and context.
func (e *Engine) UnloadModel() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.unloadLocked()
}

func (e *Engine) unloadLocked() {
	if e.idleStop != nil {
		close(e.idleStop)
		e.idleStop = nil
	}
	if e.sampler != nil {
		e.sampler.Free()
		e.sampler = nil
	}
	if e.ctx != nil {
		e.ctx.Free()
		e.ctx = nil
	}
	if e.model != nil {
		e.model.Free()
		e.model = nil
	}
	e.vocab = nil
	e.modelID = ""
	e.modelPath = ""
	e.state = StateIdle
}

// IsLoaded returns true if a model is loaded and ready for inference.
func (e *Engine) IsLoaded() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.state == StateReady && e.model != nil && e.ctx != nil
}

// State returns the current engine state.
func (e *Engine) State() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.state
}

// LoadedModelID returns the ID of the currently loaded model, or empty string.
func (e *Engine) LoadedModelID() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.modelID
}

// ModelDesc returns the description of the loaded model.
func (e *Engine) ModelDesc() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.model == nil {
		return ""
	}
	return e.model.Desc()
}

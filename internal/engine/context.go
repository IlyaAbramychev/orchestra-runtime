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
	mtmd    *mtmdContext
	sampler *llamaSampler
	// batchSize mirrors llama_context n_batch for safe prompt prefill chunking.
	batchSize int

	modelID   string
	modelPath string
	loadOpts  LoadOptions
	loadedAt  time.Time
	lastError string
	state     string // idle, loading, ready, error

	// Idle auto-unload (inspired by Ollama's OLLAMA_KEEP_ALIVE):
	// release GPU/RAM after the model hasn't been used for `idleTimeout`.
	// Zero disables auto-unload.
	idleTimeout time.Duration
	lastUsedAt  time.Time
	idleTimer   *time.Timer
}

const (
	StateIdle       = "idle"
	StateLoading    = "loading"
	StateReady      = "ready"
	StateGenerating = "generating"
	StateUnloading  = "unloading"
	StateError      = "error"
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
	if d <= 0 {
		e.stopIdleTimerLocked()
		return
	}
	if e.state == StateReady && e.model != nil {
		e.resetIdleTimerLocked(d)
	}
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
	e.markUsedLocked()
}

// ApplyKeepAlive applies a per-request keep-alive hint. Semantics (Ollama):
//
//	seconds == nil   — no override, default idle timeout remains
//	seconds == 0     — unload the model right now
//	seconds  > 0     — stay loaded for this many seconds after last use
//	seconds  < 0     — stay loaded forever (disables auto-unload this session)
//
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
		// "forever" — disable timer; it can be re-enabled by the next LoadModel
		// or a server-side SetIdleTimeout call.
		e.stopIdleTimerLocked()
		e.idleTimeout = 0
	} else {
		e.idleTimeout = time.Duration(*seconds) * time.Second
		if e.state == StateReady && e.model != nil {
			e.resetIdleTimerLocked(e.idleTimeout)
		}
	}
	e.markUsedLocked()
	e.mu.Unlock()
}

func (e *Engine) markUsedLocked() {
	e.lastUsedAt = time.Now()
	if e.idleTimeout > 0 && e.state == StateReady && e.model != nil {
		e.resetIdleTimerLocked(e.idleTimeout)
	}
}

func (e *Engine) resetIdleTimerLocked(d time.Duration) {
	if d <= 0 {
		e.stopIdleTimerLocked()
		return
	}
	if e.idleTimer != nil {
		e.idleTimer.Reset(d)
		return
	}
	e.idleTimer = time.AfterFunc(d, e.onIdleTimer)
}

func (e *Engine) stopIdleTimerLocked() {
	if e.idleTimer != nil {
		e.idleTimer.Stop()
		e.idleTimer = nil
	}
}

func (e *Engine) onIdleTimer() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.state != StateReady || e.model == nil || e.idleTimeout <= 0 {
		e.idleTimer = nil
		return
	}
	idleFor := time.Since(e.lastUsedAt)
	if idleFor < e.idleTimeout {
		e.resetIdleTimerLocked(e.idleTimeout - idleFor)
		return
	}
	slog.Info("idle auto-unload triggered",
		"model_id", e.modelID,
		"idle_for", idleFor.Round(time.Second))
	e.unloadLocked()
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
//
//	GPULayers    number of layers on GPU (-1 = all, 0 = CPU only)
//	CtxSize      n_ctx; ≤ model's trained ctx to avoid RoPE extrapolation
//	Threads      CPU threads for inference; 0 = auto
//
// Advanced (leave zero to inherit llama.cpp's default):
//
//	BatchSize       n_batch, ≥ NCtx is fine but rarely needed
//	RopeFreqBase    override RoPE base (0 = from GGUF)
//	RopeFreqScale   override RoPE scale (0 = from GGUF)
//	FlashAttn       -1 auto (default), 0 off, 1 on
//	OffloadKQV      move KV cache to GPU (true on GPU systems)
//	UseMmap         mmap the GGUF (true by default; false = full RAM copy)
//	UseMlock        pin pages in RAM (guarantees no swap)
//	TypeK/TypeV     KV cache quant: "" (f16 default), "q8_0", "q4_0", …
//	MMProjPath      optional multimodal projector GGUF for image inputs
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
	MMProjPath    string
	MMProjUseGPU  bool
}

// DefaultLoadOptions returns the set of values we use when callers pass zero
// or a legacy signature. Matches the old behaviour prior to the expanded API.
func DefaultLoadOptions() LoadOptions {
	return LoadOptions{
		GPULayers:  -1,
		CtxSize:    4096,
		Threads:    0,
		BatchSize:  1024,
		FlashAttn:  -1, // auto
		OffloadKQV: true,
		UseMmap:    true,
		UseMlock:   false,
		MMProjUseGPU: true,
	}
}

// normalize fills in anything the caller left as a zero value with a sensible
// default. Keeps external callers concise without losing ergonomics.
func (o *LoadOptions) normalize() {
	if o.CtxSize == 0 {
		o.CtxSize = 4096
	}
	if o.BatchSize <= 0 {
		// LM Studio-style default: use a larger prefill batch on big contexts
		// while keeping memory bounded for smaller models/devices.
		if o.CtxSize >= 2048 {
			o.BatchSize = 2048
		} else {
			o.BatchSize = o.CtxSize
		}
	}
	if o.BatchSize > o.CtxSize {
		o.BatchSize = o.CtxSize
	}
	// LM Studio-like floor on large contexts: too small n_batch hurts long-prompt
	// prefill throughput and increases edge-case instability in some model/backends.
	if o.CtxSize >= 32768 && o.BatchSize < 1024 {
		if o.CtxSize >= 1024 {
			o.BatchSize = 1024
		} else {
			o.BatchSize = o.CtxSize
		}
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
	e.lastError = ""
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
		"mmproj_path", opts.MMProjPath,
	)

	model, err := llamaModelLoad(path, ModelParams{
		NGPULayers: opts.GPULayers,
		UseMmap:    opts.UseMmap,
		UseMlock:   opts.UseMlock,
	})
	if err != nil {
		e.state = StateError
		e.lastError = err.Error()
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
		e.lastError = err.Error()
		return fmt.Errorf("create context: %w", err)
	}
	mtmd, err := mtmdContextLoad(opts.MMProjPath, model, opts)
	if err != nil {
		ctx.Free()
		model.Free()
		e.state = StateError
		e.lastError = err.Error()
		return err
	}

	e.model = model
	e.ctx = ctx
	e.vocab = model.Vocab()
	e.mtmd = mtmd
	e.batchSize = opts.BatchSize
	e.modelID = modelID
	e.modelPath = path
	e.loadOpts = opts
	e.loadedAt = time.Now().UTC()
	e.lastError = ""
	e.state = StateReady
	e.markUsedLocked()

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
	e.stopIdleTimerLocked()
	if e.sampler != nil {
		e.sampler.Free()
		e.sampler = nil
	}
	if e.mtmd != nil {
		e.mtmd.Free()
		e.mtmd = nil
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
	e.batchSize = 0
	e.modelID = ""
	e.modelPath = ""
	e.loadOpts = LoadOptions{}
	e.loadedAt = time.Time{}
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

// LoadedContextSize returns the effective llama.cpp context window for the
// currently loaded model. It can differ from the requested n_ctx because
// llama.cpp may round or clamp context parameters during context creation.
func (e *Engine) LoadedContextSize() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.ctx == nil {
		return 0
	}
	return e.ctx.NCtx()
}

func (e *Engine) LoadedOptions() LoadOptions {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.loadOpts
}

func (e *Engine) LoadedAt() time.Time {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.loadedAt
}

func (e *Engine) LastError() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.lastError
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

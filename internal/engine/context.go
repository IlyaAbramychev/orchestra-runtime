package engine

import (
	"fmt"
	"log/slog"
	"sync"
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

func (e *Engine) InitBackend() {
	llamaBackendInit()
	slog.Info("llama.cpp backend initialized")
}

func (e *Engine) FreeBackend() {
	llamaBackendFree()
	slog.Info("llama.cpp backend freed")
}

// LoadModel loads a GGUF model into memory.
func (e *Engine) LoadModel(modelID, path string, gpuLayers, ctxSize, threads int) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Unload any existing model
	e.unloadLocked()

	e.state = StateLoading
	slog.Info("loading model", "path", path, "gpu_layers", gpuLayers, "ctx_size", ctxSize)

	model, err := llamaModelLoad(path, ModelParams{
		NGPULayers: gpuLayers,
		UseMmap:    true,
	})
	if err != nil {
		e.state = StateError
		return fmt.Errorf("load model: %w", err)
	}

	ctx, err := llamaNewContext(model, ContextParams{
		NCtx:     ctxSize,
		NBatch:   512,
		NThreads: threads,
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

	slog.Info("model loaded",
		"id", modelID,
		"desc", model.Desc(),
		"size_bytes", model.Size(),
		"n_params", model.NParams(),
		"n_ctx_train", model.NCtxTrain(),
		"n_ctx", ctx.NCtx(),
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

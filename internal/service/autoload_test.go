package service

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/storage"
)

type autoLoadBackend struct {
	loadedID string
	loads    int
	lastOpts engine.LoadOptions
}

func (b *autoLoadBackend) InitBackend()          {}
func (b *autoLoadBackend) FreeBackend()          {}
func (b *autoLoadBackend) Close() error          { return nil }
func (b *autoLoadBackend) UnloadModel()          { b.loadedID = "" }
func (b *autoLoadBackend) IsLoaded() bool        { return b.loadedID != "" }
func (b *autoLoadBackend) LoadedModelID() string { return b.loadedID }
func (b *autoLoadBackend) State() string         { return engine.StateReady }
func (b *autoLoadBackend) ModelDesc() string     { return "" }
func (b *autoLoadBackend) SetIdleTimeout(time.Duration) {
}
func (b *autoLoadBackend) IdleTimeout() time.Duration { return 0 }
func (b *autoLoadBackend) ApplyKeepAlive(*int64)      {}
func (b *autoLoadBackend) MarkUsed()                  {}

func (b *autoLoadBackend) LoadModel(id, _ string, opts engine.LoadOptions) error {
	b.loadedID = id
	b.loads++
	b.lastOpts = opts
	return nil
}

func (b *autoLoadBackend) Complete(
	context.Context,
	[]engine.ChatMessage,
	engine.CompletionParams,
) (*engine.CompletionResult, error) {
	return &engine.CompletionResult{Text: "ok", FinishReason: "stop"}, nil
}

func (b *autoLoadBackend) CompleteStream(
	context.Context,
	[]engine.ChatMessage,
	engine.CompletionParams,
) (<-chan engine.CompletionChunk, error) {
	ch := make(chan engine.CompletionChunk, 1)
	ch <- engine.CompletionChunk{Done: true, FinishReason: "stop"}
	close(ch)
	return ch, nil
}

func (b *autoLoadBackend) Embed(context.Context, string, bool) (*engine.EmbeddingResult, error) {
	return &engine.EmbeddingResult{Vector: []float32{1}}, nil
}

func TestInferenceAutoLoadsRequestedModel(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "qwen3.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "model-1",
		Name:     "qwen3",
		Filename: "qwen3.gguf",
		Status:   "ready",
		FilePath: modelPath,
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	manager.SetDefaultLoadOptions(engine.LoadOptions{
		GPULayers: 2,
		CtxSize:   8192,
		Threads:   4,
		BatchSize: 512,
		UseMmap:   true,
	})
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "qwen3",
		Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if backend.loadedID != "model-1" {
		t.Fatalf("expected model-1 loaded, got %q", backend.loadedID)
	}
	if backend.loads != 1 {
		t.Fatalf("expected one load, got %d", backend.loads)
	}
	if backend.lastOpts.CtxSize != 8192 || backend.lastOpts.GPULayers != 2 || backend.lastOpts.Threads != 4 {
		t.Fatalf("autoload did not use configured defaults: %+v", backend.lastOpts)
	}

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "qwen3.gguf",
		Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("second complete: %v", err)
	}
	if backend.loads != 1 {
		t.Fatalf("expected no duplicate load, got %d", backend.loads)
	}
}

func TestEmbeddingAutoLoadsRequestedModel(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "embed.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "embed-1",
		Name:     "embedder",
		Filename: "embed.gguf",
		Status:   "ready",
		FilePath: modelPath,
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	embedding := NewEmbeddingServiceWithScheduler(scheduler)
	embedding.SetModelLoader(manager)

	results, err := embedding.EmbedForModel(context.Background(), "embedder", []string{"hello"}, true)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected one embedding result, got %d", len(results))
	}
	if backend.loadedID != "embed-1" {
		t.Fatalf("expected embed-1 loaded, got %q", backend.loadedID)
	}
}

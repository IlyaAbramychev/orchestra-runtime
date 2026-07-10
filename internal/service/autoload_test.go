package service

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/storage"
)

type autoLoadBackend struct {
	loadedID   string
	loads      int
	lastOpts   engine.LoadOptions
	lastParams engine.CompletionParams
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
	ctx context.Context,
	messages []engine.ChatMessage,
	params engine.CompletionParams,
) (*engine.CompletionResult, error) {
	b.lastParams = params
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

func TestDefaultLoadOptionsForModelUsesRecommendationOnlyWithUntouchedDefault(t *testing.T) {
	registry, err := storage.NewModelRegistry(t.TempDir())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:     "recommended",
		Name:   "qwen",
		Status: "ready",
		RecommendedSettings: storage.RecommendedModelSettings{
			ContextSize: 8192,
		},
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	manager := NewModelManager(registry, &autoLoadBackend{}, t.TempDir())
	if got := manager.DefaultLoadOptionsForModel("recommended").CtxSize; got != 8192 {
		t.Fatalf("automatic context = %d; want 8192", got)
	}

	configured := engine.DefaultLoadOptions()
	configured.CtxSize = 32768
	manager.SetDefaultLoadOptions(configured)
	if got := manager.DefaultLoadOptionsForModel("recommended").CtxSize; got != 32768 {
		t.Fatalf("explicit global context = %d; want 32768", got)
	}
}

func TestInferenceAutoLoadsModelScopedMMProj(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "qwen2.5-vl.gguf")
	mmprojPath := filepath.Join(tmp, "qwen2.5-vl-mmproj-f16.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := os.WriteFile(mmprojPath, []byte("mmproj"), 0644); err != nil {
		t.Fatalf("write mmproj: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:             "model-1",
		Name:           "qwen2.5-vl",
		Filename:       "qwen2.5-vl.gguf",
		Status:         "ready",
		FilePath:       modelPath,
		MMProjFilename: filepath.Base(mmprojPath),
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "qwen2.5-vl",
		Messages: []model.ChatMessage{{Role: "user", Content: "describe", Images: []string{"aGVsbG8="}}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if backend.lastOpts.MMProjPath != mmprojPath {
		t.Fatalf("expected mmproj path %q, got %q", mmprojPath, backend.lastOpts.MMProjPath)
	}
}

func TestImportFromDirectoryDetectsMMProjSibling(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &autoLoadBackend{}
	manager := NewModelManager(registry, backend, tmp)

	modelDir := filepath.Join(tmp, "author", "vision-model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	modelPath := filepath.Join(modelDir, "vision-model-q4_k_m.gguf")
	mmprojPath := filepath.Join(modelDir, "vision-model-mmproj-f16.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := os.WriteFile(mmprojPath, []byte("mmproj"), 0644); err != nil {
		t.Fatalf("write mmproj: %v", err)
	}

	imported, err := manager.ImportFromDirectory(tmp)
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if len(imported) != 1 {
		t.Fatalf("expected one imported model, got %d", len(imported))
	}
	if imported[0].MMProjFilename != filepath.Base(mmprojPath) {
		t.Fatalf("expected mmproj filename %q, got %q", filepath.Base(mmprojPath), imported[0].MMProjFilename)
	}
}

func TestImportFromDirectoryLeavesMMProjUnsetWhenAmbiguous(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &autoLoadBackend{}
	manager := NewModelManager(registry, backend, tmp)

	modelDir := filepath.Join(tmp, "author", "vision-model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	modelPath := filepath.Join(modelDir, "vision-model.gguf")
	mmprojA := filepath.Join(modelDir, "mmproj-a.gguf")
	mmprojB := filepath.Join(modelDir, "mmproj-b.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := os.WriteFile(mmprojA, []byte("mmproj"), 0644); err != nil {
		t.Fatalf("write mmproj a: %v", err)
	}
	if err := os.WriteFile(mmprojB, []byte("mmproj"), 0644); err != nil {
		t.Fatalf("write mmproj b: %v", err)
	}

	imported, err := manager.ImportFromDirectory(tmp)
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if len(imported) != 1 {
		t.Fatalf("expected one imported model, got %d", len(imported))
	}
	if imported[0].MMProjFilename != "" {
		t.Fatalf("expected ambiguous mmproj detection to remain unset, got %q", imported[0].MMProjFilename)
	}
}

func TestInferenceRejectsAmbiguousAutoDetectedMMProj(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "vision.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	for _, name := range []string{"mmproj-a.gguf", "mmproj-b.gguf"} {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte("mmproj"), 0644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "model-1",
		Name:     "vision",
		Filename: "vision.gguf",
		Status:   "ready",
		FilePath: modelPath,
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "vision",
		Messages: []model.ChatMessage{{Role: "user", Content: "describe", Images: []string{"aGVsbG8="}}},
	})
	if err == nil || !strings.Contains(err.Error(), "multiple mmproj files found") {
		t.Fatalf("expected ambiguous mmproj error, got %v", err)
	}
	if backend.loads != 0 {
		t.Fatalf("expected load to be blocked, got %d loads", backend.loads)
	}
}

func TestInferenceRejectsMissingConfiguredMMProj(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "vision.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:             "model-1",
		Name:           "vision",
		Filename:       "vision.gguf",
		Status:         "ready",
		FilePath:       modelPath,
		MMProjFilename: "missing-mmproj.gguf",
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "vision",
		Messages: []model.ChatMessage{{Role: "user", Content: "describe", Images: []string{"aGVsbG8="}}},
	})
	if err == nil || !strings.Contains(err.Error(), "configured mmproj") {
		t.Fatalf("expected missing configured mmproj error, got %v", err)
	}
	if backend.loads != 0 {
		t.Fatalf("expected load to be blocked, got %d loads", backend.loads)
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

func TestInferenceRejectsEmbeddingOnlyModel(t *testing.T) {
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
		Capabilities: storage.ModelCapabilities{
			Embeddings: true,
		},
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "embedder",
		Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "does not support chat") {
		t.Fatalf("expected chat capability error, got %v", err)
	}
	if backend.loads != 0 {
		t.Fatalf("expected no load, got %d", backend.loads)
	}
}

func TestEmbeddingRejectsChatOnlyModel(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "chat.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "chat-1",
		Name:     "chat-model",
		Filename: "chat.gguf",
		Status:   "ready",
		FilePath: modelPath,
		Capabilities: storage.ModelCapabilities{
			Chat: true,
		},
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	embedding := NewEmbeddingServiceWithScheduler(scheduler)
	embedding.SetModelLoader(manager)

	_, err = embedding.EmbedForModel(context.Background(), "chat-model", []string{"hello"}, true)
	if err == nil || !strings.Contains(err.Error(), "does not support embeddings") {
		t.Fatalf("expected embedding capability error, got %v", err)
	}
	if backend.loads != 0 {
		t.Fatalf("expected no load, got %d", backend.loads)
	}
}

func TestInferenceAppliesModelStopTokensByDefault(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "chat.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:         "chat-1",
		Name:       "chat-model",
		Filename:   "chat.gguf",
		Status:     "ready",
		FilePath:   modelPath,
		StopTokens: []string{"<|end|>", "<|stop|>"},
		Capabilities: storage.ModelCapabilities{
			Chat: true,
		},
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "chat-model",
		Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	want := []string{"<|end|>", "<|stop|>"}
	if !reflect.DeepEqual(backend.lastParams.Stop, want) {
		t.Fatalf("expected default stop tokens %v, got %v", want, backend.lastParams.Stop)
	}
}

func TestInferenceRequestStopOverridesModelDefaults(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "chat.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:         "chat-1",
		Name:       "chat-model",
		Filename:   "chat.gguf",
		Status:     "ready",
		FilePath:   modelPath,
		StopTokens: []string{"<|end|>"},
		Capabilities: storage.ModelCapabilities{
			Chat: true,
		},
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "chat-model",
		Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
		Stop:     []string{"CUSTOM"},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	want := []string{"CUSTOM"}
	if !reflect.DeepEqual(backend.lastParams.Stop, want) {
		t.Fatalf("expected request stop tokens %v, got %v", want, backend.lastParams.Stop)
	}
}

func TestInferenceAppliesModelChatTemplateByDefault(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	modelPath := filepath.Join(tmp, "chat.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "chat-1",
		Name:     "chat-model",
		Filename: "chat.gguf",
		Status:   "ready",
		FilePath: modelPath,
		Template: "{{ .Prompt }}",
		Capabilities: storage.ModelCapabilities{
			Chat: true,
		},
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &autoLoadBackend{}
	scheduler := NewRuntimeScheduler(backend, 1)
	manager := NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(manager)

	_, err = inference.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "chat-model",
		Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if backend.lastParams.ChatTemplate != "{{ .Prompt }}" {
		t.Fatalf("expected model chat template, got %q", backend.lastParams.ChatTemplate)
	}
}

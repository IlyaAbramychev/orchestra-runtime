package service

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
)

type fakeBackend struct {
	block              chan struct{}
	started            chan struct{}
	loadBlock          chan struct{}
	loadStart          chan struct{}
	unloaded           chan struct{}
	loadedID           string
	completedWithModel string
	loadErrors         []error
	loadOptions        []engine.LoadOptions
}

func (f *fakeBackend) InitBackend() {}
func (f *fakeBackend) FreeBackend() {}
func (f *fakeBackend) Close() error { return nil }
func (f *fakeBackend) LoadModel(_ string, _ string, opts engine.LoadOptions) error {
	f.loadOptions = append(f.loadOptions, opts)
	if f.loadStart != nil {
		select {
		case f.loadStart <- struct{}{}:
		default:
		}
	}
	if f.loadBlock != nil {
		<-f.loadBlock
	}
	if len(f.loadErrors) > 0 {
		err := f.loadErrors[0]
		f.loadErrors = f.loadErrors[1:]
		return err
	}
	return nil
}

func TestSchedulerRetriesMemoryFailureWithinSingleLifecycleSlot(t *testing.T) {
	backend := &fakeBackend{loadErrors: []error{errors.New("backend out of memory"), nil}}
	scheduler := NewRuntimeScheduler(backend, 2)
	first := engine.DefaultLoadOptions()
	first.CtxSize = 32768
	second := first
	second.CtxSize = 4096
	second.TypeK = "q8_0"
	second.TypeV = "q8_0"

	selected, err := scheduler.LoadModelAttempts(context.Background(), "model", "/tmp/model.gguf", []engine.LoadOptions{first, second})
	if err != nil {
		t.Fatalf("LoadModelAttempts: %v", err)
	}
	if selected.CtxSize != second.CtxSize || len(backend.loadOptions) != 2 {
		t.Fatalf("selected=%+v attempts=%d; want second profile after two attempts", selected, len(backend.loadOptions))
	}
}

func TestSchedulerDoesNotRetryNonMemoryFailure(t *testing.T) {
	backend := &fakeBackend{loadErrors: []error{errors.New("unsupported model format")}}
	scheduler := NewRuntimeScheduler(backend, 2)
	first := engine.DefaultLoadOptions()
	second := first
	second.CtxSize = 2048

	_, err := scheduler.LoadModelAttempts(context.Background(), "model", "/tmp/model.gguf", []engine.LoadOptions{first, second})
	if err == nil || !strings.Contains(err.Error(), "unsupported model format") {
		t.Fatalf("expected original error, got %v", err)
	}
	if len(backend.loadOptions) != 1 {
		t.Fatalf("non-memory failure retried %d times", len(backend.loadOptions))
	}
}
func (f *fakeBackend) UnloadModel() {
	if f.unloaded != nil {
		close(f.unloaded)
	}
}
func (f *fakeBackend) IsLoaded() bool { return true }
func (f *fakeBackend) LoadedModelID() string {
	if f.loadedID != "" {
		return f.loadedID
	}
	return "test"
}
func (f *fakeBackend) State() string     { return engine.StateReady }
func (f *fakeBackend) ModelDesc() string { return "" }
func (f *fakeBackend) Complete(ctx context.Context, _ []engine.ChatMessage, _ engine.CompletionParams) (*engine.CompletionResult, error) {
	f.completedWithModel = f.LoadedModelID()
	select {
	case f.started <- struct{}{}:
	default:
	}
	if f.block == nil {
		return &engine.CompletionResult{FinishReason: "stop"}, nil
	}
	select {
	case <-f.block:
		return &engine.CompletionResult{FinishReason: "stop"}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

type switchingModelLoader struct {
	backend  *fakeBackend
	attempts int
}

func (l *switchingModelLoader) EnsureLoaded(context.Context, string) error { return nil }

func (l *switchingModelLoader) EnsureLoadedFor(_ context.Context, model, _ string) error {
	l.attempts++
	l.backend.loadedID = model
	if l.attempts == 1 {
		// Simulate another queued request switching models after auto-load but
		// before this request acquires the inference slot.
		l.backend.loadedID = "other-model"
	}
	return nil
}

func (l *switchingModelLoader) ResolveModelID(model string) (string, error) {
	return model, nil
}

func (l *switchingModelLoader) DefaultsForModel(string) (ModelRequestDefaults, error) {
	return ModelRequestDefaults{}, nil
}

func TestInferenceNeverRunsAgainstModelSwitchedAfterAutoLoad(t *testing.T) {
	backend := &fakeBackend{}
	svc := NewInferenceService(backend, 2)
	loader := &switchingModelLoader{backend: backend}
	svc.SetModelLoader(loader)

	_, err := svc.Complete(context.Background(), &model.ChatCompletionRequest{
		Model:    "requested-model",
		Messages: []model.ChatMessage{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if backend.completedWithModel != "requested-model" {
		t.Fatalf("inference used %q, want requested-model", backend.completedWithModel)
	}
	if loader.attempts != 2 {
		t.Fatalf("expected model verification to trigger one reload, got %d attempts", loader.attempts)
	}
}

func TestToEngineMessagesPreservesOpenAIToolHistory(t *testing.T) {
	var request model.ChatCompletionRequest
	err := json.Unmarshal([]byte(`{
		"model":"test",
		"messages":[
			{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"file contents"}
		]
	}`), &request)
	if err != nil {
		t.Fatalf("decode request: %v", err)
	}

	got := toEngineMessages(request.Messages)
	if len(got) != 2 || len(got[0].ToolCalls) != 1 || got[0].ToolCalls[0].Name != "read_file" {
		t.Fatalf("assistant tool call was lost: %+v", got)
	}
	if got[1].ToolCallID != "call_1" || got[1].Content != "file contents" {
		t.Fatalf("tool result identity was lost: %+v", got[1])
	}
}
func (f *fakeBackend) CompleteStream(context.Context, []engine.ChatMessage, engine.CompletionParams) (<-chan engine.CompletionChunk, error) {
	ch := make(chan engine.CompletionChunk)
	return ch, nil
}
func (f *fakeBackend) Embed(context.Context, string, bool) (*engine.EmbeddingResult, error) {
	return &engine.EmbeddingResult{}, nil
}
func (f *fakeBackend) SetIdleTimeout(time.Duration) {}
func (f *fakeBackend) IdleTimeout() time.Duration   { return 0 }
func (f *fakeBackend) ApplyKeepAlive(*int64)        {}
func (f *fakeBackend) MarkUsed()                    {}

func TestInferenceQueueRejectsWhenFull(t *testing.T) {
	backend := &fakeBackend{
		block:   make(chan struct{}),
		started: make(chan struct{}, 2),
	}
	svc := NewInferenceService(backend, 1)
	req := &model.ChatCompletionRequest{
		Messages: []model.ChatMessage{{Role: "user", Content: "hello"}},
	}

	firstDone := make(chan error, 1)
	go func() {
		_, err := svc.Complete(context.Background(), req)
		firstDone <- err
	}()

	select {
	case <-backend.started:
	case <-time.After(time.Second):
		t.Fatal("first request did not start")
	}

	secondDone := make(chan error, 1)
	go func() {
		_, err := svc.Complete(context.Background(), req)
		secondDone <- err
	}()

	waitForQueueDepth(t, svc, 1)

	_, err := svc.Complete(context.Background(), req)
	if err == nil || !strings.Contains(err.Error(), "queue full") {
		t.Fatalf("expected queue full error, got %v", err)
	}

	close(backend.block)

	if err := <-firstDone; err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	if err := <-secondDone; err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	if depth := svc.QueueDepth(); depth != 0 {
		t.Fatalf("expected empty queue, got depth %d", depth)
	}
}

func TestSchedulerUnloadWaitsForActiveInference(t *testing.T) {
	backend := &fakeBackend{
		block:    make(chan struct{}),
		started:  make(chan struct{}, 1),
		unloaded: make(chan struct{}),
	}
	scheduler := NewRuntimeScheduler(backend, 1)
	svc := NewInferenceServiceWithScheduler(scheduler)
	req := &model.ChatCompletionRequest{
		Messages: []model.ChatMessage{{Role: "user", Content: "hello"}},
	}

	inferDone := make(chan error, 1)
	go func() {
		_, err := svc.Complete(context.Background(), req)
		inferDone <- err
	}()

	select {
	case <-backend.started:
	case <-time.After(time.Second):
		t.Fatal("inference did not start")
	}

	unloadDone := make(chan error, 1)
	go func() {
		unloadDone <- scheduler.UnloadModel(context.Background())
	}()

	select {
	case <-backend.unloaded:
		t.Fatal("unload ran while inference was active")
	case <-time.After(50 * time.Millisecond):
	}

	close(backend.block)

	if err := <-inferDone; err != nil {
		t.Fatalf("inference failed: %v", err)
	}
	if err := <-unloadDone; err != nil {
		t.Fatalf("unload failed: %v", err)
	}
	select {
	case <-backend.unloaded:
	case <-time.After(time.Second):
		t.Fatal("unload did not run after inference completed")
	}
}

func TestSchedulerReportsGeneratingStateDuringInference(t *testing.T) {
	backend := &fakeBackend{
		block:   make(chan struct{}),
		started: make(chan struct{}, 1),
	}
	scheduler := NewRuntimeScheduler(backend, 1)
	svc := NewInferenceServiceWithScheduler(scheduler)
	req := &model.ChatCompletionRequest{
		Model:    "test",
		Messages: []model.ChatMessage{{Role: "user", Content: "hello"}},
	}

	done := make(chan error, 1)
	go func() {
		_, err := svc.Complete(context.Background(), req)
		done <- err
	}()

	select {
	case <-backend.started:
	case <-time.After(time.Second):
		t.Fatal("inference did not start")
	}
	if got := scheduler.State(); got != engine.StateGenerating {
		t.Fatalf("expected generating state, got %s", got)
	}
	if got := scheduler.ActiveModelID(); got != "test" {
		t.Fatalf("expected active model test, got %q", got)
	}

	close(backend.block)
	if err := <-done; err != nil {
		t.Fatalf("inference failed: %v", err)
	}
	if got := scheduler.State(); got != engine.StateReady {
		t.Fatalf("expected ready after inference, got %s", got)
	}
}

func TestSchedulerReportsLoadingStateDuringLoad(t *testing.T) {
	backend := &fakeBackend{
		loadStart: make(chan struct{}, 1),
		loadBlock: make(chan struct{}),
	}
	scheduler := NewRuntimeScheduler(backend, 1)

	done := make(chan error, 1)
	go func() {
		done <- scheduler.LoadModel(context.Background(), "model-1", "/tmp/model.gguf", engine.DefaultLoadOptions())
	}()

	select {
	case <-backend.loadStart:
	case <-time.After(time.Second):
		t.Fatal("load did not start")
	}
	snapshot := scheduler.Snapshot()
	if snapshot.State != engine.StateLoading {
		t.Fatalf("expected loading state, got %s", snapshot.State)
	}
	if snapshot.ActiveModelID != "model-1" {
		t.Fatalf("expected active model model-1, got %q", snapshot.ActiveModelID)
	}

	close(backend.loadBlock)
	if err := <-done; err != nil {
		t.Fatalf("load failed: %v", err)
	}
}

func waitForQueueDepth(t *testing.T, svc *InferenceService, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if got := svc.QueueDepth(); got == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("queue depth did not become %d, got %d", want, svc.QueueDepth())
}

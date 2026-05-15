package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
	"github.com/operium/orchestra-runtime/internal/storage"
)

func TestModelStatusIncludesRuntimeSnapshot(t *testing.T) {
	registry, err := storage.NewModelRegistry(t.TempDir())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "test",
		Name:     "test-model",
		Filename: "test.gguf",
		Status:   "ready",
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{block: make(chan struct{})}
	scheduler := service.NewRuntimeScheduler(backend, 2)
	manager := service.NewModelManagerWithScheduler(registry, scheduler, t.TempDir())
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	handler := NewModelsHandler(manager, backend)

	done := make(chan error, 1)
	go func() {
		_, err := inference.Complete(context.Background(), &model.ChatCompletionRequest{
			Model:    "test",
			Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
		})
		done <- err
	}()
	waitForSchedulerState(t, scheduler, engine.StateGenerating)

	req := httptest.NewRequest(http.MethodGet, "/api/models/test/status", nil)
	routeCtx := chi.NewRouteContext()
	routeCtx.URLParams.Add("id", "test")
	req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, routeCtx))
	rec := httptest.NewRecorder()

	handler.Status(rec, req)

	close(backend.block)
	if err := <-done; err != nil {
		t.Fatalf("inference failed: %v", err)
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.ModelStatusResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Status != engine.StateGenerating {
		t.Fatalf("status = %q", resp.Status)
	}
	if resp.RuntimeState != engine.StateGenerating {
		t.Fatalf("runtime_state = %q", resp.RuntimeState)
	}
	if !resp.Active {
		t.Fatal("expected active model")
	}
}

func TestModelListsKeepActiveModelVisible(t *testing.T) {
	registry, err := storage.NewModelRegistry(t.TempDir())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "test",
		Name:     "test-model",
		Filename: "test.gguf",
		Status:   "ready",
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{block: make(chan struct{})}
	scheduler := service.NewRuntimeScheduler(backend, 2)
	manager := service.NewModelManagerWithScheduler(registry, scheduler, t.TempDir())
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	handler := NewModelsHandler(manager, backend)

	done := make(chan error, 1)
	go func() {
		_, err := inference.Complete(context.Background(), &model.ChatCompletionRequest{
			Model:    "test",
			Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
		})
		done <- err
	}()
	waitForSchedulerState(t, scheduler, engine.StateGenerating)

	openAIReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	openAIRec := httptest.NewRecorder()
	handler.ListOpenAI(openAIRec, openAIReq)

	tagsReq := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	tagsRec := httptest.NewRecorder()
	handler.ListOllamaTags(tagsRec, tagsReq)

	close(backend.block)
	if err := <-done; err != nil {
		t.Fatalf("inference failed: %v", err)
	}

	if openAIRec.Code != http.StatusOK {
		t.Fatalf("expected openai 200, got %d: %s", openAIRec.Code, openAIRec.Body.String())
	}
	var openAIResp model.OpenAIModelList
	if err := json.Unmarshal(openAIRec.Body.Bytes(), &openAIResp); err != nil {
		t.Fatalf("decode openai response: %v", err)
	}
	if len(openAIResp.Data) != 1 || openAIResp.Data[0].ID != "test-model" {
		t.Fatalf("expected active model in /v1/models, got %#v", openAIResp.Data)
	}

	if tagsRec.Code != http.StatusOK {
		t.Fatalf("expected tags 200, got %d: %s", tagsRec.Code, tagsRec.Body.String())
	}
	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(tagsRec.Body.Bytes(), &tagsResp); err != nil {
		t.Fatalf("decode tags response: %v", err)
	}
	if len(tagsResp.Models) != 1 || tagsResp.Models[0].Name != "test-model" {
		t.Fatalf("expected active model in /api/tags, got %#v", tagsResp.Models)
	}
}

func TestListRunningUsesRuntimeSnapshot(t *testing.T) {
	registry, err := storage.NewModelRegistry(t.TempDir())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "test",
		Name:     "test-model",
		Filename: "test.gguf",
		Status:   "ready",
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{block: make(chan struct{})}
	scheduler := service.NewRuntimeScheduler(backend, 2)
	manager := service.NewModelManagerWithScheduler(registry, scheduler, t.TempDir())
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	handler := NewModelsHandler(manager, backend)

	done := make(chan error, 1)
	go func() {
		_, err := inference.Complete(context.Background(), &model.ChatCompletionRequest{
			Model:    "test",
			Messages: []model.ChatMessage{{Role: "user", Content: "hi"}},
		})
		done <- err
	}()
	waitForSchedulerState(t, scheduler, engine.StateGenerating)

	req := httptest.NewRequest(http.MethodGet, "/api/ps", nil)
	rec := httptest.NewRecorder()
	handler.ListRunning(rec, req)

	close(backend.block)
	if err := <-done; err != nil {
		t.Fatalf("inference failed: %v", err)
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Models []struct {
			Name  string `json:"name"`
			State string `json:"state"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Models) != 1 || resp.Models[0].Name != "test-model" {
		t.Fatalf("expected running test-model, got %#v", resp.Models)
	}
	if resp.Models[0].State != engine.StateGenerating {
		t.Fatalf("expected generating state, got %q", resp.Models[0].State)
	}
}

func waitForSchedulerState(t *testing.T, scheduler *service.RuntimeScheduler, want string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if got := scheduler.State(); got == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("scheduler state did not become %s, got %s", want, scheduler.State())
}

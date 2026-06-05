package handler

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
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

func TestModelStatusIncludesDownloadTelemetry(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	handler := NewModelsHandler(manager, backend)

	var attempts atomic.Int32
	secondStarted := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attempts.Add(1)
		if attempt == 1 {
			http.Error(w, "temporary failure", http.StatusBadGateway)
			return
		}
		if attempt == 2 {
			close(secondStarted)
			<-release
		}
		_, _ = w.Write([]byte("model"))
	}))
	defer server.Close()

	id, err := manager.PullModel("telemetry", server.URL+"/model.gguf")
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	select {
	case <-secondStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for retry")
	}

	req := httptest.NewRequest(http.MethodGet, "/api/models/"+id+"/status", nil)
	routeCtx := chi.NewRouteContext()
	routeCtx.URLParams.Add("id", id)
	req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, routeCtx))
	rec := httptest.NewRecorder()

	handler.Status(rec, req)

	close(release)
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.ModelStatusResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.DownloadAttempt != 2 {
		t.Fatalf("download_attempt = %d", resp.DownloadAttempt)
	}
	if resp.MaxAttempts != 3 {
		t.Fatalf("download_max_attempts = %d", resp.MaxAttempts)
	}
	if resp.LastDownloadError != "HTTP 502" {
		t.Fatalf("last_download_error = %q", resp.LastDownloadError)
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

func TestShowIncludesModelMetadata(t *testing.T) {
	registry, err := storage.NewModelRegistry(t.TempDir())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:         "embed",
		Name:       "bge-embed",
		Filename:   "bge-small-embed-q4.gguf",
		Status:     "ready",
		Family:     "bge",
		Parameters: "1B",
		Template:   "{{ .Prompt }}",
		System:     "Answer briefly.",
		License:    []string{"Apache-2.0"},
		StopTokens: []string{"<|end|>"},
		SourceURL:  "ollama://library/bge-embed:latest",
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	manager := service.NewModelManagerWithScheduler(registry, scheduler, t.TempDir())
	handler := NewModelsHandler(manager, backend)

	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"bge-embed"}`))
	rec := httptest.NewRecorder()
	handler.Show(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Capabilities        model.ModelCapabilities        `json:"capabilities"`
		RecommendedSettings model.RecommendedModelSettings `json:"recommended_settings"`
		Modelfile           string                         `json:"modelfile"`
		Parameters          string                         `json:"parameters"`
		Template            string                         `json:"template"`
		System              string                         `json:"system"`
		License             []string                       `json:"license"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Capabilities.Chat {
		t.Fatal("embedding model should not default to chat capability")
	}
	if !resp.Capabilities.Embeddings {
		t.Fatal("expected embedding capability")
	}
	if resp.RecommendedSettings.ContextSize == 0 {
		t.Fatal("expected recommended context size")
	}
	if !strings.Contains(resp.Modelfile, "FROM library/bge-embed:latest") {
		t.Fatalf("modelfile = %q", resp.Modelfile)
	}
	if !strings.Contains(resp.Modelfile, "TEMPLATE") || !strings.Contains(resp.Modelfile, "SYSTEM") || !strings.Contains(resp.Modelfile, "LICENSE") {
		t.Fatalf("modelfile missing manifest sections: %q", resp.Modelfile)
	}
	if resp.Parameters != `PARAMETER stop "<|end|>"` {
		t.Fatalf("parameters = %q", resp.Parameters)
	}
	if resp.Template != "{{ .Prompt }}" {
		t.Fatalf("template = %q", resp.Template)
	}
	if resp.System != "Answer briefly." {
		t.Fatalf("system = %q", resp.System)
	}
	if len(resp.License) != 1 || resp.License[0] != "Apache-2.0" {
		t.Fatalf("license = %#v", resp.License)
	}
}

func TestDeleteOllamaResolvesModelName(t *testing.T) {
	registry, err := storage.NewModelRegistry(t.TempDir())
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:       "stable-id",
		Name:     "llama3.2:latest",
		Filename: "llama3.2.gguf",
		Status:   "ready",
		External: true,
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, t.TempDir())
	handler := NewModelsHandler(manager, backend)

	req := httptest.NewRequest(http.MethodDelete, "/api/delete", strings.NewReader(`{"model":"llama3.2:latest"}`))
	rec := httptest.NewRecorder()

	handler.DeleteOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := registry.Get("stable-id"); got != nil {
		t.Fatalf("expected model to be deleted, got %#v", got)
	}
}

func TestCopyOllamaCreatesModelAlias(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:                  "source-id",
		Name:                "llama3.2:latest",
		Filename:            "llama3.2.gguf",
		Status:              "ready",
		Family:              "llama",
		Parameters:          "3B",
		Template:            "{{ .Prompt }}",
		StopTokens:          []string{"<|eot_id|>"},
		Capabilities:        storage.ModelCapabilities{Chat: true},
		RecommendedSettings: storage.RecommendedModelSettings{ContextSize: 8192},
		SourceURL:           "ollama://llama3.2:latest",
		FilePath:            tmp + "/llama3.2.gguf",
		DownloadedAt:        time.Now().UTC(),
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	handler := NewModelsHandler(manager, backend)

	req := httptest.NewRequest(http.MethodPost, "/api/copy", strings.NewReader(`{"source":"llama3.2:latest","destination":"llama-copy:latest"}`))
	rec := httptest.NewRecorder()

	handler.CopyOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if body := strings.TrimSpace(rec.Body.String()); body != "" {
		t.Fatalf("expected empty response body, got %q", body)
	}
	copied, err := manager.ResolveModel("llama-copy:latest")
	if err != nil {
		t.Fatalf("resolve copied model: %v", err)
	}
	if copied.ID == "source-id" {
		t.Fatal("expected copied model to have a new id")
	}
	if copied.FilePath != tmp+"/llama3.2.gguf" {
		t.Fatalf("file path = %q", copied.FilePath)
	}
	if copied.Template != "{{ .Prompt }}" || len(copied.StopTokens) != 1 || copied.StopTokens[0] != "<|eot_id|>" {
		t.Fatalf("metadata not copied: %+v", copied)
	}
}

func TestPullOllamaNonStreamDownloadsDirectGGUF(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	handler := NewModelsHandler(manager, backend)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("model"))
	}))
	defer server.Close()

	body := `{"model":"tiny:latest","source_url":"` + server.URL + `/tiny.gguf","stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.PullOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.OllamaPullResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Status != "success" {
		t.Fatalf("status = %q", resp.Status)
	}
	if _, err := manager.ResolveModel("tiny:latest"); err != nil {
		t.Fatalf("expected pulled model to be registered: %v", err)
	}
}

func TestPullOllamaStreamsProgress(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	handler := NewModelsHandler(manager, backend)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("model"))
	}))
	defer server.Close()

	body := `{"model":"tiny-stream","source_url":"` + server.URL + `/tiny-stream.gguf"}`
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.PullOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	lines := strings.Split(strings.TrimSpace(rec.Body.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected streamed chunks, got %q", rec.Body.String())
	}
	var first model.OllamaPullResponse
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("decode first chunk: %v", err)
	}
	if first.Status != "pulling manifest" {
		t.Fatalf("first status = %q", first.Status)
	}
	var last model.OllamaPullResponse
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatalf("decode last chunk: %v", err)
	}
	if last.Status != "success" {
		t.Fatalf("last status = %q; body=%s", last.Status, rec.Body.String())
	}
}

func TestPullOllamaNonStreamDownloadsRegistryManifest(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	handler := NewModelsHandler(manager, backend)

	modelBlob := []byte("registry-model")
	modelDigest := testDigest(modelBlob)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/tiny/manifests/latest":
			writeJSON(w, http.StatusOK, map[string]any{
				"schemaVersion": 2,
				"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
				"layers": []map[string]any{
					{
						"mediaType": "application/vnd.ollama.image.model",
						"digest":    modelDigest,
						"size":      len(modelBlob),
					},
				},
			})
		case "/v2/library/tiny/blobs/" + modelDigest:
			_, _ = w.Write(modelBlob)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	modelRef := strings.TrimPrefix(server.URL, "http://") + "/library/tiny:latest"
	body := `{"model":"` + modelRef + `","insecure":true,"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.PullOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if _, err := manager.ResolveModel("library/tiny"); err != nil {
		t.Fatalf("expected registry model to be registered: %v", err)
	}
}

func testDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", sum)
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

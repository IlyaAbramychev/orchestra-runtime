package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/service"
	"github.com/operium/orchestra-runtime/internal/storage"
)

func TestOllamaChatGoldenResponse(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewBufferString(
		`{"model":"golden","stream":false,"messages":[{"role":"user","content":"hi"}]}`,
	))
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	requireStatus(t, rec, http.StatusOK)
	assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{
		"model":      "golden",
		"created_at": "<timestamp>",
		"message": map[string]any{
			"role":    "assistant",
			"content": "hello",
		},
		"done":                 true,
		"done_reason":          "stop",
		"total_duration":       float64(10),
		"prompt_eval_count":    float64(3),
		"prompt_eval_duration": float64(4),
		"eval_count":           float64(2),
		"eval_duration":        float64(6),
	}, "created_at")
}

func TestOllamaGenerateGoldenResponse(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewBufferString(
		`{"model":"golden","prompt":"hi","stream":false}`,
	))
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	requireStatus(t, rec, http.StatusOK)
	assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{
		"model":                "golden",
		"created_at":           "<timestamp>",
		"response":             "hello",
		"done":                 true,
		"done_reason":          "stop",
		"total_duration":       float64(10),
		"prompt_eval_count":    float64(3),
		"prompt_eval_duration": float64(4),
		"eval_count":           float64(2),
		"eval_duration":        float64(6),
	}, "created_at")
}

func TestOllamaModelEndpointGoldenResponses(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	entry := &storage.ModelEntry{
		ID:                  "test",
		Name:                "golden:latest",
		Filename:            "golden-q4.gguf",
		Size:                1024,
		Quantization:        "Q4_K_M",
		Family:              "llama",
		Parameters:          "7B",
		Modelfile:           "FROM golden:latest\n\nPARAMETER stop \"<|end|>\"",
		Template:            "{{ .Prompt }}",
		System:              "Be concise.",
		OllamaParameters:    "PARAMETER stop \"<|end|>\"",
		License:             []string{"MIT"},
		StopTokens:          []string{"<|end|>"},
		Capabilities:        storage.ModelCapabilities{Chat: true, Tools: true},
		RecommendedSettings: storage.RecommendedModelSettings{ContextSize: 8192},
		SourceURL:           "ollama://golden:latest",
		SHA256:              strings.Repeat("a", 64),
		Status:              "ready",
		FilePath:            tmp + "/golden-q4.gguf",
		DownloadedAt:        time.Date(2026, 6, 5, 8, 0, 0, 0, time.UTC),
	}
	if err := registry.Add(entry); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	handler := NewModelsHandler(manager, backend)

	t.Run("tags", func(t *testing.T) {
		rec := httptest.NewRecorder()
		handler.ListOllamaTags(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
		requireStatus(t, rec, http.StatusOK)
		assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{
			"models": []any{
				map[string]any{
					"name":        "golden:latest",
					"model":       "golden:latest",
					"modified_at": "2026-06-05T08:00:00Z",
					"size":        float64(1024),
					"digest":      strings.Repeat("a", 64),
					"details": map[string]any{
						"format":             "gguf",
						"family":             "llama",
						"parameter_size":     "7B",
						"quantization_level": "Q4_K_M",
					},
					"capabilities": map[string]any{
						"chat":       true,
						"embeddings": false,
						"rerank":     false,
						"tools":      true,
						"thinking":   false,
					},
				},
			},
		})
	})

	t.Run("show", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"golden:latest"}`))
		handler.Show(rec, req)
		requireStatus(t, rec, http.StatusOK)
		assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{
			"modelfile":  "FROM golden:latest\n\nPARAMETER stop \"<|end|>\"",
			"parameters": "PARAMETER stop \"<|end|>\"",
			"template":   "{{ .Prompt }}",
			"system":     "Be concise.",
			"license":    []any{"MIT"},
			"details": map[string]any{
				"format":             "gguf",
				"family":             "llama",
				"parameter_size":     "7B",
				"quantization_level": "Q4_K_M",
			},
			"capabilities": map[string]any{
				"chat":       true,
				"embeddings": false,
				"rerank":     false,
				"tools":      true,
				"thinking":   false,
			},
			"stop_tokens": []any{"<|end|>"},
			"recommended_settings": map[string]any{
				"context_size": float64(8192),
			},
			"model_info": map[string]any{
				"general.name":       "golden:latest",
				"general.size_bytes": float64(1024),
				"general.file_path":  tmp + "/golden-q4.gguf",
				"general.sha256":     strings.Repeat("a", 64),
				"general.source_url": "ollama://golden:latest",
			},
		})
	})

	t.Run("ps", func(t *testing.T) {
		rec := httptest.NewRecorder()
		handler.ListRunning(rec, httptest.NewRequest(http.MethodGet, "/api/ps", nil))
		requireStatus(t, rec, http.StatusOK)
		assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{
			"models": []any{
				map[string]any{
					"name":      "golden:latest",
					"model":     "golden:latest",
					"size":      float64(1024),
					"size_vram": float64(1024),
					"state":     "ready",
				},
			},
		})
	})
}

func TestOllamaPullDeleteAndVersionGoldenResponses(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	backend := &fakeChatBackend{}
	manager := service.NewModelManager(registry, backend, tmp)
	modelsHandler := NewModelsHandler(manager, backend)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("model"))
	}))
	defer server.Close()

	t.Run("pull", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(
			`{"model":"pulled:latest","source_url":"`+server.URL+`/pulled.gguf","stream":false}`,
		))
		modelsHandler.PullOllama(rec, req)
		requireStatus(t, rec, http.StatusOK)
		assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{"status": "success"})
	})

	t.Run("delete", func(t *testing.T) {
		copyRec := httptest.NewRecorder()
		copyReq := httptest.NewRequest(http.MethodPost, "/api/copy", strings.NewReader(
			`{"source":"pulled:latest","destination":"pulled-copy:latest"}`,
		))
		modelsHandler.CopyOllama(copyRec, copyReq)
		requireStatus(t, copyRec, http.StatusOK)
		if body := strings.TrimSpace(copyRec.Body.String()); body != "" {
			t.Fatalf("expected empty copy body, got %q", body)
		}

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodDelete, "/api/delete", strings.NewReader(`{"model":"pulled:latest"}`))
		modelsHandler.DeleteOllama(rec, req)
		requireStatus(t, rec, http.StatusOK)
		if body := strings.TrimSpace(rec.Body.String()); body != "" {
			t.Fatalf("expected empty delete body, got %q", body)
		}
	})

	t.Run("version", func(t *testing.T) {
		rec := httptest.NewRecorder()
		systemHandler := NewSystemHandler(service.NewSystemInfo(backend))
		systemHandler.Version(rec, httptest.NewRequest(http.MethodGet, "/api/version", nil))
		requireStatus(t, rec, http.StatusOK)
		assertGoldenJSON(t, rec.Body.Bytes(), map[string]any{"version": service.Version})
	})
}

func requireStatus(t *testing.T, rec *httptest.ResponseRecorder, want int) {
	t.Helper()
	if rec.Code != want {
		t.Fatalf("expected status %d, got %d: %s", want, rec.Code, rec.Body.String())
	}
}

func assertGoldenJSON(t *testing.T, body []byte, expected map[string]any, normalizedKeys ...string) {
	t.Helper()
	var actual map[string]any
	if err := json.Unmarshal(body, &actual); err != nil {
		t.Fatalf("decode response: %v\nbody=%s", err, body)
	}
	for _, key := range normalizedKeys {
		if _, ok := actual[key]; ok {
			actual[key] = "<timestamp>"
		}
	}
	if !reflect.DeepEqual(actual, expected) {
		actualJSON, _ := json.MarshalIndent(actual, "", "  ")
		expectedJSON, _ := json.MarshalIndent(expected, "", "  ")
		t.Fatalf("response mismatch\nactual: %s\nexpected: %s", actualJSON, expectedJSON)
	}
}

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/config"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/handler"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
	"github.com/operium/orchestra-runtime/internal/storage"
)

func TestOllamaCompatibilitySmokeSuite(t *testing.T) {
	server := httptest.NewServer(newOllamaSmokeRouter(t))
	t.Cleanup(server.Close)

	t.Run("version capabilities and tags", func(t *testing.T) {
		var version map[string]string
		smokeJSON(t, server.URL, http.MethodGet, "/api/version", "", http.StatusOK, &version)
		if version["version"] == "" {
			t.Fatalf("missing version: %+v", version)
		}

		var capabilities model.RuntimeCapabilitiesResponse
		smokeJSON(t, server.URL, http.MethodGet, "/api/capabilities", "", http.StatusOK, &capabilities)
		if !capabilities.Ollama.Compatible {
			t.Fatalf("expected ollama compatible capabilities: %+v", capabilities)
		}

		var tags struct {
			Models []struct {
				Name string `json:"name"`
			} `json:"models"`
		}
		smokeJSON(t, server.URL, http.MethodGet, "/api/tags", "", http.StatusOK, &tags)
		if len(tags.Models) < 2 {
			t.Fatalf("expected seeded smoke models, got %+v", tags)
		}
	})

	t.Run("chat non-stream", func(t *testing.T) {
		var resp model.OllamaChatResponse
		smokeJSON(t, server.URL, http.MethodPost, "/api/chat",
			`{"model":"chat-smoke:latest","stream":false,"messages":[{"role":"user","content":"hello"}]}`,
			http.StatusOK, &resp)
		if !resp.Done || resp.Message.Content != "hello from smoke" {
			t.Fatalf("unexpected chat response: %+v", resp)
		}
	})

	t.Run("chat tool calls", func(t *testing.T) {
		var resp model.OllamaChatResponse
		smokeJSON(t, server.URL, http.MethodPost, "/api/chat",
			`{"model":"chat-smoke:latest","stream":false,"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"call tool"}]}`,
			http.StatusOK, &resp)
		if resp.DoneReason != "tool_calls" || len(resp.Message.ToolCalls) != 1 {
			t.Fatalf("unexpected tool response: %+v", resp)
		}
		if resp.Message.ToolCalls[0].Function.Name != "get_weather" {
			t.Fatalf("unexpected tool call: %+v", resp.Message.ToolCalls[0])
		}
	})

	t.Run("chat tool calls stream", func(t *testing.T) {
		chunks := smokeChatStream(t, server.URL, `{"model":"chat-smoke:latest","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"call tool stream"}]}`)
		if len(chunks) != 2 || len(chunks[0].Message.ToolCalls) != 1 || chunks[1].DoneReason != "tool_calls" {
			t.Fatalf("unexpected streaming tool chunks: %+v", chunks)
		}
	})

	t.Run("chat thinking", func(t *testing.T) {
		var resp model.OllamaChatResponse
		smokeJSON(t, server.URL, http.MethodPost, "/api/chat",
			`{"model":"chat-smoke:latest","stream":false,"think":true,"messages":[{"role":"user","content":"think"}]}`,
			http.StatusOK, &resp)
		if resp.Message.Thinking != "smoke reasoning" || resp.Message.Content != "smoke answer" {
			t.Fatalf("unexpected thinking response: %+v", resp.Message)
		}
	})

	t.Run("generate schema", func(t *testing.T) {
		var resp model.GenerateResponse
		smokeJSON(t, server.URL, http.MethodPost, "/api/generate",
			`{"model":"chat-smoke:latest","stream":false,"prompt":"schema","format":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}`,
			http.StatusOK, &resp)
		if resp.Response != `{"answer":"ok"}` {
			t.Fatalf("unexpected schema response: %+v", resp)
		}
	})

	t.Run("generate schema stream", func(t *testing.T) {
		chunks := smokeGenerateStream(t, server.URL, `{"model":"chat-smoke:latest","prompt":"schema","format":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}`)
		if len(chunks) != 2 || chunks[0].Response != `{"answer":"ok"}` || !chunks[1].Done {
			t.Fatalf("unexpected schema stream chunks: %+v", chunks)
		}
	})

	t.Run("generate stream", func(t *testing.T) {
		chunks := smokeGenerateStream(t, server.URL, `{"model":"chat-smoke:latest","prompt":"stream"}`)
		if len(chunks) != 2 || chunks[0].Response != "smoke" || !chunks[1].Done {
			t.Fatalf("unexpected stream chunks: %+v", chunks)
		}
	})

	t.Run("unsupported images reject explicitly", func(t *testing.T) {
		var resp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		smokeJSON(t, server.URL, http.MethodPost, "/api/chat",
			`{"model":"chat-smoke:latest","stream":false,"messages":[{"role":"user","content":"image","images":["aGVsbG8="]}]}`,
			http.StatusBadRequest, &resp)
		if !strings.Contains(resp.Error.Message, "multimodal images") {
			t.Fatalf("unexpected error: %+v", resp)
		}
	})

	t.Run("embeddings", func(t *testing.T) {
		var resp model.EmbedResponse
		smokeJSON(t, server.URL, http.MethodPost, "/api/embed",
			`{"model":"embed-smoke:latest","input":["alpha","beta"]}`,
			http.StatusOK, &resp)
		if len(resp.Embeddings) != 2 || len(resp.Embeddings[0]) != 3 {
			t.Fatalf("unexpected embeddings: %+v", resp)
		}
	})
}

func newOllamaSmokeRouter(t *testing.T) http.Handler {
	t.Helper()
	tmp := t.TempDir()
	backend := &ollamaSmokeBackend{}
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	addSmokeModel(t, registry, tmp, "chat-smoke", "chat-smoke:latest", storage.ModelCapabilities{Chat: true, Tools: true, Thinking: true})
	addSmokeModel(t, registry, tmp, "embed-smoke", "embed-smoke:latest", storage.ModelCapabilities{Embeddings: true})

	scheduler := service.NewRuntimeScheduler(backend, 4)
	modelMgr := service.NewModelManagerWithScheduler(registry, scheduler, tmp)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(modelMgr)
	embedding := service.NewEmbeddingServiceWithScheduler(scheduler)
	embedding.SetModelLoader(modelMgr)
	sysInfo := service.NewSystemInfo(backend)
	sysInfo.SetScheduler(scheduler)

	chatH := handler.NewChatHandler(inference)
	genH := handler.NewGenerateHandler(inference)
	embH := handler.NewEmbedHandler(embedding, inference)
	modelsH := handler.NewModelsHandler(modelMgr, backend)
	systemH := handler.NewSystemHandler(sysInfo)
	systemH.SetInference(inference)

	s := &Server{cfg: &config.Config{CORSOrigins: []string{"*"}}}
	return s.buildRouter(chatH, genH, embH, modelsH, systemH, handler.NewAdminHandler(nil, func() {}))
}

func addSmokeModel(t *testing.T, registry *storage.ModelRegistry, dir, id, name string, capabilities storage.ModelCapabilities) {
	t.Helper()
	path := dir + "/" + id + ".gguf"
	if err := os.WriteFile(path, []byte("smoke"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	if err := registry.Add(&storage.ModelEntry{
		ID:           id,
		Name:         name,
		Filename:     id + ".gguf",
		Size:         5,
		Family:       "smoke",
		Parameters:   "1B",
		Capabilities: capabilities,
		Status:       model.StatusReady,
		FilePath:     path,
		DownloadedAt: time.Date(2026, 6, 5, 8, 0, 0, 0, time.UTC),
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}
}

func smokeChatStream(t *testing.T, baseURL, body string) []model.OllamaChatResponse {
	t.Helper()
	resp := smokeStreamResponse(t, baseURL, "/api/chat", body)
	defer resp.Body.Close()

	var chunks []model.OllamaChatResponse
	decoder := json.NewDecoder(resp.Body)
	for {
		var chunk model.OllamaChatResponse
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("decode chat stream: %v", err)
		}
		chunks = append(chunks, chunk)
	}
	return chunks
}

func smokeGenerateStream(t *testing.T, baseURL, body string) []model.GenerateResponse {
	t.Helper()
	resp := smokeStreamResponse(t, baseURL, "/api/generate", body)
	defer resp.Body.Close()

	var chunks []model.GenerateResponse
	decoder := json.NewDecoder(resp.Body)
	for {
		var chunk model.GenerateResponse
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("decode generate stream: %v", err)
		}
		chunks = append(chunks, chunk)
	}
	return chunks
}

func smokeStreamResponse(t *testing.T, baseURL, path, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, baseURL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(resp.Body)
		t.Fatalf("%s status = %d: %s", path, resp.StatusCode, buf.String())
	}
	if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		resp.Body.Close()
		t.Fatalf("content type = %q", got)
	}
	return resp
}

func smokeJSON(t *testing.T, baseURL, method, path, body string, wantStatus int, out any) {
	t.Helper()
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, baseURL+path, reader)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request %s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != wantStatus {
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(resp.Body)
		t.Fatalf("%s %s status = %d, want %d: %s", method, path, resp.StatusCode, wantStatus, buf.String())
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		t.Fatalf("decode %s %s: %v", method, path, err)
	}
}

type ollamaSmokeBackend struct {
	loaded string
}

func (b *ollamaSmokeBackend) InitBackend() {}
func (b *ollamaSmokeBackend) FreeBackend() {}
func (b *ollamaSmokeBackend) Close() error { return nil }
func (b *ollamaSmokeBackend) LoadModel(modelID, _ string, _ engine.LoadOptions) error {
	b.loaded = modelID
	return nil
}
func (b *ollamaSmokeBackend) UnloadModel()          { b.loaded = "" }
func (b *ollamaSmokeBackend) IsLoaded() bool        { return b.loaded != "" }
func (b *ollamaSmokeBackend) LoadedModelID() string { return b.loaded }
func (b *ollamaSmokeBackend) State() string         { return engine.StateReady }
func (b *ollamaSmokeBackend) ModelDesc() string     { return "" }
func (b *ollamaSmokeBackend) Complete(_ context.Context, messages []engine.ChatMessage, _ engine.CompletionParams) (*engine.CompletionResult, error) {
	for _, message := range messages {
		if len(message.Images) > 0 {
			return nil, fmt.Errorf("multimodal images require a loaded mmproj")
		}
	}
	text := "hello from smoke"
	if len(messages) > 0 {
		content := messages[len(messages)-1].Content
		switch {
		case strings.Contains(content, "call tool"):
			text = `{"tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]}`
		case strings.Contains(content, "think"):
			text = `<think>smoke reasoning</think>smoke answer`
		case strings.Contains(content, "schema"):
			text = `{"answer":"ok"}`
		}
	}
	return &engine.CompletionResult{
		Text:             text,
		PromptTokens:     3,
		CompletionTokens: 2,
		FinishReason:     "stop",
		Timings:          engine.Timings{TotalNs: 10, PromptEvalNs: 4, EvalNs: 6},
	}, nil
}
func (b *ollamaSmokeBackend) CompleteStream(_ context.Context, messages []engine.ChatMessage, _ engine.CompletionParams) (<-chan engine.CompletionChunk, error) {
	for _, message := range messages {
		if len(message.Images) > 0 {
			return nil, fmt.Errorf("multimodal images require a loaded mmproj")
		}
	}
	text := "smoke"
	if len(messages) > 0 {
		content := messages[len(messages)-1].Content
		switch {
		case strings.Contains(content, "call tool"):
			text = `{"tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]}`
		case strings.Contains(content, "think"):
			text = `<think>smoke reasoning</think>smoke answer`
		case strings.Contains(content, "schema"):
			text = `{"answer":"ok"}`
		}
	}
	ch := make(chan engine.CompletionChunk, 2)
	ch <- engine.CompletionChunk{Text: text}
	ch <- engine.CompletionChunk{
		Done:             true,
		FinishReason:     "stop",
		PromptTokens:     3,
		CompletionTokens: 1,
		Timings:          engine.Timings{TotalNs: 10, PromptEvalNs: 4, EvalNs: 6},
	}
	close(ch)
	return ch, nil
}
func (b *ollamaSmokeBackend) Embed(context.Context, string, bool) (*engine.EmbeddingResult, error) {
	return &engine.EmbeddingResult{Vector: []float32{0.1, 0.2, 0.3}, PromptTokens: 1}, nil
}
func (b *ollamaSmokeBackend) SetIdleTimeout(time.Duration) {}
func (b *ollamaSmokeBackend) IdleTimeout() time.Duration   { return 0 }
func (b *ollamaSmokeBackend) ApplyKeepAlive(*int64)        {}
func (b *ollamaSmokeBackend) MarkUsed()                    {}

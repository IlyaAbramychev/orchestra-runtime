package handler

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

type fakeChatBackend struct {
	notLoaded    bool
	completeText string
	completeErr  error
	streamErr    error
	streamChunks []engine.CompletionChunk
	embedErr     error
	block        chan struct{}
}

func (f *fakeChatBackend) InitBackend()                                       {}
func (f *fakeChatBackend) FreeBackend()                                       {}
func (f *fakeChatBackend) Close() error                                       { return nil }
func (f *fakeChatBackend) LoadModel(string, string, engine.LoadOptions) error { return nil }
func (f *fakeChatBackend) UnloadModel()                                       {}
func (f *fakeChatBackend) IsLoaded() bool                                     { return !f.notLoaded }
func (f *fakeChatBackend) LoadedModelID() string                              { return "test" }
func (f *fakeChatBackend) State() string                                      { return engine.StateReady }
func (f *fakeChatBackend) ModelDesc() string                                  { return "" }
func (f *fakeChatBackend) Complete(context.Context, []engine.ChatMessage, engine.CompletionParams) (*engine.CompletionResult, error) {
	if f.completeErr != nil {
		return nil, f.completeErr
	}
	if f.block != nil {
		<-f.block
	}
	text := f.completeText
	if text == "" {
		text = "hello"
	}
	return &engine.CompletionResult{
		Text:             text,
		PromptTokens:     3,
		CompletionTokens: 2,
		FinishReason:     "stop",
		Timings: engine.Timings{
			TotalNs:      10,
			PromptEvalNs: 4,
			EvalNs:       6,
		},
	}, nil
}
func (f *fakeChatBackend) CompleteStream(context.Context, []engine.ChatMessage, engine.CompletionParams) (<-chan engine.CompletionChunk, error) {
	if f.streamErr != nil {
		return nil, f.streamErr
	}
	if f.streamChunks != nil {
		ch := make(chan engine.CompletionChunk, len(f.streamChunks))
		for _, chunk := range f.streamChunks {
			ch <- chunk
		}
		close(ch)
		return ch, nil
	}
	ch := make(chan engine.CompletionChunk, 2)
	ch <- engine.CompletionChunk{Text: "hel"}
	ch <- engine.CompletionChunk{
		Done:             true,
		FinishReason:     "stop",
		PromptTokens:     3,
		CompletionTokens: 1,
		Timings: engine.Timings{
			TotalNs:      10,
			PromptEvalNs: 4,
			EvalNs:       6,
		},
	}
	close(ch)
	return ch, nil
}
func (f *fakeChatBackend) Embed(context.Context, string, bool) (*engine.EmbeddingResult, error) {
	if f.embedErr != nil {
		return nil, f.embedErr
	}
	return &engine.EmbeddingResult{}, nil
}
func (f *fakeChatBackend) SetIdleTimeout(time.Duration) {}
func (f *fakeChatBackend) IdleTimeout() time.Duration   { return 0 }
func (f *fakeChatBackend) ApplyKeepAlive(*int64)        {}
func (f *fakeChatBackend) MarkUsed()                    {}

func TestOllamaChatNonStreamShape(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	body := bytes.NewBufferString(`{"model":"test","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.OllamaChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !resp.Done || resp.Message.Role != "assistant" || resp.Message.Content != "hello" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.PromptEvalCount != 3 || resp.EvalCount != 2 || resp.DoneReason != "stop" {
		t.Fatalf("missing final metrics: %+v", resp)
	}
}

func TestOllamaChatFormatJSONValidatesResponse(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{completeText: `{"answer":"ok"}`}, 1))
	body := bytes.NewBufferString(`{"model":"test","stream":false,"format":"json","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.OllamaChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Message.Content != `{"answer":"ok"}` {
		t.Fatalf("content = %q", resp.Message.Content)
	}
}

func TestOllamaChatFormatJSONRejectsInvalidModelOutput(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{completeText: `not json`}, 1))
	body := bytes.NewBufferString(`{"model":"test","stream":false,"format":"json","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status 502, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestOllamaChatFormatRejectsStreaming(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	body := bytes.NewBufferString(`{"model":"test","format":"json","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestOllamaChatToolCallsResponseShape(t *testing.T) {
	backend := &fakeChatBackend{completeText: `{"tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}]}`}
	h := NewChatHandler(service.NewInferenceService(backend, 1))
	body := bytes.NewBufferString(`{
		"model":"test",
		"stream":false,
		"tools":[
			{
				"type":"function",
				"function":{
					"name":"get_weather",
					"description":"Get weather",
					"parameters":{
						"type":"object",
						"properties":{"city":{"type":"string"}},
						"required":["city"]
					}
				}
			}
		],
		"messages":[{"role":"user","content":"weather in Paris"}]
	}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.OllamaChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.DoneReason != "tool_calls" {
		t.Fatalf("done reason = %q", resp.DoneReason)
	}
	if resp.Message.Content != "" {
		t.Fatalf("content = %q", resp.Message.Content)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %+v", resp.Message.ToolCalls)
	}
	call := resp.Message.ToolCalls[0]
	if call.Type != "function" || call.Function.Name != "get_weather" {
		t.Fatalf("unexpected tool call: %+v", call)
	}
	if call.Function.Arguments["city"] != "Paris" {
		t.Fatalf("unexpected arguments: %+v", call.Function.Arguments)
	}
}

func TestOllamaChatToolsRejectStreaming(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	body := bytes.NewBufferString(`{"model":"test","tools":[{"type":"function","function":{"name":"get_weather"}}],"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestOllamaChatToolCallsRejectUnknownTool(t *testing.T) {
	backend := &fakeChatBackend{completeText: `{"tool_calls":[{"function":{"name":"delete_everything","arguments":{}}}]}`}
	h := NewChatHandler(service.NewInferenceService(backend, 1))
	body := bytes.NewBufferString(`{"model":"test","stream":false,"tools":[{"type":"function","function":{"name":"get_weather"}}],"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status 502, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestOllamaChatDefaultsToNDJSONStream(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	body := bytes.NewBufferString(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "application/x-ndjson" {
		t.Fatalf("unexpected content type %q", got)
	}

	scanner := bufio.NewScanner(bytes.NewReader(rec.Body.Bytes()))
	var chunks []model.OllamaChatResponse
	for scanner.Scan() {
		var chunk model.OllamaChatResponse
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan stream: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %s", len(chunks), rec.Body.String())
	}
	if chunks[0].Done || chunks[0].Message.Content != "hel" {
		t.Fatalf("unexpected first chunk: %+v", chunks[0])
	}
	if !chunks[1].Done || chunks[1].DoneReason != "stop" || chunks[1].PromptEvalCount != 3 {
		t.Fatalf("unexpected final chunk: %+v", chunks[1])
	}
}

func TestRuntimeErrorStatusMapping(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want int
	}{
		{name: "queue", err: fmt.Errorf("runtime queue full"), want: http.StatusTooManyRequests},
		{name: "overflow", err: fmt.Errorf("prompt too long: 999 tokens >= context window 10"), want: http.StatusBadRequest},
		{name: "typed overflow", err: engine.NewContextLengthExceededError(32801, 12032, true), want: http.StatusBadRequest},
		{name: "timeout", err: context.DeadlineExceeded, want: http.StatusGatewayTimeout},
		{name: "cancelled", err: context.Canceled, want: 499},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{completeErr: tc.err}, 1))
			body := bytes.NewBufferString(`{"model":"test","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
			req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
			rec := httptest.NewRecorder()

			h.ChatOllama(rec, req)

			if rec.Code != tc.want {
				t.Fatalf("expected status %d, got %d: %s", tc.want, rec.Code, rec.Body.String())
			}
		})
	}
}

func TestOpenAIChatStreamContextLengthErrorReturnsBadRequest(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{
		streamErr: engine.NewContextLengthExceededError(32801, 12032, true),
	}, 1))
	body := bytes.NewBufferString(`{"model":"test","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	rec := httptest.NewRecorder()

	h.ChatCompletion(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Error struct {
			Code            string `json:"code"`
			Message         string `json:"message"`
			PromptTokens    int    `json:"promptTokens"`
			ContextSize     int    `json:"contextSize"`
			MaxOutputTokens int    `json:"maxOutputTokens"`
			OverflowTokens  int    `json:"overflowTokens"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Error.Code != engine.ContextLengthExceededCode {
		t.Fatalf("unexpected error payload: %+v", resp)
	}
	if resp.Error.PromptTokens != 32801 || resp.Error.ContextSize != 12032 {
		t.Fatalf("unexpected error message: %+v", resp)
	}
}

func TestNoModelLoadedMapsToNotFound(t *testing.T) {
	h := NewChatHandler(service.NewInferenceService(&fakeChatBackend{notLoaded: true}, 1))
	body := bytes.NewBufferString(`{"model":"test","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected status 404, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestOllamaChatStreamErrorReturnsDoneErrorChunk(t *testing.T) {
	backend := &fakeChatBackend{
		streamChunks: []engine.CompletionChunk{
			{Text: "hel"},
			{Err: fmt.Errorf("decode failed")},
		},
	}
	h := NewChatHandler(service.NewInferenceService(backend, 1))
	body := bytes.NewBufferString(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rec := httptest.NewRecorder()

	h.ChatOllama(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	scanner := bufio.NewScanner(bytes.NewReader(rec.Body.Bytes()))
	var chunks []model.OllamaChatResponse
	for scanner.Scan() {
		var chunk model.OllamaChatResponse
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %s", len(chunks), rec.Body.String())
	}
	if !chunks[1].Done || chunks[1].DoneReason != "error" || chunks[1].Error != "decode failed" {
		t.Fatalf("unexpected error chunk: %+v", chunks[1])
	}
}

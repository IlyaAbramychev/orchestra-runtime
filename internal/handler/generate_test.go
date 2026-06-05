package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

func TestGenerateRejectsUnsupportedFields(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	cases := []string{
		`{"model":"test","prompt":"hi","template":"{{ .Prompt }}"}`,
		`{"model":"test","prompt":"hi","context":[1,2,3]}`,
		`{"model":"test","prompt":"hi","images":["aGVsbG8="]}`,
		`{"model":"test","prompt":"hi","suffix":"end"}`,
	}
	for _, body := range cases {
		req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()

		h.Generate(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d: %s", body, rec.Code, rec.Body.String())
		}
	}
}

func TestGenerateFormatJSONValidatesResponse(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{completeText: `{"answer":"ok"}`}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","format":"json","stream":false}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.GenerateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Response != `{"answer":"ok"}` {
		t.Fatalf("response = %q", resp.Response)
	}
}

func TestGenerateFormatJSONRejectsInvalidModelOutput(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{completeText: `not json`}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","format":"json","stream":false}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status 502, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGenerateFormatSchemaRejectsNonConformingOutput(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{completeText: `{"answer":123}`}, 1))
	body := bytes.NewBufferString(`{
		"model":"test",
		"prompt":"hi",
		"stream":false,
		"format":{
			"type":"object",
			"properties":{"answer":{"type":"string"}},
			"required":["answer"]
		}
	}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status 502, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGenerateFormatStreamsBuffered(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{
		streamChunks: []engine.CompletionChunk{
			{Text: `{"answer":"ok"}`},
			{Done: true, FinishReason: "stop", PromptTokens: 3, CompletionTokens: 2},
		},
	}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","format":"json"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	chunks := decodeGenerateStream(t, rec.Body.Bytes())
	if len(chunks) != 2 || chunks[0].Response != `{"answer":"ok"}` || !chunks[1].Done {
		t.Fatalf("unexpected chunks: %+v", chunks)
	}
}

func TestGenerateThinkSeparatesThinking(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{completeText: `<think>reasoning trace</think>final answer`}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","think":"low","stream":false}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.GenerateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Thinking != "reasoning trace" || resp.Response != "final answer" {
		t.Fatalf("unexpected thinking response: %+v", resp)
	}
}

func TestGenerateThinkStreamsBuffered(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{
		streamChunks: []engine.CompletionChunk{
			{Text: `<think>reasoning trace</think>final answer`},
			{Done: true, FinishReason: "stop", PromptTokens: 3, CompletionTokens: 2},
		},
	}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","think":true}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	chunks := decodeGenerateStream(t, rec.Body.Bytes())
	if len(chunks) != 3 || chunks[0].Thinking != "reasoning trace" || chunks[1].Response != "final answer" || !chunks[2].Done {
		t.Fatalf("unexpected chunks: %+v", chunks)
	}
}

func TestGenerateStreamErrorReturnsDoneErrorChunk(t *testing.T) {
	backend := &fakeChatBackend{
		streamChunks: []engine.CompletionChunk{
			{Text: "hel"},
			{Err: fmt.Errorf("decode failed")},
		},
	}
	h := NewGenerateHandler(service.NewInferenceService(backend, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	rec := httptest.NewRecorder()

	h.Generate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	scanner := bufio.NewScanner(bytes.NewReader(rec.Body.Bytes()))
	var chunks []model.GenerateResponse
	for scanner.Scan() {
		var chunk model.GenerateResponse
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

func decodeGenerateStream(t *testing.T, body []byte) []model.GenerateResponse {
	t.Helper()
	scanner := bufio.NewScanner(bytes.NewReader(body))
	var chunks []model.GenerateResponse
	for scanner.Scan() {
		var chunk model.GenerateResponse
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan stream: %v", err)
	}
	return chunks
}

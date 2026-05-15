package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

func TestOpenAICompletionDefaultsToNonStreamJSON(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","max_tokens":4}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", body)
	rec := httptest.NewRecorder()

	h.Completion(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("unexpected content type %q", got)
	}
	var resp model.CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Object != "text_completion" || len(resp.Choices) != 1 || resp.Choices[0].Text != "hello" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 3 || resp.Usage.CompletionTokens != 2 {
		t.Fatalf("missing usage: %+v", resp.Usage)
	}
}

func TestOpenAICompletionStreamsSSEWhenRequested(t *testing.T) {
	h := NewGenerateHandler(service.NewInferenceService(&fakeChatBackend{}, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","stream":true}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", body)
	rec := httptest.NewRecorder()

	h.Completion(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("unexpected content type %q", got)
	}

	scanner := bufio.NewScanner(bytes.NewReader(rec.Body.Bytes()))
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan stream: %v", err)
	}
	if len(dataLines) != 3 {
		t.Fatalf("expected 3 data lines, got %d: %s", len(dataLines), rec.Body.String())
	}
	if dataLines[2] != "[DONE]" {
		t.Fatalf("expected [DONE], got %q", dataLines[2])
	}
	var first model.CompletionChunk
	if err := json.Unmarshal([]byte(dataLines[0]), &first); err != nil {
		t.Fatalf("decode first chunk: %v", err)
	}
	if first.Choices[0].Text != "hel" {
		t.Fatalf("unexpected first chunk: %+v", first)
	}
}

func TestOpenAICompletionStreamErrorDoesNotSendDone(t *testing.T) {
	backend := &fakeChatBackend{
		streamChunks: []engine.CompletionChunk{
			{Text: "hel"},
			{Err: fmt.Errorf("decode failed")},
		},
	}
	h := NewGenerateHandler(service.NewInferenceService(backend, 1))
	body := bytes.NewBufferString(`{"model":"test","prompt":"hi","stream":true}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", body)
	rec := httptest.NewRecorder()

	h.Completion(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Fatalf("stream error must not be followed by [DONE]: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"error"`) || !strings.Contains(rec.Body.String(), "decode failed") {
		t.Fatalf("missing error payload: %s", rec.Body.String())
	}
}

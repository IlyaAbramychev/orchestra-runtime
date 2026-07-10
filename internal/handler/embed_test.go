package handler

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/service"
)

func TestOllamaEmbedRejectsUnsupportedFields(t *testing.T) {
	scheduler := service.NewRuntimeScheduler(&fakeChatBackend{}, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	cases := []string{
		`{"model":"test","input":"hi","truncate":true}`,
		`{"model":"test","input":"hi","options":{"temperature":0}}`,
	}
	for _, body := range cases {
		req := httptest.NewRequest(http.MethodPost, "/api/embed", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()

		h.Embed(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d: %s", body, rec.Code, rec.Body.String())
		}
	}
}

func TestCapabilityMismatchMapsToBadRequest(t *testing.T) {
	status := runtimeHTTPStatus(errors.New("model chat-model does not support embeddings"))
	if status != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", status)
	}
}

func TestCustomTemplateErrorMapsToBadRequest(t *testing.T) {
	status := runtimeHTTPStatus(errors.New("custom chat template failed: failed to apply chat template"))
	if status != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", status)
	}
}

func TestOpenAIEmbeddingsRejectUnsupportedFields(t *testing.T) {
	scheduler := service.NewRuntimeScheduler(&fakeChatBackend{}, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	cases := []string{
		`{"model":"test","input":"hi","encoding_format":"hex"}`,
	}
	for _, body := range cases {
		req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()

		h.EmbedOpenAI(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d: %s", body, rec.Code, rec.Body.String())
		}
		var resp struct {
			Error struct {
				Message string `json:"message"`
				Code    string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode error response: %v", err)
		}
		if resp.Error.Code != "unsupported_encoding_format" {
			t.Fatalf("unexpected error payload: %+v", resp)
		}
	}
}

func TestOpenAIEmbeddingsRejectInvalidDimensions(t *testing.T) {
	backend := &fakeChatBackend{
		embedResult: &engine.EmbeddingResult{Vector: []float32{0.5, -1.25}, PromptTokens: 2},
	}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	cases := []string{
		`{"model":"test","input":"hi","dimensions":0}`,
		`{"model":"test","input":"hi","dimensions":3}`,
	}
	for _, body := range cases {
		req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()

		h.EmbedOpenAI(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d: %s", body, rec.Code, rec.Body.String())
		}
		var resp struct {
			Error struct {
				Message string `json:"message"`
				Code    string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode error response: %v", err)
		}
		if resp.Error.Code != "invalid_dimensions" {
			t.Fatalf("unexpected error payload: %+v", resp)
		}
	}
}

func TestOpenAIEmbeddingsRejectInvalidInput(t *testing.T) {
	scheduler := service.NewRuntimeScheduler(&fakeChatBackend{}, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"test","input":[]}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if resp.Error.Code != "invalid_input" {
		t.Fatalf("unexpected error payload: %+v", resp)
	}
}

func TestOpenAIEmbeddingsDefaultFloatResponse(t *testing.T) {
	backend := &fakeChatBackend{
		embedResult: &engine.EmbeddingResult{Vector: []float32{0.5, -1.25}, PromptTokens: 2},
	}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"test","input":"hi"}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Data) != 1 || len(resp.Data[0].Embedding) != 2 {
		t.Fatalf("unexpected response: %s", rec.Body.String())
	}
	if resp.Data[0].Embedding[0] != 0.5 || resp.Data[0].Embedding[1] != -1.25 {
		t.Fatalf("unexpected embedding values: %+v", resp.Data[0].Embedding)
	}
}

func TestOpenAIEmbeddingsBase64Encoding(t *testing.T) {
	backend := &fakeChatBackend{
		embedResult: &engine.EmbeddingResult{Vector: []float32{0.5, -1.25}, PromptTokens: 2},
	}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"test","input":["alpha","beta"],"encoding_format":"base64"}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 embeddings, got %d: %s", len(resp.Data), rec.Body.String())
	}
	for i, item := range resp.Data {
		decoded, err := base64.StdEncoding.DecodeString(item.Embedding)
		if err != nil {
			t.Fatalf("decode base64 embedding %d: %v", i, err)
		}
		if len(decoded) != 8 {
			t.Fatalf("expected 8 bytes for embedding %d, got %d", i, len(decoded))
		}
		first := math.Float32frombits(binary.LittleEndian.Uint32(decoded[0:4]))
		second := math.Float32frombits(binary.LittleEndian.Uint32(decoded[4:8]))
		if first != 0.5 || second != -1.25 {
			t.Fatalf("unexpected decoded embedding %d: [%v %v]", i, first, second)
		}
	}
}

func TestOpenAIEmbeddingsDimensionsTruncation(t *testing.T) {
	backend := &fakeChatBackend{
		embedResult: &engine.EmbeddingResult{Vector: []float32{0.5, -1.25}, PromptTokens: 2},
	}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"test","input":"hi","dimensions":1}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Data) != 1 || len(resp.Data[0].Embedding) != 1 {
		t.Fatalf("unexpected response: %s", rec.Body.String())
	}
	if resp.Data[0].Embedding[0] != 0.5 {
		t.Fatalf("unexpected truncated embedding: %+v", resp.Data[0].Embedding)
	}
}

func TestOpenAIEmbeddingsBase64EncodingWithDimensions(t *testing.T) {
	backend := &fakeChatBackend{
		embedResult: &engine.EmbeddingResult{Vector: []float32{0.5, -1.25}, PromptTokens: 2},
	}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"test","input":["alpha","beta"],"encoding_format":"base64","dimensions":1}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 embeddings, got %d: %s", len(resp.Data), rec.Body.String())
	}
	for i, item := range resp.Data {
		decoded, err := base64.StdEncoding.DecodeString(item.Embedding)
		if err != nil {
			t.Fatalf("decode base64 embedding %d: %v", i, err)
		}
		if len(decoded) != 4 {
			t.Fatalf("expected 4 bytes for embedding %d, got %d", i, len(decoded))
		}
		value := math.Float32frombits(binary.LittleEndian.Uint32(decoded[0:4]))
		if value != 0.5 {
			t.Fatalf("unexpected decoded embedding %d: %v", i, value)
		}
	}
}

func TestOpenAIEmbeddingsModelNotFoundErrorCode(t *testing.T) {
	backend := &fakeChatBackend{notLoaded: true}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"missing-model","input":"hi"}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if resp.Error.Code != "model_not_found" {
		t.Fatalf("unexpected error payload: %+v", resp)
	}
}

func TestOpenAIEmbeddingsCapabilityMismatchErrorCode(t *testing.T) {
	backend := &fakeChatBackend{embedErr: errors.New("model chat-model does not support embeddings")}
	scheduler := service.NewRuntimeScheduler(backend, 1)
	inference := service.NewInferenceServiceWithScheduler(scheduler)
	h := NewEmbedHandler(service.NewEmbeddingServiceWithScheduler(scheduler), inference)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(`{"model":"chat-model","input":"hi"}`))
	rec := httptest.NewRecorder()

	h.EmbedOpenAI(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if resp.Error.Code != "model_capability_mismatch" {
		t.Fatalf("unexpected error payload: %+v", resp)
	}
}

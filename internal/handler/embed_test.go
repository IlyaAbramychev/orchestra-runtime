package handler

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

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
		`{"model":"test","input":"hi","dimensions":256}`,
		`{"model":"test","input":"hi","encoding_format":"base64"}`,
	}
	for _, body := range cases {
		req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()

		h.EmbedOpenAI(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d: %s", body, rec.Code, rec.Body.String())
		}
	}
}

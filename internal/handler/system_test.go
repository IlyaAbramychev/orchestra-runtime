package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

func TestVersionUsesOllamaShape(t *testing.T) {
	oldVersion := service.Version
	service.Version = "1.2.3-test"
	t.Cleanup(func() {
		service.Version = oldVersion
	})

	handler := NewSystemHandler(nil)
	req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	rec := httptest.NewRecorder()

	handler.Version(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Version != "1.2.3-test" {
		t.Fatalf("version = %q", resp.Version)
	}
}

func TestCapabilitiesReportsFeatureSupport(t *testing.T) {
	oldVersion := service.Version
	service.Version = "1.2.3-test"
	t.Cleanup(func() {
		service.Version = oldVersion
	})

	handler := NewSystemHandler(nil)
	req := httptest.NewRequest(http.MethodGet, "/api/capabilities", nil)
	rec := httptest.NewRecorder()

	handler.Capabilities(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp model.RuntimeCapabilitiesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Service != "orchestra-runtime" || resp.Version != "1.2.3-test" {
		t.Fatalf("unexpected identity: %+v", resp)
	}
	if !resp.Ollama.Compatible || len(resp.Ollama.Endpoints) == 0 {
		t.Fatalf("missing ollama capabilities: %+v", resp.Ollama)
	}

	features := map[string]string{}
	featureNotes := map[string]string{}
	for _, feature := range resp.Features {
		features[feature.Name] = feature.Status
		featureNotes[feature.Name] = feature.Notes
	}
	if features["tool_calls_non_streaming"] != "supported" {
		t.Fatalf("tool_calls_non_streaming = %q", features["tool_calls_non_streaming"])
	}
	if features["multimodal_images"] != "partial" {
		t.Fatalf("multimodal_images = %q", features["multimodal_images"])
	}
	if featureNotes["multimodal_images"] == "" || !strings.Contains(featureNotes["multimodal_images"], "model-scoped") {
		t.Fatalf("multimodal_images note = %q", featureNotes["multimodal_images"])
	}
	if features["direct_gguf_pull"] != "extension" {
		t.Fatalf("direct_gguf_pull = %q", features["direct_gguf_pull"])
	}
	if features["grammar_constrained_decoding"] != "supported" {
		t.Fatalf("grammar_constrained_decoding = %q", features["grammar_constrained_decoding"])
	}
}

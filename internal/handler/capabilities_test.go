package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/operium/orchestra-runtime/internal/model"
)

// TestCapabilitiesEmbeddings verifies that the embeddings capability includes support for base64, dimensions and error codes.
func TestCapabilitiesEmbeddings(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/capabilities", nil)
	w := httptest.NewRecorder()

	h := NewSystemHandler(nil)
	h.Capabilities(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}

	var caps model.RuntimeCapabilitiesResponse
	if err := json.NewDecoder(resp.Body).Decode(&caps); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	var embeddings *model.FeatureCapability
	for i := range caps.Features {
		if caps.Features[i].Name == "embeddings" {
			embeddings = &caps.Features[i]
			break
		}
	}
	if embeddings == nil {
		t.Fatal("embeddings capability missing")
	}
	if embeddings.Status != "supported" {
		t.Fatalf("expected embeddings status supported, got %q", embeddings.Status)
	}

	for _, want := range []string{"base64", "dimensions", "error.code"} {
		if !strings.Contains(embeddings.Notes, want) {
			t.Fatalf("expected embeddings notes to mention %q, got %q", want, embeddings.Notes)
		}
	}
}

func TestCapabilitiesAdvertisesOpenAIMultimodalLimits(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/capabilities", nil)
	w := httptest.NewRecorder()

	NewSystemHandler(nil).Capabilities(w, req)

	var caps model.RuntimeCapabilitiesResponse
	if err := json.NewDecoder(w.Body).Decode(&caps); err != nil {
		t.Fatalf("decode capabilities: %v", err)
	}
	for _, feature := range caps.Features {
		if feature.Name != "multimodal_images" {
			continue
		}
		details, ok := feature.Details.(map[string]any)
		if !ok {
			t.Fatalf("multimodal details missing: %+v", feature)
		}
		if details["remoteURLs"] != false {
			t.Fatalf("unexpected remote URL capability: %+v", details)
		}
		if details["maxImagesPerRequest"] != float64(maxImagesPerRequest) {
			t.Fatalf("unexpected image limit: %+v", details)
		}
		return
	}
	t.Fatal("multimodal_images capability missing")
}

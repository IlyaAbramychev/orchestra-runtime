package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

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

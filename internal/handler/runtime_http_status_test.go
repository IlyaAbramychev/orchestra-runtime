package handler

import (
	"errors"
	"net/http"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
)

// fake errors for testing
var (
	errNil error = nil
	// added missing import for engine package

	errContextOverflow = engine.NewContextLengthExceededError(10, 2048, false)
	// added missing import for engine package

	errMultimodalConfig       = errors.New("configured mmproj")
	errMultimodalProjector    = errors.New("mmproj does not support vision input")
	errModelNotFound          = errors.New("model not found")
	errModelReferenceNotFound = errors.New("model missing-model not found")
	errUnknown                = errors.New("unknown error")
)

func TestRuntimeHTTPStatus(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
	}{
		{"nil", errNil, http.StatusInternalServerError},
		{"context overflow", errContextOverflow, http.StatusBadRequest},
		{"multimodal config", errMultimodalConfig, http.StatusBadRequest},
		{"multimodal projector", errMultimodalProjector, http.StatusBadRequest},
		{"model not found", errModelNotFound, http.StatusNotFound},
		{"model reference not found", errModelReferenceNotFound, http.StatusNotFound},
		{"memory safety budget", errors.New("load would exceed RAM safety budget"), http.StatusUnprocessableEntity},
		{"unsupported model capability", errors.New("model plain-chat does not support thinking"), http.StatusBadRequest},
		{"unknown", errUnknown, http.StatusInternalServerError},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := runtimeHTTPStatus(tc.err)
			if got != tc.want {
				t.Fatalf("expected %d, got %d", tc.want, got)
			}
		})
	}
}

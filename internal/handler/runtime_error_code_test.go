package handler

import (
	"errors"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
)

// TestRuntimeErrorCode verifies that runtimeErrorCode returns the expected code for various error types.
func TestRuntimeErrorCode(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "invalid image input",
			err:  errors.New("failed to decode image input"),
			want: "invalid_image_input",
		},
		{
			name: "invalid image detail",
			err:  errors.New("messages[0].content[1].image_url.detail must be auto, low, or high"),
			want: "invalid_image_input",
		},
		{
			name: "multimodal projector incompatible",
			err:  errors.New("mmproj does not support vision input"),
			want: "multimodal_projector_incompatible",
		},
		{
			name: "multimodal configuration error",
			err:  errors.New("configured mmproj missing"),
			want: "multimodal_configuration_error",
		},
		{
			name: "context overflow",
			err:  engine.NewContextLengthExceededError(5000, 4000, true),
			want: engine.ContextLengthExceededCode,
		},
		{
			name: "model reference not found",
			err:  errors.New("model missing-model not found"),
			want: "model_not_found",
		},
		{
			name: "queue full",
			err:  errors.New("runtime queue full"),
			want: "queue_full",
		},
		{
			name: "memory safety budget",
			err:  errors.New("load would exceed RAM safety budget: model too large"),
			want: "memory_budget_exceeded",
		},
		{
			name: "unknown error",
			err:  errors.New("something went wrong"),
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := runtimeErrorCode(tt.err)
			if got != tt.want {
				t.Fatalf("runtimeErrorCode(%q) = %q; want %q", tt.err.Error(), got, tt.want)
			}
		})
	}
}

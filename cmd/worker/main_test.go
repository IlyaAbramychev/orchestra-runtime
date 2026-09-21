package main

import (
	"fmt"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

func TestInferenceErrorCodePreservesNativeChatFailure(t *testing.T) {
	err := &engine.NativeChatUnavailableError{Cause: fmt.Errorf("invalid Jinja template")}
	if got := inferenceErrorCode(err); got != engine.NativeChatUnavailableCode {
		t.Fatalf("code=%q", got)
	}
	if got := inferenceErrorCode(fmt.Errorf("decode failed")); got != rpc.ErrCodeInference {
		t.Fatalf("generic code=%q", got)
	}
}

func TestToEngineParamsPreservesChatTemplate(t *testing.T) {
	got := toEngineParams(rpc.CompletionParams{ChatTemplate: "{{ custom_template }}"})
	if got.ChatTemplate != "{{ custom_template }}" {
		t.Fatalf("chat template lost in worker RPC conversion: %q", got.ChatTemplate)
	}
}

func TestToEngineMessagesPreservesMultimodalParts(t *testing.T) {
	got := toEngineMessages([]rpc.ChatMessage{{
		Role: "user",
		Parts: []rpc.ContentPart{{
			Type:     "image_url",
			ImageURL: "data:image/png;base64,aGVsbG8=",
		}},
	}})
	if len(got) != 1 || len(got[0].Parts) != 1 {
		t.Fatalf("parts lost in worker conversion: %+v", got)
	}
	if got[0].Parts[0].ImageURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image URL lost: %+v", got[0].Parts[0])
	}
}

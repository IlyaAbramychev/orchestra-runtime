package main

import (
	"testing"

	"github.com/operium/orchestra-runtime/internal/rpc"
)

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

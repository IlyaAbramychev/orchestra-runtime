package model

import (
	"encoding/json"
	"testing"
)

func TestChatMessageUnmarshalOpenAIMultimodalContent(t *testing.T) {
	var message ChatMessage
	err := json.Unmarshal([]byte(`{
		"role":"user",
		"content":[
			{"type":"text","text":"before"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8=","detail":"high"}},
			{"type":"text","text":"after"}
		]
	}`), &message)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if message.Content != "beforeafter" {
		t.Fatalf("aggregated content = %q", message.Content)
	}
	if len(message.Parts) != 3 {
		t.Fatalf("parts = %+v", message.Parts)
	}
	if message.Parts[1].ImageURL != "data:image/png;base64,aGVsbG8=" || message.Parts[1].ImageDetail != "high" {
		t.Fatalf("image part = %+v", message.Parts[1])
	}
}

func TestChatMessageUnmarshalRejectsUnsupportedContentPart(t *testing.T) {
	var message ChatMessage
	err := json.Unmarshal([]byte(`{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}`), &message)
	if err == nil {
		t.Fatal("expected unsupported content part error")
	}
}

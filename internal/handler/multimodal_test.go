package handler

import (
	"strings"
	"testing"

	"github.com/operium/orchestra-runtime/internal/model"
)

func TestValidateMultimodalMessagesRejectsUnsupportedMIMEType(t *testing.T) {
	err := validateMultimodalMessages([]model.ChatMessage{{
		Parts: []model.MessageContentPart{{
			Type:     "image_url",
			ImageURL: "data:image/gif;base64,aGVsbG8=",
		}},
	}})
	if err == nil || !strings.Contains(err.Error(), "unsupported image format") {
		t.Fatalf("expected unsupported format error, got %v", err)
	}
}

func TestValidateMultimodalMessagesRejectsOversizedImage(t *testing.T) {
	payload := strings.Repeat("A", (maxDecodedImageBytes*4/3)+8)
	err := validateMultimodalMessages([]model.ChatMessage{{
		Parts: []model.MessageContentPart{{
			Type:     "image_url",
			ImageURL: "data:image/png;base64," + payload,
		}},
	}})
	if err == nil || !strings.Contains(err.Error(), "decoded image size") {
		t.Fatalf("expected image size error, got %v", err)
	}
}

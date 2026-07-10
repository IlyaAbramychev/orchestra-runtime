package engine

import (
	"bytes"
	"testing"
)

func TestTrimAtStopRemovesStopSequence(t *testing.T) {
	got, stopped := trimAtStop("hello<stop>leak", []string{"<stop>"})
	if !stopped {
		t.Fatal("expected stop")
	}
	if got != "hello" {
		t.Fatalf("expected trimmed text, got %q", got)
	}
}

func TestStopStreamFilterHoldsPotentialStopTail(t *testing.T) {
	f := newStopStreamFilter([]string{"<stop>"})

	if out, stopped := f.PushCheck("hello<st"); stopped || out != "hel" {
		t.Fatalf("first push out=%q stopped=%v", out, stopped)
	}
	if out, stopped := f.PushCheck("op>leak"); !stopped || out != "lo" {
		t.Fatalf("second push out=%q stopped=%v", out, stopped)
	}
	if out := f.Flush(); out != "" {
		t.Fatalf("expected empty flush after stop, got %q", out)
	}
}

func TestStopStreamFilterFlushesWhenNoStop(t *testing.T) {
	f := newStopStreamFilter([]string{"<stop>"})
	out, stopped := f.PushCheck("hello")
	if stopped {
		t.Fatal("did not expect stop")
	}
	if out != "" {
		t.Fatalf("expected held text due stop tail guard, got %q", out)
	}
	if out := f.Flush(); out != "hello" {
		t.Fatalf("expected flush to release held text, got %q", out)
	}
}

func TestBuildPromptReturnsErrorForInvalidCustomTemplate(t *testing.T) {
	e := New()
	_, err := e.buildPrompt([]ChatMessage{{Role: "user", Content: "hello"}}, "{{")
	if err == nil {
		t.Fatal("expected invalid custom template error")
	}
}

func TestDecodeImageBase64AcceptsDataURI(t *testing.T) {
	raw := "aGVsbG8="
	img, err := decodeImageBase64("data:image/png;base64," + raw)
	if err != nil {
		t.Fatalf("decodeImageBase64 returned error: %v", err)
	}
	if string(img) != "hello" {
		t.Fatalf("decoded image = %q", string(img))
	}
}

func TestDecodeMessageImagesPreservesOrderAcrossMessages(t *testing.T) {
	images, err := decodeMessageImages([]ChatMessage{
		{Role: "user", Images: []string{"aGVsbG8=", "d29ybGQ="}},
		{Role: "user", Images: []string{"data:image/png;base64,IQ=="}},
	})
	if err != nil {
		t.Fatalf("decodeMessageImages returned error: %v", err)
	}
	if len(images) != 3 {
		t.Fatalf("expected 3 decoded images, got %d", len(images))
	}
	want := [][]byte{[]byte("hello"), []byte("world"), []byte("!")}
	for i := range want {
		if !bytes.Equal(images[i], want[i]) {
			t.Fatalf("decoded image %d = %q, want %q", i, string(images[i]), string(want[i]))
		}
	}
}

func TestWithMediaMarkersInsertsOneMarkerPerImage(t *testing.T) {
	messages := []ChatMessage{{
		Role:    "user",
		Content: "describe both",
		Images:  []string{"aGVsbG8=", "d29ybGQ="},
	}}

	out := withMediaMarkers(messages)

	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d", len(out))
	}
	if len(out[0].Images) != 0 {
		t.Fatalf("expected images stripped after marker insertion, got %+v", out[0].Images)
	}
	want := mtmdDefaultMarker() + "\n" + mtmdDefaultMarker() + "\ndescribe both"
	if out[0].Content != want {
		t.Fatalf("content = %q, want %q", out[0].Content, want)
	}
	if messages[0].Content != "describe both" || len(messages[0].Images) != 2 {
		t.Fatalf("original messages mutated: %+v", messages[0])
	}
}

func TestWithMediaMarkersPreservesInterleavedContentPartOrder(t *testing.T) {
	messages := []ChatMessage{{
		Role: "user",
		Parts: []ContentPart{
			{Type: "text", Text: "before"},
			{Type: "image_url", ImageURL: "data:image/png;base64,aGVsbG8="},
			{Type: "text", Text: "after"},
		},
	}}

	out := withMediaMarkers(messages)
	want := "before\n" + mtmdDefaultMarker() + "\nafter"
	if out[0].Content != want {
		t.Fatalf("content = %q, want %q", out[0].Content, want)
	}
	if len(out[0].Parts) != 0 {
		t.Fatalf("expected content parts stripped after marker insertion: %+v", out[0].Parts)
	}
	images, err := decodeMessageImages(messages)
	if err != nil {
		t.Fatalf("decodeMessageImages: %v", err)
	}
	if len(images) != 1 || string(images[0]) != "hello" {
		t.Fatalf("decoded images = %+v", images)
	}
}

func TestNativeChatParserSeparatesReasoningFromContent(t *testing.T) {
	content, reasoning := splitReasoningContent("<think>inspect workspace</think>\nNo changes needed.")
	if reasoning != "inspect workspace" || content != "No changes needed." {
		t.Fatalf("reasoning/content not separated: content=%q reasoning=%q", content, reasoning)
	}
}

func TestLooksLikeToolProtocolDetectsDamagedEnvelope(t *testing.T) {
	if !looksLikeToolProtocol(`{"tool_calls":[{"function":{"name":"read_file"}}]}, {"path":"README.md"}]`) {
		t.Fatal("damaged tool envelope was not detected")
	}
	if looksLikeToolProtocol("A normal answer about functions.") {
		t.Fatal("plain content was misclassified as a tool protocol")
	}
}

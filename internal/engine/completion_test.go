package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestNativeRenderFailureIsClassifiedForBothCompletionModes(t *testing.T) {
	eng := New()
	eng.state = StateReady // rendering fails before a llama context is touched
	params := DefaultCompletionParams()
	params.NativeChat = true
	params.ToolsJSON = `[{"type":"function","function":{"name":"read_file"}}]`
	messages := []ChatMessage{{Role: "user", Content: "read"}}
	_, err := eng.Complete(context.Background(), messages, params)
	var unavailable *NativeChatUnavailableError
	if !errors.As(err, &unavailable) || unavailable.Code() != NativeChatUnavailableCode {
		t.Fatalf("Complete error = %v", err)
	}
	_, err = eng.CompleteStream(context.Background(), messages, params)
	if !errors.As(err, &unavailable) {
		t.Fatalf("CompleteStream error = %v", err)
	}
}

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

func TestNativeParseFailureDoesNotExposeRawReasoningOrToolEnvelope(t *testing.T) {
	eng := &Engine{}
	render := &NativeChatRender{Parser: "not a serialized PEG parser"}
	for _, toolsActive := range []bool{false, true} {
		result := &CompletionResult{Text: `<think>secret</think>{"tool_calls":[`, FinishReason: "stop"}
		eng.applyNativeResult(result, result.Text, render, toolsActive)
		want := "chat_protocol_error"
		if toolsActive {
			want = "tool_protocol_error"
		}
		if result.FinishReason != want || result.Text != "" || result.Reasoning != "" {
			t.Fatalf("tools=%v result=%+v", toolsActive, result)
		}
	}
}

func TestNativePartialParserExposesIncrementalPlainContent(t *testing.T) {
	render := &NativeChatRender{} // llama.cpp's content-only parser
	for _, prefix := range []string{"H", "He", "Hel", "Hell", "Hello"} {
		encoded, err := ParseNativeChatPartial(prefix, render)
		if err != nil {
			t.Fatalf("partial parse %q: %v", prefix, err)
		}
		var parsed nativeParsedMessage
		if err := json.Unmarshal(encoded, &parsed); err != nil || parsed.Content != prefix {
			t.Fatalf("partial parse %q = %+v, %v", prefix, parsed, err)
		}
	}
}

func TestContentOnlyPartialParserNeverStreamsOpenThinkTags(t *testing.T) {
	render := &NativeChatRender{} // llama.cpp content-only parser
	for _, tc := range []struct{ raw, visible, reasoning string }{
		{"<think>private reasoning</think>Visible answer", "Visible answer", "private reasoning"},
		{"<think>private reasoning</think>\nVisible answer", "Visible answer", "private reasoning"},
		{"Before <think>private reasoning</think> After", "Before  After", "private reasoning"},
		{"  Before <think>private reasoning</think> After", "  Before  After", "private reasoning"},
		{"<think>private</think>Visible <think> reasoning</think> answer", "Visible  answer", "private reasoning"},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			var sentContent, sentReasoning string
			for i := 1; i <= len(tc.raw); i++ {
				messageJSON, err := ParseNativeChatPartial(tc.raw[:i], render)
				if err != nil {
					t.Fatalf("byte %d: %v", i, err)
				}
				var parsed nativeParsedMessage
				if err := json.Unmarshal(messageJSON, &parsed); err != nil {
					t.Fatalf("byte %d: %v", i, err)
				}
				content, reasoning := splitReasoningContentPartial(parsed.Content)
				if !strings.HasPrefix(content, sentContent) || !strings.HasPrefix(reasoning, sentReasoning) {
					t.Fatalf("byte %d changed emitted prefix: content=%q prior=%q reasoning=%q prior=%q", i, content, sentContent, reasoning, sentReasoning)
				}
				sentContent, sentReasoning = content, reasoning
				if strings.Contains(sentContent, "<think") || strings.Contains(sentContent, "private") || strings.Contains(sentContent, "</think") {
					t.Fatalf("byte %d leaked reasoning into content: %q", i, sentContent)
				}
			}
			result := &CompletionResult{Text: tc.raw, FinishReason: "stop"}
			(&Engine{}).applyNativeResult(result, tc.raw, render, false)
			if !strings.HasPrefix(result.Text, sentContent) || !strings.HasPrefix(result.Reasoning, sentReasoning) {
				t.Fatalf("final parse changed emitted prefix: result=%+v content=%q reasoning=%q", result, sentContent, sentReasoning)
			}
			if result.Text != tc.visible || result.Reasoning != tc.reasoning {
				t.Fatalf("final split mismatch: %+v", result)
			}
		})
	}
}

func TestUnclosedThinkTagStaysOutOfContent(t *testing.T) {
	content, reasoning := splitReasoningContent("<think>private reasoning")
	if content != "" || reasoning != "private reasoning" {
		t.Fatalf("content=%q reasoning=%q", content, reasoning)
	}
}

func TestDamagedToolEnvelopeKeepsProtocolFinishReason(t *testing.T) {
	raw := `{"tool_calls":[{"function":{"name":"read_file","arguments":"{"}}]}`
	result := &CompletionResult{Text: raw, FinishReason: "stop"}
	(&Engine{}).applyNativeResult(result, raw, &NativeChatRender{}, true)
	if result.FinishReason != "tool_protocol_error" || result.Text != "" || len(result.ToolCalls) != 0 {
		t.Fatalf("damaged tool response = %+v", result)
	}
}

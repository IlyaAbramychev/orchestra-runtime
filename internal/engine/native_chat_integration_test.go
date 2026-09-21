package engine

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestNativeChatStreamingIntegration(t *testing.T) {
	modelPath := os.Getenv("ORCHESTRA_TEST_TOOL_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set ORCHESTRA_TEST_TOOL_MODEL_PATH to run native streaming integration")
	}
	eng := New()
	eng.InitBackend()
	defer eng.Close()
	if err := eng.LoadModel("native-stream-fixture", modelPath, LoadOptions{CtxSize: 2048, GPULayers: -1, UseMmap: true}); err != nil {
		t.Fatal(err)
	}
	params := DefaultCompletionParams()
	params.NativeChat = true
	params.ThinkingSet = true
	params.EnableThinking = false
	params.MaxTokens = 256
	params.Temperature = 0
	ch, err := eng.CompleteStream(context.Background(), []ChatMessage{{Role: "user", Content: "Write one sentence of at least twenty words about why clear task ownership helps a team."}}, params)
	if err != nil {
		t.Fatal(err)
	}
	var content, reasoning strings.Builder
	contentChunks, reasoningChunks := 0, 0
	var final CompletionChunk
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatal(chunk.Err)
		}
		content.WriteString(chunk.Text)
		reasoning.WriteString(chunk.Reasoning)
		if chunk.Text != "" && !chunk.Done {
			contentChunks++
		}
		if chunk.Reasoning != "" && !chunk.Done {
			reasoningChunks++
		}
		if chunk.Done {
			final = chunk
		}
	}
	if !final.Done || content.String() == "" || contentChunks+reasoningChunks < 2 || strings.Contains(content.String(), "<|channel|>") || strings.Contains(content.String(), "<think>") {
		t.Fatalf("stream final=%+v content_chunks=%d reasoning_chunks=%d content=%q reasoning=%q", final, contentChunks, reasoningChunks, content.String(), reasoning.String())
	}
	complete, err := eng.Complete(context.Background(), []ChatMessage{{Role: "user", Content: "Write one sentence of at least twenty words about why clear task ownership helps a team."}}, params)
	if err != nil {
		t.Fatal(err)
	}
	if content.String() != complete.Text || reasoning.String() != complete.Reasoning {
		t.Fatalf("stream and non-stream differ: stream content=%q reasoning=%q; complete content=%q reasoning=%q", content.String(), reasoning.String(), complete.Text, complete.Reasoning)
	}
	t.Logf("content chunks=%d content=%q reasoning=%q final_text=%q", contentChunks, content.String(), reasoning.String(), final.Text)
}

// Real-GGUF fixture for the native llama.cpp chat-template/tool pipeline.
// It is intentionally env-gated because CI does not ship multi-GB models.
func TestNativeChatToolTemplateIntegration(t *testing.T) {
	modelPath := os.Getenv("ORCHESTRA_TEST_TOOL_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set ORCHESTRA_TEST_TOOL_MODEL_PATH to run the real native tool-template fixture")
	}
	eng := New()
	eng.InitBackend()
	defer eng.Close()
	if err := eng.LoadModel("native-tool-fixture", modelPath, LoadOptions{CtxSize: 2048, GPULayers: -1, UseMmap: true}); err != nil {
		t.Fatalf("load model: %v", err)
	}
	messages, _ := json.Marshal([]map[string]any{{"role": "user", "content": "Read README.md"}})
	tools := `[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}]`
	render, err := RenderNativeChat(eng.model, "", string(messages), tools, 1, false, false)
	if err != nil {
		t.Fatalf("render native chat: %v", err)
	}
	t.Logf("format=%d grammar_lazy=%v triggers=%d caps=%v generation_prompt=%q", render.Format, render.GrammarLazy, len(render.GrammarTriggers), render.Capabilities, render.GenerationPrompt)
	if len(render.Grammar) > 1000 {
		t.Logf("grammar prefix=%q suffix=%q", render.Grammar[:500], render.Grammar[len(render.Grammar)-500:])
	} else {
		t.Logf("grammar=%q", render.Grammar)
	}
	if render.Prompt == "" || render.Grammar == "" || render.Parser == "" {
		t.Fatalf("native tool render is incomplete: %+v", render)
	}
	if !render.Capabilities["supports_tools"] || !render.Capabilities["supports_tool_calls"] {
		t.Fatalf("fixture template does not advertise native tools: %v", render.Capabilities)
	}
	if os.Getenv("ORCHESTRA_TEST_NATIVE_COMPLETE") != "1" {
		return
	}
	params := DefaultCompletionParams()
	params.MaxTokens = 128
	params.Temperature = 0
	params.Seed = 42
	params.NativeChat = true
	params.ToolsJSON = tools
	params.ToolChoice = 1
	params.ThinkingSet = true
	params.EnableThinking = false
	result, err := eng.Complete(context.Background(), []ChatMessage{{Role: "user", Content: "Call read_file now with path exactly README.md."}}, params)
	if err != nil {
		t.Fatalf("native tool completion: %v", err)
	}
	if result.FinishReason != "tool_calls" || len(result.ToolCalls) == 0 || result.ToolCalls[0].Name != "read_file" {
		t.Fatalf("native tool result = %+v", result)
	}
}

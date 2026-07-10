package engine

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"strings"
	"time"
	"unicode/utf8"
)

// splitUTF8 returns the longest prefix of buf that is valid UTF-8,
// and the trailing bytes (at most 3) that are an incomplete multi-byte
// sequence. Used to hold back partial characters during streaming so that
// clients always see complete code points.
func splitUTF8(buf []byte) (complete []byte, pending []byte) {
	if len(buf) == 0 {
		return nil, nil
	}
	// An incomplete UTF-8 sequence at the tail is at most 3 bytes long.
	// Walk back up to 3 bytes; if that prefix is valid, return it.
	maxTrim := 3
	if maxTrim > len(buf) {
		maxTrim = len(buf)
	}
	if utf8.Valid(buf) {
		return buf, nil
	}
	for trim := 1; trim <= maxTrim; trim++ {
		if utf8.Valid(buf[:len(buf)-trim]) {
			return buf[:len(buf)-trim], append([]byte(nil), buf[len(buf)-trim:]...)
		}
	}
	// Couldn't find a valid prefix — return whole buf as pending.
	return nil, append([]byte(nil), buf...)
}

// CompletionParams controls token generation.
//
// Field coverage matches Ollama's `options` (https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
// and LM Studio's sampling panel so clients can reuse their configs verbatim.
type CompletionParams struct {
	MaxTokens        int
	Temperature      float32
	TopK             int
	TopP             float32
	MinP             float32
	TypicalP         float32
	RepeatPenalty    float32
	RepeatLastN      int
	FrequencyPenalty float32
	PresencePenalty  float32
	// Seed < 0 means random. Any value ≥ 0 makes sampling reproducible.
	Seed int64
	// Mirostat: 0 = off, 1 = v1, 2 = v2. Incompatible with top_k/top_p.
	Mirostat    int
	MirostatTau float32
	MirostatEta float32
	Stop        []string
	// ChatTemplate overrides the model's built-in llama.cpp chat template.
	// Empty string means use the GGUF embedded template.
	ChatTemplate string
	// Grammar is an optional llama.cpp GBNF grammar used to constrain decoding.
	Grammar string
	// NativeChat routes messages and tools through llama.cpp's Jinja chat
	// template and parser instead of prompt-side tool instructions.
	NativeChat        bool
	ToolsJSON         string
	ToolChoice        int // 0 auto, 1 required, 2 none (llama.cpp enum)
	ParallelToolCalls bool
	ThinkingSet       bool
	EnableThinking    bool
	GrammarLazy       bool
	GrammarTriggers   []GrammarTrigger
	GenerationPrompt  string
	/** Raw prompt mode skips the chat template and sends bytes as-is. Used
	 *  for POST /api/generate to allow raw completion-style prompts. */
	RawPrompt bool
}

// DefaultCompletionParams returns sensible defaults.
func DefaultCompletionParams() CompletionParams {
	return CompletionParams{
		MaxTokens:        512,
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.9,
		MinP:             0.05,
		RepeatPenalty:    1.1,
		RepeatLastN:      64,
		FrequencyPenalty: 0,
		PresencePenalty:  0,
		Seed:             -1,
		MirostatTau:      5.0,
		MirostatEta:      0.1,
	}
}

// Timings captures per-request wall-clock measurements so clients can compute
// tok/s and display progress. Fields mirror Ollama's /api/chat response.
type Timings struct {
	// TotalNs is the full request lifetime from entry to final chunk.
	TotalNs int64
	// PromptEvalNs is the time spent decoding the prompt.
	PromptEvalNs int64
	// EvalNs is the time spent generating completion tokens.
	EvalNs int64
}

// CompletionResult is the result of a non-streaming completion.
type CompletionResult struct {
	Text             string
	Reasoning        string
	ToolCalls        []ToolCall
	PromptTokens     int
	TextPromptTokens int
	VisionTokens     int
	CompletionTokens int
	FinishReason     string // "stop", "length"
	Timings          Timings
}

// CompletionChunk is a single token in a streaming completion.
type CompletionChunk struct {
	Text         string
	Reasoning    string
	ToolCalls    []ToolCall
	Done         bool
	FinishReason string
	Err          error
	// Timings and token counts are populated only on the final `Done` chunk.
	PromptTokens     int
	TextPromptTokens int
	VisionTokens     int
	CompletionTokens int
	Timings          Timings
}

type nativeParsedMessage struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content"`
	ToolCalls        []struct {
		ID       string `json:"id"`
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

// Complete generates a completion for the given messages (non-streaming).
func (e *Engine) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (*CompletionResult, error) {
	start := time.Now()

	e.mu.Lock()
	defer e.mu.Unlock()
	// Reset idle timer on every completion request so actively-used models
	// don't get auto-unloaded mid-session.
	defer e.markUsedLocked()

	if e.state != StateReady {
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	promptMessages := messages
	var images [][]byte
	if messagesHaveImages(messages) {
		var err error
		images, err = decodeMessageImages(messages)
		if err != nil {
			return nil, err
		}
		promptMessages = withMediaMarkers(messages)
	}

	var (
		prompt       string
		err          error
		nativeRender *NativeChatRender
	)
	if params.RawPrompt && len(promptMessages) == 1 {
		prompt = promptMessages[0].Content
	} else if params.NativeChat {
		prompt, nativeRender, err = e.buildNativePrompt(promptMessages, &params)
		if err != nil {
			return nil, fmt.Errorf("build native chat prompt: %w", err)
		}
	} else {
		prompt, err = e.buildPrompt(promptMessages, params.ChatTemplate)
		if err != nil {
			return nil, fmt.Errorf("build prompt: %w", err)
		}
	}
	nCtx := e.ctx.NCtx()
	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Clear KV cache for stateless request (OpenAI-compatible semantics)
	e.ctx.ClearKVCache()

	// Evaluate prompt (measure: prompt-eval duration)
	promptStart := time.Now()
	prefillBatchSize := e.promptBatchSize()
	batch := llamaBatchInit(prefillBatchSize, 1)
	defer batch.Free()
	nPrompt, sampleIdx, err := e.prefillPrompt(ctx, batch, prompt, images)
	if err != nil {
		return nil, err
	}
	promptEvalNs := time.Since(promptStart).Nanoseconds()
	textPromptTokens, visionTokens := e.multimodalTokenBreakdown(prompt, nPrompt, len(images) > 0)

	// Guard: llama.cpp SIGSEGVs on batch decode when position >= n_ctx,
	// which takes down the whole process. Refuse overflowing requests here.
	if nPrompt >= nCtx {
		return nil, NewContextLengthExceededError(nPrompt, nCtx, false, maxTokens)
	}
	// Leave 1 slot of headroom so we never hit pos == n_ctx during decode.
	if room := nCtx - nPrompt - 1; maxTokens > room {
		maxTokens = room
	}

	// Create sampler
	sampler, err := e.createSampler(params)
	if err != nil {
		return nil, err
	}
	defer sampler.Free()

	// Generate tokens (measure: eval duration)
	evalStart := time.Now()
	var generated []byte
	nGen := 0
	finishReason := "length"
	pos := nPrompt

	for nGen < maxTokens {
		if ctx.Err() != nil {
			finishReason = "stop"
			break
		}

		token := sampler.Sample(e.ctx, sampleIdx)
		sampler.Accept(token)

		// Check for end of generation
		if e.vocab.IsEOG(token) {
			finishReason = "stop"
			break
		}

		piece := e.vocab.TokenToStr(token)
		generated = append(generated, piece...)
		nGen++

		// Check stop sequences
		if trimmed, ok := trimAtStop(string(generated), params.Stop); ok {
			generated = []byte(trimmed)
			finishReason = "stop"
			break
		}

		// Prepare next batch
		batch.Clear()
		if err := batch.Add(token, pos, 0, true); err != nil {
			return nil, err
		}
		pos++

		if err := llamaDecode(e.ctx, batch); err != nil {
			return nil, fmt.Errorf("decode token %d: %w", nGen, err)
		}
		sampleIdx = batch.NTokens() - 1
	}

	result := &CompletionResult{
		Text:             string(generated),
		PromptTokens:     nPrompt,
		TextPromptTokens: textPromptTokens,
		VisionTokens:     visionTokens,
		CompletionTokens: nGen,
		FinishReason:     finishReason,
		Timings: Timings{
			TotalNs:      time.Since(start).Nanoseconds(),
			PromptEvalNs: promptEvalNs,
			EvalNs:       time.Since(evalStart).Nanoseconds(),
		},
	}
	if nativeRender != nil {
		e.applyNativeResult(result, string(generated), nativeRender, params.ToolsJSON != "")
	}
	return result, nil
}

// CompleteStream generates tokens and sends them to a channel.
func (e *Engine) CompleteStream(ctx context.Context, messages []ChatMessage, params CompletionParams) (<-chan CompletionChunk, error) {
	start := time.Now()

	e.mu.Lock()
	// Reset idle timer up front so the watcher doesn't fire while we generate.
	e.markUsedLocked()

	if e.state != StateReady {
		e.mu.Unlock()
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	promptMessages := messages
	var images [][]byte
	var (
		prompt       string
		err          error
		nativeRender *NativeChatRender
	)
	if messagesHaveImages(messages) {
		images, err = decodeMessageImages(messages)
		if err != nil {
			e.mu.Unlock()
			return nil, err
		}
		promptMessages = withMediaMarkers(messages)
	}
	if params.RawPrompt && len(promptMessages) == 1 {
		prompt = promptMessages[0].Content
	} else if params.NativeChat {
		prompt, nativeRender, err = e.buildNativePrompt(promptMessages, &params)
		if err != nil {
			e.mu.Unlock()
			return nil, fmt.Errorf("build native chat prompt: %w", err)
		}
	} else {
		prompt, err = e.buildPrompt(promptMessages, params.ChatTemplate)
		if err != nil {
			e.mu.Unlock()
			return nil, fmt.Errorf("build prompt: %w", err)
		}
	}
	nCtx := e.ctx.NCtx()
	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	ch := make(chan CompletionChunk, 32)

	go func() {
		defer e.mu.Unlock()
		defer close(ch)
		// Stamp usage again at end-of-stream so long generations don't get
		// auto-unloaded right after finishing.
		defer e.markUsedLocked()

		// Defense-in-depth: if llama.cpp panics via CGo (rare, but happens on
		// some Metal edge cases), don't take down the whole runtime process —
		// surface it through the stream instead.
		defer func() {
			if r := recover(); r != nil {
				ch <- CompletionChunk{Err: fmt.Errorf("inference panic: %v", r)}
			}
		}()

		// Clear KV cache for stateless request (OpenAI-compatible semantics)
		e.ctx.ClearKVCache()

		// Evaluate prompt (measure: prompt-eval duration)
		promptStart := time.Now()
		prefillBatchSize := e.promptBatchSize()
		batch := llamaBatchInit(prefillBatchSize, 1)
		defer batch.Free()
		nPrompt, sampleIdx, err := e.prefillPrompt(ctx, batch, prompt, images)
		if err != nil {
			ch <- CompletionChunk{Err: err}
			return
		}
		promptEvalNs := time.Since(promptStart).Nanoseconds()
		textPromptTokens, visionTokens := e.multimodalTokenBreakdown(prompt, nPrompt, len(images) > 0)

		// Guard: llama.cpp SIGSEGVs on batch decode when position >= n_ctx,
		// which kills the whole process (seen as "terminated" on the HTTP side).
		if nPrompt >= nCtx {
			ch <- CompletionChunk{Err: NewContextLengthExceededError(nPrompt, nCtx, true, maxTokens)}
			return
		}
		// Leave 1 slot so we never hit pos == n_ctx during sampling.
		if room := nCtx - nPrompt - 1; maxTokens > room {
			maxTokens = room
		}

		sampler, err := e.createSampler(params)
		if err != nil {
			ch <- CompletionChunk{Err: err}
			return
		}
		defer sampler.Free()

		// Generation loop (measure: eval duration)
		evalStart := time.Now()
		var nativeOutput strings.Builder
		emit := func(value string) {
			if value == "" {
				return
			}
			if nativeRender != nil {
				nativeOutput.WriteString(value)
				return
			}
			ch <- CompletionChunk{Text: value}
		}
		finalize := func(reason string, nGen int) CompletionChunk {
			chunk := CompletionChunk{
				Done:             true,
				FinishReason:     reason,
				PromptTokens:     nPrompt,
				TextPromptTokens: textPromptTokens,
				VisionTokens:     visionTokens,
				CompletionTokens: nGen,
				Timings: Timings{
					TotalNs:      time.Since(start).Nanoseconds(),
					PromptEvalNs: promptEvalNs,
					EvalNs:       time.Since(evalStart).Nanoseconds(),
				},
			}
			if nativeRender != nil {
				result := &CompletionResult{Text: nativeOutput.String(), FinishReason: reason}
				e.applyNativeResult(result, nativeOutput.String(), nativeRender, params.ToolsJSON != "")
				chunk.Text = result.Text
				chunk.Reasoning = result.Reasoning
				chunk.ToolCalls = result.ToolCalls
				chunk.FinishReason = result.FinishReason
			}
			return chunk
		}

		stopFilter := newStopStreamFilter(params.Stop)
		var pendingBytes []byte // incomplete UTF-8 tail from previous iteration
		nGen := 0
		pos := nPrompt

		for nGen < maxTokens {
			if ctx.Err() != nil {
				if len(pendingBytes) > 0 {
					if out := stopFilter.Push(string(pendingBytes)); out != "" {
						emit(out)
					}
				}
				if out := stopFilter.Flush(); out != "" {
					emit(out)
				}
				ch <- finalize("stop", nGen)
				return
			}

			token := sampler.Sample(e.ctx, sampleIdx)
			sampler.Accept(token)

			if e.vocab.IsEOG(token) {
				if len(pendingBytes) > 0 {
					if out := stopFilter.Push(string(pendingBytes)); out != "" {
						emit(out)
					}
				}
				if out := stopFilter.Flush(); out != "" {
					emit(out)
				}
				ch <- finalize("stop", nGen)
				return
			}

			piece := e.vocab.TokenToStr(token)
			nGen++

			// UTF-8 safe streaming: emit only valid prefix, carry incomplete tail.
			buffer := append(pendingBytes, piece...)
			complete, pending := splitUTF8(buffer)
			pendingBytes = pending
			if len(complete) > 0 {
				out, stopped := stopFilter.PushCheck(string(complete))
				if out != "" {
					emit(out)
				}
				if stopped {
					ch <- finalize("stop", nGen)
					return
				}
			}

			batch.Clear()
			if err := batch.Add(token, pos, 0, true); err != nil {
				ch <- CompletionChunk{Err: err}
				return
			}
			pos++

			if err := llamaDecode(e.ctx, batch); err != nil {
				ch <- CompletionChunk{Err: fmt.Errorf("decode token %d: %w", nGen, err)}
				return
			}
			sampleIdx = batch.NTokens() - 1
		}

		// Hit max_tokens — flush any pending UTF-8 tail before finishing
		if len(pendingBytes) > 0 {
			if out := stopFilter.Push(string(pendingBytes)); out != "" {
				emit(out)
			}
		}
		if out := stopFilter.Flush(); out != "" {
			emit(out)
		}
		ch <- finalize("length", nGen)
	}()

	return ch, nil
}

func (e *Engine) multimodalTokenBreakdown(prompt string, total int, hasImages bool) (textTokens int, visionTokens int) {
	if !hasImages || total <= 0 || e.vocab == nil {
		return 0, 0
	}
	textOnlyPrompt := strings.ReplaceAll(prompt, mtmdDefaultMarker(), "")
	textTokens = len(e.vocab.Tokenize(textOnlyPrompt, true, true))
	if textTokens > total {
		textTokens = total
	}
	return textTokens, total - textTokens
}

func (e *Engine) promptBatchSize() int {
	// n_batch can be lower than prompt length; prefill must be chunked by this limit.
	if e.batchSize > 0 {
		return e.batchSize
	}
	return 512
}

func (e *Engine) prefillPrompt(ctx context.Context, batch *llamaBatch, prompt string, images [][]byte) (int, int, error) {
	if len(images) > 0 {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		nPast, err := mtmdEvalPrompt(e.mtmd, e.ctx, prompt, images, e.promptBatchSize())
		if err != nil {
			return 0, 0, err
		}
		batch.Clear()
		return nPast, -1, nil
	}

	tokens := e.vocab.Tokenize(prompt, true, true)
	if len(tokens) == 0 {
		return 0, 0, fmt.Errorf("tokenization produced no tokens")
	}
	for startIdx := 0; startIdx < len(tokens); startIdx += e.promptBatchSize() {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		endIdx := startIdx + e.promptBatchSize()
		if endIdx > len(tokens) {
			endIdx = len(tokens)
		}
		batch.Clear()
		for i := startIdx; i < endIdx; i++ {
			logits := i == len(tokens)-1
			if err := batch.Add(tokens[i], i, 0, logits); err != nil {
				return 0, 0, err
			}
		}
		if err := llamaDecode(e.ctx, batch); err != nil {
			return 0, 0, fmt.Errorf("decode prompt chunk %d-%d: %w", startIdx, endIdx, err)
		}
	}
	return len(tokens), batch.NTokens() - 1, nil
}

func messagesHaveImages(messages []ChatMessage) bool {
	for _, msg := range messages {
		if len(msg.Images) > 0 {
			return true
		}
		for _, part := range msg.Parts {
			if part.Type == "image_url" {
				return true
			}
		}
	}
	return false
}

func withMediaMarkers(messages []ChatMessage) []ChatMessage {
	out := make([]ChatMessage, len(messages))
	marker := mtmdDefaultMarker()
	for i, msg := range messages {
		out[i] = msg
		if !messageHasImage(msg) {
			continue
		}
		var builder strings.Builder
		endsWithNewline := false
		appendBoundary := func() {
			if builder.Len() > 0 && !endsWithNewline {
				builder.WriteByte('\n')
				endsWithNewline = true
			}
		}
		for _, part := range msg.Parts {
			switch part.Type {
			case "text":
				if builder.Len() > 0 && !endsWithNewline && strings.HasSuffix(builder.String(), marker) {
					builder.WriteByte('\n')
				}
				builder.WriteString(part.Text)
				endsWithNewline = strings.HasSuffix(part.Text, "\n")
			case "image_url":
				appendBoundary()
				builder.WriteString(marker)
				endsWithNewline = false
			}
		}
		for range msg.Images {
			appendBoundary()
			builder.WriteString(marker)
			endsWithNewline = false
		}
		if len(msg.Parts) == 0 && msg.Content != "" {
			appendBoundary()
			builder.WriteString(msg.Content)
		}
		out[i].Content = builder.String()
		out[i].Parts = nil
		out[i].Images = nil
	}
	return out
}

func messageHasImage(msg ChatMessage) bool {
	if len(msg.Images) > 0 {
		return true
	}
	for _, part := range msg.Parts {
		if part.Type == "image_url" {
			return true
		}
	}
	return false
}

func decodeMessageImages(messages []ChatMessage) ([][]byte, error) {
	var out [][]byte
	for _, msg := range messages {
		for idx, part := range msg.Parts {
			if part.Type != "image_url" {
				continue
			}
			img, err := decodeImageBase64(part.ImageURL)
			if err != nil {
				return nil, fmt.Errorf("decode content image %d: %w", idx, err)
			}
			out = append(out, img)
		}
		for idx, raw := range msg.Images {
			img, err := decodeImageBase64(raw)
			if err != nil {
				return nil, fmt.Errorf("decode image %d: %w", idx, err)
			}
			out = append(out, img)
		}
	}
	return out, nil
}

func decodeImageBase64(raw string) ([]byte, error) {
	encoded := strings.TrimSpace(raw)
	if encoded == "" {
		return nil, fmt.Errorf("empty image payload")
	}
	if comma := strings.Index(encoded, ","); comma > 0 && strings.HasPrefix(encoded[:comma], "data:") {
		encoded = encoded[comma+1:]
	}
	img, err := base64.StdEncoding.DecodeString(encoded)
	if err == nil {
		return img, nil
	}
	img, rawErr := base64.RawStdEncoding.DecodeString(encoded)
	if rawErr == nil {
		return img, nil
	}
	return nil, fmt.Errorf("invalid base64 image: %w", err)
}

// buildPrompt applies the chat template to convert messages into a prompt string.
func (e *Engine) buildPrompt(messages []ChatMessage, template string) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages")
	}

	// Empty template tells llama.cpp to use the model's embedded template.
	result, err := ApplyChatTemplate(template, messages, true)
	if err != nil {
		if template != "" {
			return "", fmt.Errorf("custom chat template failed: %w", err)
		}
		slog.Warn("chat template failed, falling back to ChatML", "error", err)
		return buildChatMLPrompt(messages), nil
	}
	return result, nil
}

func (e *Engine) buildNativePrompt(messages []ChatMessage, params *CompletionParams) (string, *NativeChatRender, error) {
	if len(messages) == 0 {
		return "", nil, fmt.Errorf("no messages")
	}
	encodedMessages, err := marshalNativeMessages(messages)
	if err != nil {
		return "", nil, err
	}
	enableThinking := true
	if params.ThinkingSet {
		enableThinking = params.EnableThinking
	}
	render, err := RenderNativeChat(
		e.model,
		params.ChatTemplate,
		string(encodedMessages),
		params.ToolsJSON,
		params.ToolChoice,
		params.ParallelToolCalls,
		enableThinking,
	)
	if err != nil {
		return "", nil, err
	}
	if params.ToolsJSON != "" && (!render.Capabilities["supports_tools"] || !render.Capabilities["supports_tool_calls"]) {
		slog.Warn("model chat template has limited native tool support",
			"supports_tools", render.Capabilities["supports_tools"],
			"supports_tool_calls", render.Capabilities["supports_tool_calls"])
	}
	if render.Grammar != "" {
		params.Grammar = render.Grammar
		params.GrammarLazy = render.GrammarLazy
		params.GrammarTriggers = append([]GrammarTrigger(nil), render.GrammarTriggers...)
		params.GenerationPrompt = render.GenerationPrompt
	}
	for _, stop := range render.AdditionalStops {
		if !containsString(params.Stop, stop) {
			params.Stop = append(params.Stop, stop)
		}
	}
	return render.Prompt, render, nil
}

func marshalNativeMessages(messages []ChatMessage) ([]byte, error) {
	out := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		entry := map[string]any{"role": message.Role, "content": message.Content}
		if message.Reasoning != "" {
			entry["reasoning_content"] = message.Reasoning
		}
		if message.ToolName != "" {
			entry["name"] = message.ToolName
		}
		if message.ToolCallID != "" {
			entry["tool_call_id"] = message.ToolCallID
		}
		if len(message.ToolCalls) > 0 {
			calls := make([]map[string]any, 0, len(message.ToolCalls))
			for _, call := range message.ToolCalls {
				arguments := call.Arguments
				if len(arguments) == 0 {
					arguments = json.RawMessage(`{}`)
				}
				if !json.Valid(arguments) {
					return nil, fmt.Errorf("tool call %q has invalid JSON arguments", call.Name)
				}
				calls = append(calls, map[string]any{
					"id":   call.ID,
					"type": "function",
					"function": map[string]any{
						"name":      call.Name,
						"arguments": json.RawMessage(arguments),
					},
				})
			}
			entry["tool_calls"] = calls
		}
		out = append(out, entry)
	}
	return json.Marshal(out)
}

func (e *Engine) applyNativeResult(result *CompletionResult, raw string, render *NativeChatRender, toolsActive bool) {
	messageJSON, err := ParseNativeChat(raw, render)
	if err != nil {
		if toolsActive {
			result.FinishReason = "tool_protocol_error"
		}
		return
	}
	var parsed nativeParsedMessage
	if err := json.Unmarshal(messageJSON, &parsed); err != nil {
		if toolsActive {
			result.FinishReason = "tool_protocol_error"
		}
		return
	}
	if parsed.ReasoningContent == "" {
		parsed.Content, parsed.ReasoningContent = splitReasoningContent(parsed.Content)
	}
	result.Text = parsed.Content
	result.Reasoning = parsed.ReasoningContent
	result.ToolCalls = result.ToolCalls[:0]
	for _, call := range parsed.ToolCalls {
		arguments, valid := normalizeToolArguments(call.Function.Arguments)
		if strings.TrimSpace(call.Function.Name) == "" || !valid {
			result.Text = raw
			result.ToolCalls = nil
			result.FinishReason = "tool_protocol_error"
			return
		}
		result.ToolCalls = append(result.ToolCalls, ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: arguments,
		})
	}
	if len(result.ToolCalls) > 0 {
		result.FinishReason = "tool_calls"
	} else if toolsActive && looksLikeToolProtocol(raw) {
		result.Text = raw
		result.FinishReason = "tool_protocol_error"
	}
}

func splitReasoningContent(content string) (string, string) {
	start := strings.Index(content, "<think>")
	end := strings.Index(content, "</think>")
	if start < 0 || end < start {
		return content, ""
	}
	reasoning := strings.TrimSpace(content[start+len("<think>") : end])
	visible := strings.TrimSpace(content[:start] + content[end+len("</think>"):])
	return visible, reasoning
}

func looksLikeToolProtocol(raw string) bool {
	lower := strings.ToLower(raw)
	markers := []string{`"tool_calls"`, `"function"`, "<tool_call", "<function=", "[tool]", "tool_call_id"}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func normalizeToolArguments(raw json.RawMessage) (json.RawMessage, bool) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return json.RawMessage(`{}`), true
	}
	if strings.HasPrefix(trimmed, `"`) {
		var encoded string
		if err := json.Unmarshal(raw, &encoded); err != nil {
			return nil, false
		}
		trimmed = strings.TrimSpace(encoded)
	}
	var object map[string]any
	if err := json.Unmarshal([]byte(trimmed), &object); err != nil || object == nil {
		return nil, false
	}
	normalized, err := json.Marshal(object)
	return json.RawMessage(normalized), err == nil
}

func containsString(values []string, candidate string) bool {
	for _, value := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

// buildChatMLPrompt is a fallback when the model has no built-in template.
func buildChatMLPrompt(messages []ChatMessage) string {
	var prompt string
	for _, msg := range messages {
		prompt += "<|im_start|>" + msg.Role + "\n" + msg.Content + "<|im_end|>\n"
	}
	prompt += "<|im_start|>assistant\n"
	return prompt
}

func (e *Engine) createSampler(params CompletionParams) (*llamaSampler, error) {
	// Greedy sampler bypasses the whole chain — fastest, fully deterministic.
	if params.Grammar == "" && params.Temperature <= 0 && params.Mirostat == 0 {
		return NewGreedySampler(), nil
	}
	seed := uint32(params.Seed)
	if params.Seed < 0 {
		seed = rand.Uint32()
	}
	var nVocab int32
	if e.vocab != nil {
		nVocab = int32(e.vocab.NTokens())
	}
	return NewSamplerChain(SamplerOpts{
		Temp:             params.Temperature,
		TopK:             params.TopK,
		TopP:             params.TopP,
		MinP:             params.MinP,
		TypicalP:         params.TypicalP,
		RepeatPenalty:    params.RepeatPenalty,
		RepeatLastN:      params.RepeatLastN,
		FrequencyPenalty: params.FrequencyPenalty,
		PresencePenalty:  params.PresencePenalty,
		Mirostat:         params.Mirostat,
		MirostatTau:      params.MirostatTau,
		MirostatEta:      params.MirostatEta,
		Seed:             seed,
		NVocab:           nVocab,
		Vocab:            e.vocab,
		Model:            e.model,
		Grammar:          params.Grammar,
		GrammarLazy:      params.GrammarLazy,
		GrammarTriggers:  params.GrammarTriggers,
		GenerationPrompt: params.GenerationPrompt,
	})
}

func trimAtStop(text string, stopSeqs []string) (string, bool) {
	idx := -1
	for _, stop := range stopSeqs {
		if stop == "" {
			continue
		}
		if i := strings.Index(text, stop); i >= 0 && (idx == -1 || i < idx) {
			idx = i
		}
	}
	if idx == -1 {
		return text, false
	}
	return text[:idx], true
}

type stopStreamFilter struct {
	stops      []string
	maxStopLen int
	hold       string
}

func newStopStreamFilter(stops []string) *stopStreamFilter {
	f := &stopStreamFilter{stops: stops}
	for _, stop := range stops {
		if len(stop) > f.maxStopLen {
			f.maxStopLen = len(stop)
		}
	}
	return f
}

func (f *stopStreamFilter) Push(text string) string {
	out, _ := f.PushCheck(text)
	return out
}

func (f *stopStreamFilter) PushCheck(text string) (string, bool) {
	if len(f.stops) == 0 || f.maxStopLen == 0 {
		return text, false
	}
	f.hold += text
	if trimmed, ok := trimAtStop(f.hold, f.stops); ok {
		f.hold = ""
		return trimmed, true
	}
	keep := f.maxStopLen - 1
	if keep <= 0 || len(f.hold) <= keep {
		return "", false
	}
	emitLen := len(f.hold) - keep
	out := f.hold[:emitLen]
	f.hold = f.hold[emitLen:]
	return out, false
}

func (f *stopStreamFilter) Flush() string {
	out := f.hold
	f.hold = ""
	return out
}

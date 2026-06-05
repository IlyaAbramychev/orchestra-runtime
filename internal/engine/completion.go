package engine

import (
	"context"
	"encoding/base64"
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
	PromptTokens     int
	CompletionTokens int
	FinishReason     string // "stop", "length"
	Timings          Timings
}

// CompletionChunk is a single token in a streaming completion.
type CompletionChunk struct {
	Text         string
	Done         bool
	FinishReason string
	Err          error
	// Timings and token counts are populated only on the final `Done` chunk.
	PromptTokens     int
	CompletionTokens int
	Timings          Timings
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
		prompt string
		err    error
	)
	if params.RawPrompt && len(promptMessages) == 1 {
		prompt = promptMessages[0].Content
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

	return &CompletionResult{
		Text:             string(generated),
		PromptTokens:     nPrompt,
		CompletionTokens: nGen,
		FinishReason:     finishReason,
		Timings: Timings{
			TotalNs:      time.Since(start).Nanoseconds(),
			PromptEvalNs: promptEvalNs,
			EvalNs:       time.Since(evalStart).Nanoseconds(),
		},
	}, nil
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
		prompt string
		err    error
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
		finalize := func(reason string, nGen int) CompletionChunk {
			return CompletionChunk{
				Done:             true,
				FinishReason:     reason,
				PromptTokens:     nPrompt,
				CompletionTokens: nGen,
				Timings: Timings{
					TotalNs:      time.Since(start).Nanoseconds(),
					PromptEvalNs: promptEvalNs,
					EvalNs:       time.Since(evalStart).Nanoseconds(),
				},
			}
		}

		stopFilter := newStopStreamFilter(params.Stop)
		var pendingBytes []byte // incomplete UTF-8 tail from previous iteration
		nGen := 0
		pos := nPrompt

		for nGen < maxTokens {
			if ctx.Err() != nil {
				if len(pendingBytes) > 0 {
					if out := stopFilter.Push(string(pendingBytes)); out != "" {
						ch <- CompletionChunk{Text: out}
					}
				}
				if out := stopFilter.Flush(); out != "" {
					ch <- CompletionChunk{Text: out}
				}
				ch <- finalize("stop", nGen)
				return
			}

			token := sampler.Sample(e.ctx, sampleIdx)
			sampler.Accept(token)

			if e.vocab.IsEOG(token) {
				if len(pendingBytes) > 0 {
					if out := stopFilter.Push(string(pendingBytes)); out != "" {
						ch <- CompletionChunk{Text: out}
					}
				}
				if out := stopFilter.Flush(); out != "" {
					ch <- CompletionChunk{Text: out}
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
					ch <- CompletionChunk{Text: out}
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
				ch <- CompletionChunk{Text: out}
			}
		}
		if out := stopFilter.Flush(); out != "" {
			ch <- CompletionChunk{Text: out}
		}
		ch <- finalize("length", nGen)
	}()

	return ch, nil
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
	}
	return false
}

func withMediaMarkers(messages []ChatMessage) []ChatMessage {
	out := make([]ChatMessage, len(messages))
	marker := mtmdDefaultMarker()
	for i, msg := range messages {
		out[i] = msg
		if len(msg.Images) == 0 {
			continue
		}
		var builder strings.Builder
		for idx := range msg.Images {
			if idx > 0 {
				builder.WriteByte('\n')
			}
			builder.WriteString(marker)
		}
		if msg.Content != "" {
			builder.WriteByte('\n')
			builder.WriteString(msg.Content)
		}
		out[i].Content = builder.String()
		out[i].Images = nil
	}
	return out
}

func decodeMessageImages(messages []ChatMessage) ([][]byte, error) {
	var out [][]byte
	for _, msg := range messages {
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
		Grammar:          params.Grammar,
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

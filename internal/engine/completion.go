package engine

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
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
	defer func() { e.lastUsedAt = time.Now() }()

	if e.state != StateReady {
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	var (
		prompt string
		err    error
	)
	if params.RawPrompt && len(messages) == 1 {
		prompt = messages[0].Content
	} else {
		prompt, err = e.buildPrompt(messages)
		if err != nil {
			return nil, fmt.Errorf("build prompt: %w", err)
		}
	}

	tokens := e.vocab.Tokenize(prompt, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenization produced no tokens")
	}

	nPrompt := len(tokens)
	nCtx := e.ctx.NCtx()

	// Guard: llama.cpp SIGSEGVs on batch decode when position >= n_ctx,
	// which takes down the whole process. Refuse overflowing requests here.
	if nPrompt >= nCtx {
		return nil, fmt.Errorf(
			"prompt too long: %d tokens ≥ context window %d — reduce attachments/text or load the model with a bigger n_ctx",
			nPrompt, nCtx,
		)
	}

	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	// Leave 1 slot of headroom so we never hit pos == n_ctx during decode.
	if room := nCtx - nPrompt - 1; maxTokens > room {
		maxTokens = room
	}

	// Clear KV cache for stateless request (OpenAI-compatible semantics)
	e.ctx.ClearKVCache()

	// Evaluate prompt (measure: prompt-eval duration)
	promptStart := time.Now()
	batch := llamaBatchInit(len(tokens), 1)
	defer batch.Free()

	for i, tok := range tokens {
		logits := i == len(tokens)-1
		batch.Add(tok, i, 0, logits)
	}

	if err := llamaDecode(e.ctx, batch); err != nil {
		return nil, fmt.Errorf("decode prompt: %w", err)
	}
	promptEvalNs := time.Since(promptStart).Nanoseconds()

	// Create sampler
	sampler := e.createSampler(params)
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

		token := sampler.Sample(e.ctx, batch.NTokens()-1)
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
		if shouldStop(string(generated), params.Stop) {
			finishReason = "stop"
			break
		}

		// Prepare next batch
		batch.Clear()
		batch.Add(token, pos, 0, true)
		pos++

		if err := llamaDecode(e.ctx, batch); err != nil {
			return nil, fmt.Errorf("decode token %d: %w", nGen, err)
		}
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
	e.lastUsedAt = time.Now()

	if e.state != StateReady {
		e.mu.Unlock()
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	var (
		prompt string
		err    error
	)
	if params.RawPrompt && len(messages) == 1 {
		prompt = messages[0].Content
	} else {
		prompt, err = e.buildPrompt(messages)
		if err != nil {
			e.mu.Unlock()
			return nil, fmt.Errorf("build prompt: %w", err)
		}
	}

	tokens := e.vocab.Tokenize(prompt, true, true)
	if len(tokens) == 0 {
		e.mu.Unlock()
		return nil, fmt.Errorf("tokenization produced no tokens")
	}

	nPrompt := len(tokens)
	nCtx := e.ctx.NCtx()

	// Guard: llama.cpp SIGSEGVs on batch decode when position ≥ n_ctx, which
	// kills the whole process (seen as "terminated" on the HTTP side). Reject
	// before handing tokens to the C side.
	if nPrompt >= nCtx {
		e.mu.Unlock()
		return nil, fmt.Errorf(
			"prompt too long: %d tokens ≥ context window %d — reduce attachments/text or reload the model with a bigger n_ctx",
			nPrompt, nCtx,
		)
	}

	ch := make(chan CompletionChunk, 32)

	go func() {
		defer e.mu.Unlock()
		defer close(ch)
		// Stamp usage again at end-of-stream so long generations don't get
		// auto-unloaded right after finishing.
		defer func() { e.lastUsedAt = time.Now() }()

		// Defense-in-depth: if llama.cpp panics via CGo (rare, but happens on
		// some Metal edge cases), don't take down the whole runtime process —
		// surface it through the stream instead.
		defer func() {
			if r := recover(); r != nil {
				ch <- CompletionChunk{Err: fmt.Errorf("inference panic: %v", r)}
			}
		}()

		maxTokens := params.MaxTokens
		if maxTokens <= 0 {
			maxTokens = 512
		}
		// Leave 1 slot so we never hit pos == n_ctx during sampling.
		if room := nCtx - nPrompt - 1; maxTokens > room {
			maxTokens = room
		}

		// Clear KV cache for stateless request (OpenAI-compatible semantics)
		e.ctx.ClearKVCache()

		// Evaluate prompt (measure: prompt-eval duration)
		promptStart := time.Now()
		batch := llamaBatchInit(len(tokens), 1)
		defer batch.Free()

		for i, tok := range tokens {
			logits := i == len(tokens)-1
			batch.Add(tok, i, 0, logits)
		}

		if err := llamaDecode(e.ctx, batch); err != nil {
			ch <- CompletionChunk{Err: fmt.Errorf("decode prompt: %w", err)}
			return
		}
		promptEvalNs := time.Since(promptStart).Nanoseconds()

		sampler := e.createSampler(params)
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

		var generated []byte
		var pendingBytes []byte // incomplete UTF-8 tail from previous iteration
		nGen := 0
		pos := nPrompt

		for nGen < maxTokens {
			if ctx.Err() != nil {
				if len(pendingBytes) > 0 {
					ch <- CompletionChunk{Text: string(pendingBytes)}
				}
				ch <- finalize("stop", nGen)
				return
			}

			token := sampler.Sample(e.ctx, batch.NTokens()-1)
			sampler.Accept(token)

			if e.vocab.IsEOG(token) {
				if len(pendingBytes) > 0 {
					ch <- CompletionChunk{Text: string(pendingBytes)}
				}
				ch <- finalize("stop", nGen)
				return
			}

			piece := e.vocab.TokenToStr(token)
			generated = append(generated, piece...)
			nGen++

			// UTF-8 safe streaming: emit only valid prefix, carry incomplete tail.
			buffer := append(pendingBytes, piece...)
			complete, pending := splitUTF8(buffer)
			pendingBytes = pending
			if len(complete) > 0 {
				ch <- CompletionChunk{Text: string(complete)}
			}

			if shouldStop(string(generated), params.Stop) {
				if len(pendingBytes) > 0 {
					ch <- CompletionChunk{Text: string(pendingBytes)}
				}
				ch <- finalize("stop", nGen)
				return
			}

			batch.Clear()
			batch.Add(token, pos, 0, true)
			pos++

			if err := llamaDecode(e.ctx, batch); err != nil {
				ch <- CompletionChunk{Err: fmt.Errorf("decode token %d: %w", nGen, err)}
				return
			}
		}

		// Hit max_tokens — flush any pending UTF-8 tail before finishing
		if len(pendingBytes) > 0 {
			ch <- CompletionChunk{Text: string(pendingBytes)}
		}
		ch <- finalize("length", nGen)
	}()

	return ch, nil
}

// buildPrompt applies the chat template to convert messages into a prompt string.
func (e *Engine) buildPrompt(messages []ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages")
	}

	// Try model's built-in template first (tmpl = "")
	result, err := ApplyChatTemplate("", messages, true)
	if err != nil {
		slog.Warn("built-in chat template failed, falling back to ChatML", "error", err)
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

func (e *Engine) createSampler(params CompletionParams) *llamaSampler {
	// Greedy sampler bypasses the whole chain — fastest, fully deterministic.
	if params.Temperature <= 0 && params.Mirostat == 0 {
		return NewGreedySampler()
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
	})
}

func shouldStop(text string, stopSeqs []string) bool {
	for _, stop := range stopSeqs {
		if stop != "" && len(text) >= len(stop) {
			tail := text[len(text)-len(stop):]
			if tail == stop {
				return true
			}
		}
	}
	return false
}

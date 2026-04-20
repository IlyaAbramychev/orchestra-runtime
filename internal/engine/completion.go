package engine

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
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
type CompletionParams struct {
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	Stop        []string
}

// DefaultCompletionParams returns sensible defaults.
func DefaultCompletionParams() CompletionParams {
	return CompletionParams{
		MaxTokens:   512,
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
	}
}

// CompletionResult is the result of a non-streaming completion.
type CompletionResult struct {
	Text             string
	PromptTokens     int
	CompletionTokens int
	FinishReason     string // "stop", "length"
}

// CompletionChunk is a single token in a streaming completion.
type CompletionChunk struct {
	Text         string
	Done         bool
	FinishReason string
	Err          error
}

// Complete generates a completion for the given messages (non-streaming).
func (e *Engine) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (*CompletionResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.state != StateReady {
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	prompt, err := e.buildPrompt(messages)
	if err != nil {
		return nil, fmt.Errorf("build prompt: %w", err)
	}

	tokens := e.vocab.Tokenize(prompt, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenization produced no tokens")
	}

	nPrompt := len(tokens)
	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Clear KV cache for stateless request (OpenAI-compatible semantics)
	e.ctx.ClearKVCache()

	// Evaluate prompt
	batch := llamaBatchInit(len(tokens), 1)
	defer batch.Free()

	for i, tok := range tokens {
		logits := i == len(tokens)-1
		batch.Add(tok, i, 0, logits)
	}

	if err := llamaDecode(e.ctx, batch); err != nil {
		return nil, fmt.Errorf("decode prompt: %w", err)
	}

	// Create sampler
	sampler := e.createSampler(params)
	defer sampler.Free()

	// Generate tokens
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
	}, nil
}

// CompleteStream generates tokens and sends them to a channel.
func (e *Engine) CompleteStream(ctx context.Context, messages []ChatMessage, params CompletionParams) (<-chan CompletionChunk, error) {
	e.mu.Lock()

	if e.state != StateReady {
		e.mu.Unlock()
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	prompt, err := e.buildPrompt(messages)
	if err != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("build prompt: %w", err)
	}

	tokens := e.vocab.Tokenize(prompt, true, true)
	if len(tokens) == 0 {
		e.mu.Unlock()
		return nil, fmt.Errorf("tokenization produced no tokens")
	}

	ch := make(chan CompletionChunk, 32)

	go func() {
		defer e.mu.Unlock()
		defer close(ch)

		nPrompt := len(tokens)
		maxTokens := params.MaxTokens
		if maxTokens <= 0 {
			maxTokens = 512
		}

		// Clear KV cache for stateless request (OpenAI-compatible semantics)
		e.ctx.ClearKVCache()

		// Evaluate prompt
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

		sampler := e.createSampler(params)
		defer sampler.Free()

		var generated []byte
		var pendingBytes []byte // incomplete UTF-8 tail from previous iteration
		nGen := 0
		pos := nPrompt

		for nGen < maxTokens {
			if ctx.Err() != nil {
				// Flush any pending bytes as a best-effort emit
				if len(pendingBytes) > 0 {
					ch <- CompletionChunk{Text: string(pendingBytes)}
				}
				ch <- CompletionChunk{Done: true, FinishReason: "stop"}
				return
			}

			token := sampler.Sample(e.ctx, batch.NTokens()-1)
			sampler.Accept(token)

			if e.vocab.IsEOG(token) {
				// Flush remainder (usually empty or valid by this point)
				if len(pendingBytes) > 0 {
					ch <- CompletionChunk{Text: string(pendingBytes)}
				}
				ch <- CompletionChunk{Done: true, FinishReason: "stop"}
				return
			}

			piece := e.vocab.TokenToStr(token)
			generated = append(generated, piece...)
			nGen++

			// UTF-8 safe streaming: accumulate pending + new piece,
			// emit only the valid-UTF-8 prefix, carry incomplete tail.
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
				ch <- CompletionChunk{Done: true, FinishReason: "stop"}
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
		ch <- CompletionChunk{Done: true, FinishReason: "length"}
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
	if params.Temperature <= 0 {
		return NewGreedySampler()
	}
	seed := rand.Uint32()
	return NewSamplerChain(params.Temperature, params.TopK, params.TopP, seed)
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

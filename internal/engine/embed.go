package engine

import (
	"context"
	"fmt"
	"math"
	"time"
)

// EmbeddingResult is the output of Embed for one input string.
type EmbeddingResult struct {
	Vector       []float32
	PromptTokens int
}

// Embed computes a single embedding vector for the given text. Uses the
// currently loaded model — works best with models trained for it (bge, nomic,
// e5, etc.); generative models give usable but not great embeddings.
//
// Pooling: we use mean-pooling over token embeddings (pooling_type=NONE from
// context, we apply mean manually). This matches the default behaviour of
// sentence-transformers for most bge/e5 models.
func (e *Engine) Embed(ctx context.Context, text string, normalize bool) (*EmbeddingResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	defer func() { e.lastUsedAt = time.Now() }()

	if e.state != StateReady {
		return nil, fmt.Errorf("engine not ready (state: %s)", e.state)
	}

	tokens := e.vocab.Tokenize(text, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenization produced no tokens")
	}

	nCtx := e.ctx.NCtx()
	if len(tokens) >= nCtx {
		return nil, fmt.Errorf("input too long: %d tokens ≥ n_ctx %d", len(tokens), nCtx)
	}

	nEmbd := e.model.NEmbd()
	if nEmbd <= 0 {
		return nil, fmt.Errorf("model reports n_embd = %d; not an embedding-capable model", nEmbd)
	}

	// Swap into embedding mode — llama.cpp now populates embedding buffers
	// instead of logits on decode. Always restore afterwards.
	e.ctx.SetEmbeddings(true)
	defer e.ctx.SetEmbeddings(false)
	e.ctx.ClearKVCache()

	batch := llamaBatchInit(len(tokens), 1)
	defer batch.Free()

	// For non-pooled models we need logits on every position so we can mean-pool.
	for i, tok := range tokens {
		batch.Add(tok, i, 0, true)
	}

	if err := llamaDecode(e.ctx, batch); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	// Try seq-level pooled embedding first (works if ctx has pooling_type set
	// via gguf metadata, e.g. bge models). Falls back to per-token mean.
	vec := e.ctx.EmbeddingsSeq(0, nEmbd)
	if vec == nil || allZero(vec) {
		vec = meanPool(e.ctx, len(tokens), nEmbd)
		if vec == nil {
			return nil, fmt.Errorf("failed to read embeddings")
		}
	}

	if normalize {
		l2Normalize(vec)
	}

	return &EmbeddingResult{Vector: vec, PromptTokens: len(tokens)}, nil
}

func meanPool(ctx *llamaContext, nTokens, nEmbd int) []float32 {
	acc := make([]float64, nEmbd)
	nonZero := 0
	for i := 0; i < nTokens; i++ {
		row := ctx.EmbeddingsIth(i, nEmbd)
		if row == nil {
			continue
		}
		nonZero++
		for j, v := range row {
			acc[j] += float64(v)
		}
	}
	if nonZero == 0 {
		return nil
	}
	out := make([]float32, nEmbd)
	inv := 1.0 / float64(nonZero)
	for j := range acc {
		out[j] = float32(acc[j] * inv)
	}
	return out
}

func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	n := math.Sqrt(sum)
	if n == 0 {
		return
	}
	inv := float32(1.0 / n)
	for i := range v {
		v[i] *= inv
	}
}

func allZero(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

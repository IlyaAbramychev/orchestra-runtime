package service

import (
	"context"
	"fmt"

	"github.com/operium/orchestra-runtime/internal/engine"
)

// EmbeddingService wraps the engine for embedding requests. Shares the same
// semaphore as InferenceService so we don't double-book the model.
type EmbeddingService struct {
	engine    *engine.Engine
	inference *InferenceService // shares the queue with chat requests
}

func NewEmbeddingService(eng *engine.Engine, inf *InferenceService) *EmbeddingService {
	return &EmbeddingService{engine: eng, inference: inf}
}

// Embed computes vectors for one or more inputs using the currently loaded
// model. Loops in Go — llama.cpp can batch multi-sequence but it complicates
// pooling; batching is a follow-up optimisation.
func (s *EmbeddingService) Embed(
	ctx context.Context,
	inputs []string,
	normalize bool,
) ([]*engine.EmbeddingResult, error) {
	if !s.engine.IsLoaded() {
		return nil, fmt.Errorf("no model loaded")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("inputs is required")
	}

	// Take the semaphore once for the whole batch so nobody else jumps the
	// queue mid-embedding. Ensures KV clear + decode per input are atomic.
	select {
	case s.inference.sem <- struct{}{}:
		defer func() { <-s.inference.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	out := make([]*engine.EmbeddingResult, 0, len(inputs))
	for _, in := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		res, err := s.engine.Embed(ctx, in, normalize)
		if err != nil {
			return nil, err
		}
		out = append(out, res)
	}
	return out, nil
}

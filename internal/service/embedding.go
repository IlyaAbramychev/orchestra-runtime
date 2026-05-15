package service

import (
	"context"
	"fmt"

	"github.com/operium/orchestra-runtime/internal/engine"
)

// EmbeddingService wraps the engine for embedding requests. Shares the same
// semaphore as InferenceService so we don't double-book the model.
type EmbeddingService struct {
	engine    engine.Backend
	scheduler *RuntimeScheduler
	loader    ModelLoader
}

func NewEmbeddingService(eng engine.Backend, inf *InferenceService) *EmbeddingService {
	return NewEmbeddingServiceWithScheduler(inf.scheduler)
}

func NewEmbeddingServiceWithScheduler(scheduler *RuntimeScheduler) *EmbeddingService {
	return &EmbeddingService{engine: scheduler.Backend(), scheduler: scheduler}
}

func (s *EmbeddingService) SetModelLoader(loader ModelLoader) {
	s.loader = loader
}

func (s *EmbeddingService) ensureLoaded(ctx context.Context, model string) error {
	if s.loader != nil && model != "" {
		return s.loader.EnsureLoaded(ctx, model)
	}
	if !s.engine.IsLoaded() {
		return fmt.Errorf("no model loaded")
	}
	return nil
}

// Embed computes vectors for one or more inputs using the currently loaded
// model. Loops in Go — llama.cpp can batch multi-sequence but it complicates
// pooling; batching is a follow-up optimisation.
func (s *EmbeddingService) Embed(
	ctx context.Context,
	inputs []string,
	normalize bool,
) ([]*engine.EmbeddingResult, error) {
	return s.EmbedForModel(ctx, "", inputs, normalize)
}

// EmbedForModel auto-loads model when a request supplies a model reference.
func (s *EmbeddingService) EmbedForModel(
	ctx context.Context,
	model string,
	inputs []string,
	normalize bool,
) ([]*engine.EmbeddingResult, error) {
	if err := s.ensureLoaded(ctx, model); err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("inputs is required")
	}

	// Take the shared inference slot once for the whole batch so nobody else
	// jumps the queue mid-embedding. Ensures KV clear + decode per input are
	// atomic.
	release, err := s.scheduler.acquireFor(ctx, engine.StateGenerating, s.engine.LoadedModelID())
	if err != nil {
		return nil, err
	}
	defer release()

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

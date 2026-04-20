package service

import (
	"context"
	"fmt"
	"sync"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
)

// InferenceService handles chat completion requests with queue management.
type InferenceService struct {
	engine   *engine.Engine
	sem      chan struct{}
	mu       sync.Mutex
	queueLen int
}

func NewInferenceService(eng *engine.Engine, maxQueue int) *InferenceService {
	if maxQueue <= 0 {
		maxQueue = 1
	}
	return &InferenceService{
		engine: eng,
		sem:    make(chan struct{}, 1), // 1 concurrent inference
	}
}

// QueueDepth returns the number of waiting requests.
func (s *InferenceService) QueueDepth() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queueLen
}

// Complete runs a non-streaming chat completion.
func (s *InferenceService) Complete(ctx context.Context, req *model.ChatCompletionRequest) (*engine.CompletionResult, error) {
	if !s.engine.IsLoaded() {
		return nil, fmt.Errorf("no model loaded")
	}

	s.mu.Lock()
	s.queueLen++
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.queueLen--
		s.mu.Unlock()
	}()

	// Wait for semaphore
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	msgs := toEngineMessages(req.Messages)
	params := toEngineParams(req)

	return s.engine.Complete(ctx, msgs, params)
}

// CompleteStream runs a streaming chat completion.
func (s *InferenceService) CompleteStream(ctx context.Context, req *model.ChatCompletionRequest) (<-chan engine.CompletionChunk, error) {
	if !s.engine.IsLoaded() {
		return nil, fmt.Errorf("no model loaded")
	}

	s.mu.Lock()
	s.queueLen++
	s.mu.Unlock()

	// Wait for semaphore
	select {
	case s.sem <- struct{}{}:
	case <-ctx.Done():
		s.mu.Lock()
		s.queueLen--
		s.mu.Unlock()
		return nil, ctx.Err()
	}

	msgs := toEngineMessages(req.Messages)
	params := toEngineParams(req)

	ch, err := s.engine.CompleteStream(ctx, msgs, params)
	if err != nil {
		<-s.sem
		s.mu.Lock()
		s.queueLen--
		s.mu.Unlock()
		return nil, err
	}

	// Wrap channel to release semaphore on completion
	out := make(chan engine.CompletionChunk, 32)
	go func() {
		defer close(out)
		defer func() { <-s.sem }()
		defer func() {
			s.mu.Lock()
			s.queueLen--
			s.mu.Unlock()
		}()
		for chunk := range ch {
			out <- chunk
		}
	}()

	return out, nil
}

func toEngineMessages(msgs []model.ChatMessage) []engine.ChatMessage {
	result := make([]engine.ChatMessage, len(msgs))
	for i, m := range msgs {
		result[i] = engine.ChatMessage{Role: m.Role, Content: m.Content}
	}
	return result
}

func toEngineParams(req *model.ChatCompletionRequest) engine.CompletionParams {
	params := engine.DefaultCompletionParams()
	if req.MaxTokens != nil {
		params.MaxTokens = *req.MaxTokens
	}
	if req.Temperature != nil {
		params.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = float32(*req.TopP)
	}
	if req.TopK != nil {
		params.TopK = *req.TopK
	}
	if len(req.Stop) > 0 {
		params.Stop = req.Stop
	}
	return params
}

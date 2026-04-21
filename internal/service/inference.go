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

// ApplyKeepAlive forwards a per-request keep_alive hint to the engine.
func (s *InferenceService) ApplyKeepAlive(seconds *int64) {
	s.engine.ApplyKeepAlive(seconds)
}

// Generate runs /api/generate-style completion: raw prompt in, raw text out.
// If `system` is non-empty, it is prepended as an extra "system" turn and a
// chat template is applied; otherwise the prompt is passed to the model
// verbatim (RawPrompt=true).
func (s *InferenceService) Generate(
	ctx context.Context,
	prompt, system string,
	params engine.CompletionParams,
) (*engine.CompletionResult, error) {
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

	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	msgs := buildGenerateMessages(prompt, system, &params)
	return s.engine.Complete(ctx, msgs, params)
}

// GenerateStream is the streaming twin of Generate.
func (s *InferenceService) GenerateStream(
	ctx context.Context,
	prompt, system string,
	params engine.CompletionParams,
) (<-chan engine.CompletionChunk, error) {
	if !s.engine.IsLoaded() {
		return nil, fmt.Errorf("no model loaded")
	}
	s.mu.Lock()
	s.queueLen++
	s.mu.Unlock()

	select {
	case s.sem <- struct{}{}:
	case <-ctx.Done():
		s.mu.Lock()
		s.queueLen--
		s.mu.Unlock()
		return nil, ctx.Err()
	}

	msgs := buildGenerateMessages(prompt, system, &params)
	ch, err := s.engine.CompleteStream(ctx, msgs, params)
	if err != nil {
		<-s.sem
		s.mu.Lock()
		s.queueLen--
		s.mu.Unlock()
		return nil, err
	}

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

// buildGenerateMessages chooses between raw-prompt and chat-template modes.
// If `system` is set, we apply the chat template so per-model formatting
// works; otherwise we pass the prompt through untouched (RawPrompt=true).
func buildGenerateMessages(prompt, system string, params *engine.CompletionParams) []engine.ChatMessage {
	if system == "" {
		params.RawPrompt = true
		return []engine.ChatMessage{{Role: "user", Content: prompt}}
	}
	params.RawPrompt = false
	return []engine.ChatMessage{
		{Role: "system", Content: system},
		{Role: "user", Content: prompt},
	}
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
	// Ollama's num_predict is an alias for OpenAI's max_tokens — honour both,
	// with num_predict winning if both appear.
	if req.MaxTokens != nil {
		params.MaxTokens = *req.MaxTokens
	}
	if req.NumPredict != nil {
		params.MaxTokens = *req.NumPredict
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
	if req.MinP != nil {
		params.MinP = float32(*req.MinP)
	}
	if req.TypicalP != nil {
		params.TypicalP = float32(*req.TypicalP)
	}
	if req.RepeatPenalty != nil {
		params.RepeatPenalty = float32(*req.RepeatPenalty)
	}
	if req.RepeatLastN != nil {
		params.RepeatLastN = *req.RepeatLastN
	}
	if req.FrequencyPenalty != nil {
		params.FrequencyPenalty = float32(*req.FrequencyPenalty)
	}
	if req.PresencePenalty != nil {
		params.PresencePenalty = float32(*req.PresencePenalty)
	}
	if req.Seed != nil {
		params.Seed = *req.Seed
	}
	if req.Mirostat != nil {
		params.Mirostat = *req.Mirostat
	}
	if req.MirostatTau != nil {
		params.MirostatTau = float32(*req.MirostatTau)
	}
	if req.MirostatEta != nil {
		params.MirostatEta = float32(*req.MirostatEta)
	}
	if len(req.Stop) > 0 {
		params.Stop = req.Stop
	}
	return params
}

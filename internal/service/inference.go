package service

import (
	"context"
	"fmt"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
)

// InferenceService handles chat completion requests with queue management.
// Backend is either an in-process *engine.Engine or a subprocess proxy
// (*supervisor.Remote) — chosen in server.Start based on
// ORCHESTRA_USE_SUBPROCESS.
type InferenceService struct {
	scheduler *RuntimeScheduler
	engine    engine.Backend
	loader    ModelLoader
}

type ModelRequestDefaults struct {
	StopTokens   []string
	ChatTemplate string
}

// ModelLoader resolves and loads a model for request-scoped auto-load.
type ModelLoader interface {
	EnsureLoaded(ctx context.Context, model string) error
	EnsureLoadedFor(ctx context.Context, model, capability string) error
	DefaultsForModel(model string) (ModelRequestDefaults, error)
}

func NewInferenceService(eng engine.Backend, maxQueue int) *InferenceService {
	return NewInferenceServiceWithScheduler(NewRuntimeScheduler(eng, maxQueue))
}

func NewInferenceServiceWithScheduler(scheduler *RuntimeScheduler) *InferenceService {
	return &InferenceService{
		scheduler: scheduler,
		engine:    scheduler.Backend(),
	}
}

func (s *InferenceService) SetModelLoader(loader ModelLoader) {
	s.loader = loader
}

func (s *InferenceService) ensureLoaded(ctx context.Context, model string) error {
	if s.loader != nil && model != "" {
		return s.loader.EnsureLoadedFor(ctx, model, "chat")
	}
	if !s.engine.IsLoaded() {
		return fmt.Errorf("no model loaded")
	}
	return nil
}

func (s *InferenceService) applyModelDefaults(model string, params *engine.CompletionParams) {
	if s.loader == nil || model == "" {
		return
	}
	defaults, err := s.loader.DefaultsForModel(model)
	if err != nil {
		return
	}
	if len(params.Stop) == 0 && len(defaults.StopTokens) > 0 {
		params.Stop = append([]string(nil), defaults.StopTokens...)
	}
	if params.ChatTemplate == "" && defaults.ChatTemplate != "" {
		params.ChatTemplate = defaults.ChatTemplate
	}
}

// QueueDepth returns the number of waiting requests.
func (s *InferenceService) QueueDepth() int {
	return s.scheduler.QueueDepth()
}

// ApplyKeepAlive forwards a per-request keep_alive hint to the engine.
func (s *InferenceService) ApplyKeepAlive(seconds *int64) {
	s.scheduler.ApplyKeepAlive(seconds)
}

func forwardCompletionChunks(
	ctx context.Context,
	ch <-chan engine.CompletionChunk,
	release func(),
) <-chan engine.CompletionChunk {
	out := make(chan engine.CompletionChunk, 32)
	go func() {
		defer close(out)
		defer release()
		for {
			select {
			case chunk, ok := <-ch:
				if !ok {
					return
				}
				select {
				case out <- chunk:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

// Generate runs /api/generate-style completion: raw prompt in, raw text out.
// If `system` is non-empty, it is prepended as an extra "system" turn and a
// chat template is applied; otherwise the prompt is passed to the model
// verbatim (RawPrompt=true).
func (s *InferenceService) Generate(
	ctx context.Context,
	model string,
	prompt, system string,
	params engine.CompletionParams,
) (*engine.CompletionResult, error) {
	if err := s.ensureLoaded(ctx, model); err != nil {
		return nil, err
	}
	s.applyModelDefaults(model, &params)
	release, err := s.scheduler.acquireFor(ctx, engine.StateGenerating, s.engine.LoadedModelID())
	if err != nil {
		return nil, err
	}
	defer release()

	msgs := buildGenerateMessages(prompt, system, &params)
	return s.engine.Complete(ctx, msgs, params)
}

// GenerateStream is the streaming twin of Generate.
func (s *InferenceService) GenerateStream(
	ctx context.Context,
	model string,
	prompt, system string,
	params engine.CompletionParams,
) (<-chan engine.CompletionChunk, error) {
	if err := s.ensureLoaded(ctx, model); err != nil {
		return nil, err
	}
	s.applyModelDefaults(model, &params)
	release, err := s.scheduler.acquireFor(ctx, engine.StateGenerating, s.engine.LoadedModelID())
	if err != nil {
		return nil, err
	}

	msgs := buildGenerateMessages(prompt, system, &params)
	ch, err := s.engine.CompleteStream(ctx, msgs, params)
	if err != nil {
		release()
		return nil, err
	}

	return forwardCompletionChunks(ctx, ch, release), nil
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
	if err := s.ensureLoaded(ctx, req.Model); err != nil {
		return nil, err
	}

	release, err := s.scheduler.acquireFor(ctx, engine.StateGenerating, s.engine.LoadedModelID())
	if err != nil {
		return nil, err
	}
	defer release()

	msgs := toEngineMessages(req.Messages)
	params := toEngineParams(req)
	s.applyModelDefaults(req.Model, &params)

	return s.engine.Complete(ctx, msgs, params)
}

// CompleteStream runs a streaming chat completion.
func (s *InferenceService) CompleteStream(ctx context.Context, req *model.ChatCompletionRequest) (<-chan engine.CompletionChunk, error) {
	if err := s.ensureLoaded(ctx, req.Model); err != nil {
		return nil, err
	}

	release, err := s.scheduler.acquireFor(ctx, engine.StateGenerating, s.engine.LoadedModelID())
	if err != nil {
		return nil, err
	}

	msgs := toEngineMessages(req.Messages)
	params := toEngineParams(req)
	s.applyModelDefaults(req.Model, &params)

	ch, err := s.engine.CompleteStream(ctx, msgs, params)
	if err != nil {
		release()
		return nil, err
	}

	return forwardCompletionChunks(ctx, ch, release), nil
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

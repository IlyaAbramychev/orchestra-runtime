package service

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

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
	EnsureLoadedForCapabilities(ctx context.Context, model string, capabilities []string) error
	ResolveModelID(model string) (string, error)
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

// acquireLoadedModel closes the race between request-scoped auto-load and
// inference. Auto-load uses the scheduler itself, so it must finish before we
// acquire the inference slot. Another queued request may switch the model in
// that small gap; after acquiring, verify the resolved registry ID and retry
// the load instead of ever running against the wrong model.
func acquireLoadedModel(
	ctx context.Context,
	scheduler *RuntimeScheduler,
	backend engine.Backend,
	loader ModelLoader,
	model string,
	capabilities ...string,
) (func(), error) {
	if len(capabilities) == 0 {
		capabilities = []string{"chat"}
	}
	for {
		expectedModelID := ""
		if loader != nil && model != "" {
			resolvedID, err := loader.ResolveModelID(model)
			if err != nil {
				return nil, err
			}
			expectedModelID = resolvedID
			if err := loader.EnsureLoadedForCapabilities(ctx, model, capabilities); err != nil {
				return nil, err
			}
		} else if !backend.IsLoaded() {
			return nil, fmt.Errorf("no model loaded")
		}

		activeModelID := expectedModelID
		if activeModelID == "" {
			activeModelID = backend.LoadedModelID()
		}
		release, err := scheduler.acquireFor(ctx, engine.StateGenerating, activeModelID)
		if err != nil {
			return nil, err
		}

		if !backend.IsLoaded() {
			release()
			if expectedModelID != "" {
				continue
			}
			return nil, fmt.Errorf("no model loaded")
		}
		if expectedModelID != "" && backend.LoadedModelID() != expectedModelID {
			release()
			continue
		}
		return release, nil
	}
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
	images []string,
	params engine.CompletionParams,
) (*engine.CompletionResult, error) {
	capability := "chat"
	if len(images) > 0 {
		capability = "vision"
	}
	release, err := acquireLoadedModel(ctx, s.scheduler, s.engine, s.loader, model, capability)
	if err != nil {
		return nil, err
	}
	defer release()
	s.applyModelDefaults(model, &params)

	msgs := buildGenerateMessages(prompt, system, images, &params)
	return s.engine.Complete(ctx, msgs, params)
}

// GenerateStream is the streaming twin of Generate.
func (s *InferenceService) GenerateStream(
	ctx context.Context,
	model string,
	prompt, system string,
	images []string,
	params engine.CompletionParams,
) (<-chan engine.CompletionChunk, error) {
	capability := "chat"
	if len(images) > 0 {
		capability = "vision"
	}
	release, err := acquireLoadedModel(ctx, s.scheduler, s.engine, s.loader, model, capability)
	if err != nil {
		return nil, err
	}
	s.applyModelDefaults(model, &params)

	msgs := buildGenerateMessages(prompt, system, images, &params)
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
func buildGenerateMessages(prompt, system string, images []string, params *engine.CompletionParams) []engine.ChatMessage {
	if system == "" {
		params.RawPrompt = true
		return []engine.ChatMessage{{Role: "user", Content: prompt, Images: append([]string(nil), images...)}}
	}
	params.RawPrompt = false
	return []engine.ChatMessage{
		{Role: "system", Content: system},
		{Role: "user", Content: prompt, Images: append([]string(nil), images...)},
	}
}

// Complete runs a non-streaming chat completion.
func (s *InferenceService) Complete(ctx context.Context, req *model.ChatCompletionRequest) (*engine.CompletionResult, error) {
	release, err := acquireLoadedModel(ctx, s.scheduler, s.engine, s.loader, req.Model, chatRequestCapabilities(req)...)
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
	release, err := acquireLoadedModel(ctx, s.scheduler, s.engine, s.loader, req.Model, chatRequestCapabilities(req)...)
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

func chatRequestCapabilities(req *model.ChatCompletionRequest) []string {
	capabilities := []string{"chat"}
	if len(req.Tools) > 0 {
		capabilities = append(capabilities, "tools")
	}
	if (len(req.Think) > 0 && thinkEnabled(req.Think)) ||
		(req.ReasoningEffort != "" && req.ReasoningEffort != "none") {
		capabilities = append(capabilities, "thinking")
	}
	if chatRequestHasImages(req.Messages) {
		capabilities = append(capabilities, "vision")
	}
	return capabilities
}

func chatRequestHasImages(messages []model.ChatMessage) bool {
	for _, message := range messages {
		if len(message.Images) > 0 {
			return true
		}
		for _, part := range message.Parts {
			if part.Type == "image_url" && strings.TrimSpace(part.ImageURL) != "" {
				return true
			}
		}
	}
	return false
}

func toEngineMessages(msgs []model.ChatMessage) []engine.ChatMessage {
	result := make([]engine.ChatMessage, len(msgs))
	for i, m := range msgs {
		toolCalls := make([]engine.ToolCall, 0, len(m.ToolCalls))
		for _, call := range m.ToolCalls {
			arguments, err := json.Marshal(call.Function.Arguments)
			if err != nil {
				arguments = []byte(`{}`)
			}
			toolCalls = append(toolCalls, engine.ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: arguments})
		}
		parts := make([]engine.ContentPart, len(m.Parts))
		for partIndex, part := range m.Parts {
			parts[partIndex] = engine.ContentPart{
				Type:        part.Type,
				Text:        part.Text,
				ImageURL:    part.ImageURL,
				ImageDetail: part.ImageDetail,
			}
		}
		result[i] = engine.ChatMessage{
			Role:       m.Role,
			Content:    m.Content,
			Reasoning:  firstNonBlank(m.ReasoningContent, m.Thinking),
			ToolName:   m.ToolName,
			ToolCallID: m.ToolCallID,
			ToolCalls:  toolCalls,
			Parts:      parts,
			Images:     append([]string(nil), m.Images...),
		}
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
	params.Grammar = req.Grammar
	params.NativeChat = len(req.Tools) > 0 || len(req.Think) > 0 || req.ReasoningEffort != "" || messagesNeedNativeChat(req.Messages)
	if len(req.Tools) > 0 {
		if encoded, err := json.Marshal(req.Tools); err == nil {
			params.ToolsJSON = string(encoded)
		}
		params.ToolChoice = nativeToolChoice(req.ToolChoice)
		params.ParallelToolCalls = req.ParallelToolCalls == nil || *req.ParallelToolCalls
	}
	if len(req.Think) > 0 {
		params.ThinkingSet = true
		params.EnableThinking = thinkEnabled(req.Think)
	} else if req.ReasoningEffort != "" {
		params.ThinkingSet = true
		params.EnableThinking = req.ReasoningEffort != "none"
	}
	return params
}

func messagesNeedNativeChat(messages []model.ChatMessage) bool {
	for _, message := range messages {
		if message.ReasoningContent != "" || message.Thinking != "" || len(message.ToolCalls) > 0 || message.ToolCallID != "" {
			return true
		}
	}
	return false
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func nativeToolChoice(raw json.RawMessage) int {
	var choice string
	if json.Unmarshal(raw, &choice) == nil {
		switch choice {
		case "required":
			return 1
		case "none":
			return 2
		}
	}
	if len(raw) > 0 {
		var forced model.OpenAIToolChoice
		if json.Unmarshal(raw, &forced) == nil && forced.Function.Name != "" {
			return 1
		}
	}
	return 0
}

func thinkEnabled(raw json.RawMessage) bool {
	var enabled bool
	if json.Unmarshal(raw, &enabled) == nil {
		return enabled
	}
	var level string
	if json.Unmarshal(raw, &level) == nil {
		return level != "none" && level != "false" && level != "off"
	}
	return true
}

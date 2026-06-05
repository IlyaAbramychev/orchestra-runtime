package handler

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

type ChatHandler struct {
	inference *service.InferenceService
}

func NewChatHandler(inference *service.InferenceService) *ChatHandler {
	return &ChatHandler{inference: inference}
}

// ChatCompletion handles POST /v1/chat/completions (OpenAI-compatible).
func (h *ChatHandler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
	var req model.ChatCompletionRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}

	if req.Stream {
		h.handleStream(w, r, &req)
		return
	}

	h.handleComplete(w, r, &req)
}

// ChatOllama handles POST /api/chat using Ollama's JSON/NDJSON response
// contract. It intentionally does not reuse the OpenAI SSE handler.
func (h *ChatHandler) ChatOllama(w http.ResponseWriter, r *http.Request) {
	var req model.OllamaChatRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}
	if hasOllamaChatImages(req.Messages) {
		writeError(w, http.StatusBadRequest, "multimodal images are not supported yet")
		return
	}
	if err := validateThinkOption(req.Think); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}
	if stream && hasMeaningfulRawJSON(req.Think) {
		writeError(w, http.StatusBadRequest, "streaming thinking output is not supported yet")
		return
	}
	if len(req.Tools) > 0 && hasMeaningfulRawJSON(req.Format) {
		writeError(w, http.StatusBadRequest, "format cannot be combined with tools")
		return
	}
	if stream && hasMeaningfulRawJSON(req.Format) {
		writeError(w, http.StatusBadRequest, "streaming structured output is not supported yet")
		return
	}
	if stream && len(req.Tools) > 0 {
		writeError(w, http.StatusBadRequest, "streaming tool calls are not supported yet")
		return
	}

	chatReq := ollamaToChatCompletionRequest(&req)
	if instruction, ok, err := structuredFormatInstruction(req.Format); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		chatReq.Messages = withStructuredInstruction(chatReq.Messages, instruction)
	}
	if instruction, ok, err := toolCallInstruction(req.Tools); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		chatReq.Messages = withStructuredInstruction(chatReq.Messages, instruction)
	}
	if stream {
		h.handleOllamaStream(w, r, &req, chatReq)
		return
	}
	h.handleOllamaComplete(w, r, &req, chatReq)
}

func (h *ChatHandler) handleOllamaComplete(
	w http.ResponseWriter,
	r *http.Request,
	req *model.OllamaChatRequest,
	chatReq *model.ChatCompletionRequest,
) {
	result, err := h.inference.Complete(r.Context(), chatReq)
	if err != nil {
		slog.Error("ollama chat failed", "error", err)
		writeRuntimeError(w, err)
		return
	}
	content, thinking := applyThinkingOutput(req.Think, result.Text)
	toolCalls, hasToolCalls, err := parseToolCallsFromText(content, req.Tools)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	if err := validateStructuredOutput(req.Format, content); err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	h.inference.ApplyKeepAlive(req.KeepAlive)
	message := model.ChatMessage{
		Role:     "assistant",
		Content:  content,
		Thinking: thinking,
	}
	doneReason := result.FinishReason
	if hasToolCalls {
		message.Content = ""
		message.ToolCalls = toolCalls
		doneReason = "tool_calls"
	}

	resp := model.OllamaChatResponse{
		Model:                req.Model,
		CreatedAt:            time.Now().UTC().Format(time.RFC3339Nano),
		Message:              message,
		Done:                 true,
		TotalDurationNs:      result.Timings.TotalNs,
		PromptEvalDurationNs: result.Timings.PromptEvalNs,
		PromptEvalCount:      result.PromptTokens,
		EvalDurationNs:       result.Timings.EvalNs,
		EvalCount:            result.CompletionTokens,
		DoneReason:           doneReason,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *ChatHandler) handleOllamaStream(
	w http.ResponseWriter,
	r *http.Request,
	req *model.OllamaChatRequest,
	chatReq *model.ChatCompletionRequest,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	ch, err := h.inference.CompleteStream(r.Context(), chatReq)
	if err != nil {
		slog.Error("ollama chat stream failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	for chunk := range ch {
		if chunk.Err != nil {
			slog.Error("ollama chat chunk error", "error", chunk.Err)
			resp := model.OllamaChatResponse{
				Model:      req.Model,
				CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
				Done:       true,
				DoneReason: "error",
				Error:      chunk.Err.Error(),
			}
			if data, err := json.Marshal(resp); err == nil {
				fmt.Fprintf(w, "%s\n", data)
				flusher.Flush()
			}
			h.inference.ApplyKeepAlive(req.KeepAlive)
			return
		}

		resp := model.OllamaChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		}
		if chunk.Done {
			resp.Done = true
			resp.DoneReason = chunk.FinishReason
			resp.TotalDurationNs = chunk.Timings.TotalNs
			resp.PromptEvalDurationNs = chunk.Timings.PromptEvalNs
			resp.PromptEvalCount = chunk.PromptTokens
			resp.EvalDurationNs = chunk.Timings.EvalNs
			resp.EvalCount = chunk.CompletionTokens
		} else {
			resp.Message = model.ChatMessage{Role: "assistant", Content: chunk.Text}
		}

		if data, err := json.Marshal(resp); err == nil {
			fmt.Fprintf(w, "%s\n", data)
			flusher.Flush()
		}
		if chunk.Done {
			break
		}
	}

	h.inference.ApplyKeepAlive(req.KeepAlive)
}

func (h *ChatHandler) handleComplete(w http.ResponseWriter, r *http.Request, req *model.ChatCompletionRequest) {
	result, err := h.inference.Complete(r.Context(), req)
	if err != nil {
		slog.Error("completion failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	// Apply per-request keep_alive after generation — clients expect immediate
	// response; unload happens on the next engine tick.
	h.inference.ApplyKeepAlive(req.KeepAlive)

	resp := model.ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:8],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []model.Choice{
			{
				Index: 0,
				Message: &model.ChatMessage{
					Role:    "assistant",
					Content: result.Text,
				},
				FinishReason: &result.FinishReason,
			},
		},
		Usage: &model.Usage{
			PromptTokens:     result.PromptTokens,
			CompletionTokens: result.CompletionTokens,
			TotalTokens:      result.PromptTokens + result.CompletionTokens,
		},
		Timings: &model.Timings{
			TotalDurationNs:      result.Timings.TotalNs,
			PromptEvalDurationNs: result.Timings.PromptEvalNs,
			PromptEvalCount:      result.PromptTokens,
			EvalDurationNs:       result.Timings.EvalNs,
			EvalCount:            result.CompletionTokens,
		},
	}

	writeJSON(w, http.StatusOK, resp)
}

func ollamaToChatCompletionRequest(req *model.OllamaChatRequest) *model.ChatCompletionRequest {
	out := &model.ChatCompletionRequest{
		Model:     req.Model,
		Messages:  req.Messages,
		KeepAlive: req.KeepAlive,
	}
	if req.Options == nil {
		return out
	}
	o := req.Options
	out.Temperature = o.Temperature
	out.NumPredict = o.NumPredict
	out.TopP = o.TopP
	out.TopK = o.TopK
	out.MinP = o.MinP
	out.TypicalP = o.TypicalP
	out.RepeatPenalty = o.RepeatPenalty
	out.RepeatLastN = o.RepeatLastN
	out.FrequencyPenalty = o.FrequencyPenalty
	out.PresencePenalty = o.PresencePenalty
	out.Seed = o.Seed
	out.Mirostat = o.Mirostat
	out.MirostatTau = o.MirostatTau
	out.MirostatEta = o.MirostatEta
	out.Stop = o.Stop
	return out
}

func hasOllamaChatImages(messages []model.ChatMessage) bool {
	for _, msg := range messages {
		if len(msg.Images) > 0 {
			return true
		}
	}
	return false
}

func (h *ChatHandler) handleStream(w http.ResponseWriter, r *http.Request, req *model.ChatCompletionRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	ch, err := h.inference.CompleteStream(r.Context(), req)
	if err != nil {
		slog.Error("stream completion failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	id := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	for chunk := range ch {
		if chunk.Err != nil {
			slog.Error("stream chunk error", "error", chunk.Err)
			errPayload := map[string]any{
				"error": map[string]any{
					"message": chunk.Err.Error(),
					"type":    "runtime_stream_error",
				},
			}
			data, _ := json.Marshal(errPayload)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			h.inference.ApplyKeepAlive(req.KeepAlive)
			return
		}

		var sseChunk model.ChatCompletionChunk

		if chunk.Done {
			sseChunk = model.ChatCompletionChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []model.ChunkChoice{
					{
						Index:        0,
						Delta:        &model.ChatMessage{},
						FinishReason: &chunk.FinishReason,
					},
				},
				Usage: &model.Usage{
					PromptTokens:     chunk.PromptTokens,
					CompletionTokens: chunk.CompletionTokens,
					TotalTokens:      chunk.PromptTokens + chunk.CompletionTokens,
				},
				Timings: &model.Timings{
					TotalDurationNs:      chunk.Timings.TotalNs,
					PromptEvalDurationNs: chunk.Timings.PromptEvalNs,
					PromptEvalCount:      chunk.PromptTokens,
					EvalDurationNs:       chunk.Timings.EvalNs,
					EvalCount:            chunk.CompletionTokens,
				},
			}
		} else {
			sseChunk = model.ChatCompletionChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []model.ChunkChoice{
					{
						Index: 0,
						Delta: &model.ChatMessage{
							Role:    "assistant",
							Content: chunk.Text,
						},
					},
				},
			}
		}

		data, _ := json.Marshal(sseChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		if chunk.Done {
			break
		}
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	// Apply per-request keep_alive after the stream closes.
	h.inference.ApplyKeepAlive(req.KeepAlive)
}

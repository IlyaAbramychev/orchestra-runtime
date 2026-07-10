package handler

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/engine"
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
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}
	if err := validateMultimodalMessages(req.Messages); err != nil {
		writeRuntimeError(w, err)
		return
	}
	toolMode, err := applyOpenAIToolInstructions(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Stream {
		if toolMode.active {
			h.handleOpenAIToolStream(w, r, &req, toolMode)
			return
		}
		h.handleStream(w, r, &req)
		return
	}

	h.handleComplete(w, r, &req, toolMode)
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
	if err := validateMultimodalMessages(req.Messages); err != nil {
		writeRuntimeError(w, err)
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
	if len(req.Tools) > 0 && hasMeaningfulRawJSON(req.Format) {
		writeError(w, http.StatusBadRequest, "format cannot be combined with tools")
		return
	}

	chatReq := ollamaToChatCompletionRequest(&req)
	if instruction, ok, err := structuredFormatInstruction(req.Format); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		chatReq.Messages = withStructuredInstruction(chatReq.Messages, instruction)
	}
	if grammar, ok, err := structuredFormatGrammar(req.Format); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		chatReq.Grammar = grammar
	}
	if _, _, err := toolCallInstruction(req.Tools); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if stream {
		if shouldBufferOllamaChatStream(&req) {
			h.handleOllamaBufferedStream(w, r, &req, chatReq)
			return
		}
		h.handleOllamaStream(w, r, &req, chatReq)
		return
	}
	h.handleOllamaComplete(w, r, &req, chatReq)
}

func shouldBufferOllamaChatStream(req *model.OllamaChatRequest) bool {
	return hasMeaningfulRawJSON(req.Format) || hasMeaningfulRawJSON(req.Think) || len(req.Tools) > 0
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
	content, thinking := result.Text, result.Reasoning
	if result.FinishReason == "tool_protocol_error" {
		content = ""
	}
	if thinking == "" {
		content, thinking = applyThinkingOutput(req.Think, result.Text)
	}
	toolCalls, hasToolCalls, err := resolvedToolCalls(result.ToolCalls, content, req.Tools)
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

func (h *ChatHandler) handleOllamaBufferedStream(
	w http.ResponseWriter,
	r *http.Request,
	req *model.OllamaChatRequest,
	chatReq *model.ChatCompletionRequest,
) {
	ch, err := h.inference.CompleteStream(r.Context(), chatReq)
	if err != nil {
		slog.Error("ollama buffered chat stream failed", "error", err)
		writeRuntimeError(w, err)
		return
	}
	text, final, err := collectCompletionStream(ch)
	if err != nil {
		slog.Error("ollama buffered chat stream chunk error", "error", err)
		writeRuntimeError(w, err)
		h.inference.ApplyKeepAlive(req.KeepAlive)
		return
	}

	content, thinking := text, final.Reasoning
	if final.FinishReason == "tool_protocol_error" {
		content = ""
	}
	if thinking == "" {
		content, thinking = applyThinkingOutput(req.Think, text)
	}
	toolCalls, hasToolCalls, err := resolvedToolCalls(final.ToolCalls, content, req.Tools)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		h.inference.ApplyKeepAlive(req.KeepAlive)
		return
	}
	if err := validateStructuredOutput(req.Format, content); err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		h.inference.ApplyKeepAlive(req.KeepAlive)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		h.inference.ApplyKeepAlive(req.KeepAlive)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	createdAt := time.Now().UTC().Format(time.RFC3339Nano)
	if hasToolCalls {
		writeOllamaChatStreamResponse(w, flusher, model.OllamaChatResponse{
			Model:     req.Model,
			CreatedAt: createdAt,
			Message: model.ChatMessage{
				Role:      "assistant",
				Content:   "",
				ToolCalls: toolCalls,
			},
		})
	} else {
		if thinking != "" {
			writeOllamaChatStreamResponse(w, flusher, model.OllamaChatResponse{
				Model:     req.Model,
				CreatedAt: createdAt,
				Message:   model.ChatMessage{Role: "assistant", Thinking: thinking},
			})
		}
		if content != "" {
			writeOllamaChatStreamResponse(w, flusher, model.OllamaChatResponse{
				Model:     req.Model,
				CreatedAt: createdAt,
				Message:   model.ChatMessage{Role: "assistant", Content: content},
			})
		}
	}

	doneReason := final.FinishReason
	if hasToolCalls {
		doneReason = "tool_calls"
	}
	writeOllamaChatStreamResponse(w, flusher, model.OllamaChatResponse{
		Model:                req.Model,
		CreatedAt:            createdAt,
		Done:                 true,
		DoneReason:           doneReason,
		TotalDurationNs:      final.Timings.TotalNs,
		PromptEvalDurationNs: final.Timings.PromptEvalNs,
		PromptEvalCount:      final.PromptTokens,
		EvalDurationNs:       final.Timings.EvalNs,
		EvalCount:            final.CompletionTokens,
	})
	h.inference.ApplyKeepAlive(req.KeepAlive)
}

func writeOllamaChatStreamResponse(w http.ResponseWriter, flusher http.Flusher, resp model.OllamaChatResponse) {
	if data, err := json.Marshal(resp); err == nil {
		fmt.Fprintf(w, "%s\n", data)
		flusher.Flush()
	}
}

func (h *ChatHandler) handleComplete(
	w http.ResponseWriter,
	r *http.Request,
	req *model.ChatCompletionRequest,
	toolMode openAIToolMode,
) {
	result, err := h.inference.Complete(r.Context(), req)
	if err != nil {
		slog.Error("completion failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	// Apply per-request keep_alive after generation — clients expect immediate
	// response; unload happens on the next engine tick.
	h.inference.ApplyKeepAlive(req.KeepAlive)

	if toolMode.active {
		toolCalls, hasToolCalls, parseErr := resolvedToolCalls(result.ToolCalls, result.Text, req.Tools)
		if parseErr != nil {
			writeOpenAIToolProtocolError(w, req, result, parseErr)
			return
		}
		if validationErr := validateOpenAIToolResult(toolMode, toolCalls, hasToolCalls); validationErr != nil {
			writeOpenAIToolProtocolError(w, req, result, validationErr)
			return
		}
		if result.FinishReason == "tool_protocol_error" {
			writeOpenAIToolProtocolError(w, req, result, fmt.Errorf("model returned a malformed tool call"))
			return
		}
		if hasToolCalls {
			finishReason := "tool_calls"
			writeJSON(w, http.StatusOK, model.OpenAIChatCompletionResponse{
				ID:      "chatcmpl-" + uuid.New().String()[:8],
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   req.Model,
				Choices: []model.OpenAIChoice{{
					Index: 0,
					Message: &model.OpenAIResponseMessage{
						Role:      "assistant",
						Content:   nil,
						ToolCalls: toOpenAIToolCalls(toolCalls, false),
					},
					FinishReason: &finishReason,
				}},
				Usage:   completionUsage(result.PromptTokens, result.CompletionTokens, result.TextPromptTokens, result.VisionTokens),
				Timings: completionTimings(result),
			})
			return
		}
	}

	resp := model.ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:8],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []model.Choice{
			{
				Index: 0,
				Message: &model.ChatMessage{
					Role:             "assistant",
					Content:          result.Text,
					ReasoningContent: result.Reasoning,
				},
				FinishReason: &result.FinishReason,
			},
		},
		Usage:   completionUsage(result.PromptTokens, result.CompletionTokens, result.TextPromptTokens, result.VisionTokens),
		Timings: completionTimings(result),
	}

	writeJSON(w, http.StatusOK, resp)
}

func completionTimings(result *engine.CompletionResult) *model.Timings {
	return &model.Timings{
		TotalDurationNs:      result.Timings.TotalNs,
		PromptEvalDurationNs: result.Timings.PromptEvalNs,
		PromptEvalCount:      result.PromptTokens,
		EvalDurationNs:       result.Timings.EvalNs,
		EvalCount:            result.CompletionTokens,
	}
}

func writeOpenAIToolProtocolError(w http.ResponseWriter, req *model.ChatCompletionRequest, result *engine.CompletionResult, cause error) {
	slog.Warn("model returned invalid tool protocol", "model", req.Model, "error", cause)
	finishReason := "tool_protocol_error"
	writeJSON(w, http.StatusOK, model.OpenAIChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:8],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []model.OpenAIChoice{{
			Index: 0,
			Message: &model.OpenAIResponseMessage{
				Role:             "assistant",
				Content:          nil,
				ReasoningContent: result.Reasoning,
			},
			FinishReason: &finishReason,
		}},
		Usage:   completionUsage(result.PromptTokens, result.CompletionTokens, result.TextPromptTokens, result.VisionTokens),
		Timings: completionTimings(result),
	})
}

func completionUsage(promptTokens, completionTokens, textPromptTokens, visionTokens int) *model.Usage {
	usage := &model.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}
	if textPromptTokens > 0 || visionTokens > 0 {
		usage.PromptTokensDetails = &model.PromptTokensDetails{
			TextTokens:   textPromptTokens,
			VisionTokens: visionTokens,
		}
	}
	return usage
}

func (h *ChatHandler) handleOpenAIToolStream(
	w http.ResponseWriter,
	r *http.Request,
	req *model.ChatCompletionRequest,
	toolMode openAIToolMode,
) {
	ch, err := h.inference.CompleteStream(r.Context(), req)
	if err != nil {
		slog.Error("OpenAI tool stream failed", "error", err)
		writeRuntimeError(w, err)
		return
	}
	text, final, err := collectCompletionStream(ch)
	h.inference.ApplyKeepAlive(req.KeepAlive)
	if err != nil {
		slog.Error("OpenAI tool stream chunk failed", "error", err)
		writeRuntimeError(w, err)
		return
	}
	toolCalls, hasToolCalls, err := resolvedToolCalls(final.ToolCalls, text, req.Tools)
	if err != nil {
		final.FinishReason = "tool_protocol_error"
		toolCalls = nil
		hasToolCalls = false
	}
	if validationErr := validateOpenAIToolResult(toolMode, toolCalls, hasToolCalls); validationErr != nil {
		final.FinishReason = "tool_protocol_error"
		toolCalls = nil
		hasToolCalls = false
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	id := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()
	if hasToolCalls {
		writeOpenAISSE(w, flusher, model.OpenAIChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []model.OpenAIChunkChoice{{
				Index: 0,
				Delta: &model.OpenAIResponseDelta{
					Role:      "assistant",
					ToolCalls: toOpenAIToolCalls(toolCalls, true),
				},
			}},
		})
	} else if text != "" {
		content := text
		writeOpenAISSE(w, flusher, model.OpenAIChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []model.OpenAIChunkChoice{{
				Index: 0,
				Delta: &model.OpenAIResponseDelta{Role: "assistant", Content: &content},
			}},
		})
	}
	if final.Reasoning != "" {
		writeOpenAISSE(w, flusher, model.OpenAIChatCompletionChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
			Choices: []model.OpenAIChunkChoice{{Index: 0, Delta: &model.OpenAIResponseDelta{Role: "assistant", ReasoningContent: final.Reasoning}}},
		})
	}

	finishReason := final.FinishReason
	if hasToolCalls {
		finishReason = "tool_calls"
	}
	writeOpenAISSE(w, flusher, model.OpenAIChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   req.Model,
		Choices: []model.OpenAIChunkChoice{{
			Index:        0,
			Delta:        &model.OpenAIResponseDelta{},
			FinishReason: &finishReason,
		}},
		Usage: completionUsage(final.PromptTokens, final.CompletionTokens, final.TextPromptTokens, final.VisionTokens),
		Timings: &model.Timings{
			TotalDurationNs:      final.Timings.TotalNs,
			PromptEvalDurationNs: final.Timings.PromptEvalNs,
			PromptEvalCount:      final.PromptTokens,
			EvalDurationNs:       final.Timings.EvalNs,
			EvalCount:            final.CompletionTokens,
		},
	})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAISSE(w http.ResponseWriter, flusher http.Flusher, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func ollamaToChatCompletionRequest(req *model.OllamaChatRequest) *model.ChatCompletionRequest {
	out := &model.ChatCompletionRequest{
		Model:     req.Model,
		Messages:  req.Messages,
		Tools:     req.Tools,
		Think:     req.Think,
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
				Usage: completionUsage(chunk.PromptTokens, chunk.CompletionTokens, chunk.TextPromptTokens, chunk.VisionTokens),
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

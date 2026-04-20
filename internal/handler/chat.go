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

func (h *ChatHandler) handleComplete(w http.ResponseWriter, r *http.Request, req *model.ChatCompletionRequest) {
	result, err := h.inference.Complete(r.Context(), req)
	if err != nil {
		slog.Error("completion failed", "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
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
	}

	writeJSON(w, http.StatusOK, resp)
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
		writeError(w, http.StatusInternalServerError, err.Error())
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
			break
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
}

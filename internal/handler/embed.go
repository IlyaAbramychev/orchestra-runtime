package handler

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

type EmbedHandler struct {
	embedding *service.EmbeddingService
	inference *service.InferenceService
}

func NewEmbedHandler(emb *service.EmbeddingService, inf *service.InferenceService) *EmbedHandler {
	return &EmbedHandler{embedding: emb, inference: inf}
}

// Embed serves POST /api/embed (Ollama-compat).
func (h *EmbedHandler) Embed(w http.ResponseWriter, r *http.Request) {
	var req model.EmbedRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	inputs, err := parseInputs(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	start := time.Now()
	// Ollama clients expect pre-normalised vectors for cosine similarity.
	results, err := h.embedding.Embed(r.Context(), inputs, true)
	if err != nil {
		slog.Error("embed failed", "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	h.inference.ApplyKeepAlive(req.KeepAlive)

	embeddings := make([][]float32, len(results))
	total := 0
	for i, r := range results {
		embeddings[i] = r.Vector
		total += r.PromptTokens
	}

	resp := model.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		PromptEvalCount: total,
		TotalDurationNs: time.Since(start).Nanoseconds(),
	}
	writeJSON(w, http.StatusOK, resp)
}

// EmbedOpenAI serves POST /v1/embeddings (OpenAI-compat).
func (h *EmbedHandler) EmbedOpenAI(w http.ResponseWriter, r *http.Request) {
	var req model.OpenAIEmbeddingsRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	inputs, err := parseInputs(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	results, err := h.embedding.Embed(r.Context(), inputs, true)
	if err != nil {
		slog.Error("embed failed", "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	data := make([]model.OpenAIEmbeddingRecord, len(results))
	total := 0
	for i, res := range results {
		data[i] = model.OpenAIEmbeddingRecord{
			Object:    "embedding",
			Index:     i,
			Embedding: res.Vector,
		}
		total += res.PromptTokens
	}

	writeJSON(w, http.StatusOK, model.OpenAIEmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  req.Model,
		Usage: model.OpenAIEmbeddingsUsage{
			PromptTokens: total,
			TotalTokens:  total,
		},
	})
}

// parseInputs accepts either a single string or a JSON array of strings, and
// normalises to []string. Both Ollama and OpenAI APIs use that union type.
func parseInputs(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, &badRequestErr{"input is required"}
	}
	// Try array first — more common for batch embedding.
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		if len(arr) == 0 {
			return nil, &badRequestErr{"input array is empty"}
		}
		return arr, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, &badRequestErr{"input must be string or []string"}
	}
	if s == "" {
		return nil, &badRequestErr{"input string is empty"}
	}
	return []string{s}, nil
}

type badRequestErr struct{ msg string }

func (e *badRequestErr) Error() string { return e.msg }

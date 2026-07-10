package handler

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"strings"
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
	if err := validateOllamaEmbedRequest(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	inputs, err := parseInputs(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	start := time.Now()
	// Ollama clients expect pre-normalised vectors for cosine similarity.
	results, err := h.embedding.EmbedForModel(r.Context(), req.Model, inputs, true)
	if err != nil {
		slog.Error("embed failed", "error", err)
		writeRuntimeError(w, err)
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
func writeOpenAIEmbeddingError(w http.ResponseWriter, status int, msg string, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]map[string]string{
		"error": {"message": msg, "code": code},
	})
}

func (h *EmbedHandler) EmbedOpenAI(w http.ResponseWriter, r *http.Request) {
	var req model.OpenAIEmbeddingsRequest
	if err := readJSON(r, &req); err != nil {
		writeOpenAIEmbeddingError(w, http.StatusBadRequest, "invalid request body", "invalid_input")
		return
	}
	if err := validateOpenAIEmbeddingsRequest(&req); err != nil {
		writeOpenAIEmbeddingError(w, http.StatusBadRequest, err.Error(), openAIEmbeddingValidationCode(err.Error()))
		return
	}

	inputs, err := parseInputs(req.Input)
	if err != nil {
		writeOpenAIEmbeddingError(w, http.StatusBadRequest, err.Error(), "invalid_input")
		return
	}

	results, err := h.embedding.EmbedForModel(r.Context(), req.Model, inputs, true)
	if err != nil {
		slog.Error("embed failed", "error", err)
		writeOpenAIEmbeddingError(w, runtimeHTTPStatus(err), err.Error(), openAIEmbeddingRuntimeCode(err))
		return
	}

	data := make([]model.OpenAIEmbeddingRecord, len(results))
	total := 0
	useBase64 := req.EncodingFormat == "base64"
	for i, res := range results {
		vec, err := applyEmbeddingDimensions(res.Vector, req.Dimensions)
		if err != nil {
			writeOpenAIEmbeddingError(w, http.StatusBadRequest, err.Error(), "invalid_dimensions")
			return
		}

		embedding := any(vec)
		if useBase64 {
			embedding = encodeEmbeddingBase64(vec)
		}
		data[i] = model.OpenAIEmbeddingRecord{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding,
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

func validateOllamaEmbedRequest(req *model.EmbedRequest) error {
	if req.Truncate != nil {
		return &badRequestErr{"truncate is not supported yet"}
	}
	if hasMeaningfulRawJSON(req.Options) {
		return &badRequestErr{"options is not supported yet"}
	}
	return nil
}

func validateOpenAIEmbeddingsRequest(req *model.OpenAIEmbeddingsRequest) error {
	if req.Dimensions != nil && *req.Dimensions <= 0 {
		return &badRequestErr{"dimensions must be > 0"}
	}
	if req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64" {
		return &badRequestErr{"encoding_format must be \"float\" or \"base64\""}
	}
	return nil
}

func openAIEmbeddingValidationCode(msg string) string {
	switch {
	case strings.Contains(msg, "dimensions"):
		return "invalid_dimensions"
	case strings.Contains(msg, "encoding_format"):
		return "unsupported_encoding_format"
	default:
		return "invalid_input"
	}
}

func openAIEmbeddingRuntimeCode(err error) string {
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "no model loaded"),
		isModelNotFoundMessage(msg):
		return "model_not_found"
	case strings.Contains(msg, "does not support embeddings"):
		return "model_capability_mismatch"
	default:
		if code := runtimeErrorCode(err); code != "" {
			return code
		}
		return "runtime_error"
	}
}

func applyEmbeddingDimensions(vec []float32, dimensions *int) ([]float32, error) {
	if dimensions == nil {
		return vec, nil
	}
	dim := *dimensions
	if dim > len(vec) {
		return nil, &badRequestErr{fmt.Sprintf("requested dimensions %d exceed embedding length %d", dim, len(vec))}
	}
	if dim == len(vec) {
		return vec, nil
	}
	return vec[:dim], nil
}

func encodeEmbeddingBase64(vec []float32) string {
	buf := make([]byte, len(vec)*4)
	for i, value := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(value))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

type badRequestErr struct{ msg string }

func (e *badRequestErr) Error() string { return e.msg }

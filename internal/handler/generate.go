package handler

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
)

// GenerateHandler serves Ollama's /api/generate. Same engine as chat, but
// the prompt is raw (no chat template) unless `system` is provided.
type GenerateHandler struct {
	inference *service.InferenceService
}

func NewGenerateHandler(inf *service.InferenceService) *GenerateHandler {
	return &GenerateHandler{inference: inf}
}

// Generate handles POST /api/generate.
func (h *GenerateHandler) Generate(w http.ResponseWriter, r *http.Request) {
	var req model.GenerateRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	// Ollama default: stream unless explicitly disabled.
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	params := toEngineParamsFromGenerate(&req)
	if stream {
		h.handleStream(w, r, &req, params)
		return
	}
	h.handleComplete(w, r, &req, params)
}

func (h *GenerateHandler) handleComplete(
	w http.ResponseWriter, r *http.Request, req *model.GenerateRequest, params engine.CompletionParams,
) {
	result, err := h.inference.Generate(r.Context(), req.Prompt, req.System, params)
	if err != nil {
		slog.Error("generate failed", "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	h.inference.ApplyKeepAlive(req.KeepAlive)

	resp := model.GenerateResponse{
		Model:                req.Model,
		Response:             result.Text,
		Done:                 true,
		CreatedAt:            time.Now().UTC().Format(time.RFC3339Nano),
		TotalDurationNs:      result.Timings.TotalNs,
		PromptEvalDurationNs: result.Timings.PromptEvalNs,
		PromptEvalCount:      result.PromptTokens,
		EvalDurationNs:       result.Timings.EvalNs,
		EvalCount:            result.CompletionTokens,
		DoneReason:           result.FinishReason,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *GenerateHandler) handleStream(
	w http.ResponseWriter, r *http.Request, req *model.GenerateRequest, params engine.CompletionParams,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	ch, err := h.inference.GenerateStream(r.Context(), req.Prompt, req.System, params)
	if err != nil {
		slog.Error("generate stream failed", "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Ollama's /api/generate uses newline-delimited JSON, NOT SSE.
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	for chunk := range ch {
		if chunk.Err != nil {
			slog.Error("generate chunk error", "error", chunk.Err)
			// Ollama convention: emit a final done=true with an empty response
			// — client detects termination via done + the error surfacing as
			// HTTP connection close.
			break
		}

		var resp model.GenerateResponse
		resp.Model = req.Model
		resp.CreatedAt = time.Now().UTC().Format(time.RFC3339Nano)
		if chunk.Done {
			resp.Done = true
			resp.DoneReason = chunk.FinishReason
			resp.TotalDurationNs = chunk.Timings.TotalNs
			resp.PromptEvalDurationNs = chunk.Timings.PromptEvalNs
			resp.PromptEvalCount = chunk.PromptTokens
			resp.EvalDurationNs = chunk.Timings.EvalNs
			resp.EvalCount = chunk.CompletionTokens
		} else {
			resp.Response = chunk.Text
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

// toEngineParamsFromGenerate maps /api/generate's nested options → engine params.
func toEngineParamsFromGenerate(req *model.GenerateRequest) engine.CompletionParams {
	p := engine.DefaultCompletionParams()
	o := req.Options
	if o == nil {
		return p
	}
	if o.NumPredict != nil {
		p.MaxTokens = *o.NumPredict
	}
	if o.Temperature != nil {
		p.Temperature = float32(*o.Temperature)
	}
	if o.TopP != nil {
		p.TopP = float32(*o.TopP)
	}
	if o.TopK != nil {
		p.TopK = *o.TopK
	}
	if o.MinP != nil {
		p.MinP = float32(*o.MinP)
	}
	if o.TypicalP != nil {
		p.TypicalP = float32(*o.TypicalP)
	}
	if o.RepeatPenalty != nil {
		p.RepeatPenalty = float32(*o.RepeatPenalty)
	}
	if o.RepeatLastN != nil {
		p.RepeatLastN = *o.RepeatLastN
	}
	if o.FrequencyPenalty != nil {
		p.FrequencyPenalty = float32(*o.FrequencyPenalty)
	}
	if o.PresencePenalty != nil {
		p.PresencePenalty = float32(*o.PresencePenalty)
	}
	if o.Seed != nil {
		p.Seed = *o.Seed
	}
	if o.Mirostat != nil {
		p.Mirostat = *o.Mirostat
	}
	if o.MirostatTau != nil {
		p.MirostatTau = float32(*o.MirostatTau)
	}
	if o.MirostatEta != nil {
		p.MirostatEta = float32(*o.MirostatEta)
	}
	if len(o.Stop) > 0 {
		p.Stop = o.Stop
	}
	return p
}

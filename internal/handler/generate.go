package handler

import (
	"bytes"
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
	inference         *service.InferenceService
	multimodalEnabled bool
}

func NewGenerateHandler(inf *service.InferenceService) *GenerateHandler {
	return &GenerateHandler{inference: inf}
}

func (h *GenerateHandler) SetMultimodalEnabled(enabled bool) {
	h.multimodalEnabled = enabled
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
	if err := validateGenerateRequest(&req, h.multimodalEnabled); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validateThinkOption(req.Think); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Ollama default: stream unless explicitly disabled.
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	params := toEngineParamsFromGenerate(&req)
	if grammar, ok, err := structuredFormatGrammar(req.Format); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		params.Grammar = grammar
	}
	if stream {
		if shouldBufferOllamaGenerateStream(&req) {
			h.handleBufferedStream(w, r, &req, params)
			return
		}
		h.handleStream(w, r, &req, params)
		return
	}
	h.handleComplete(w, r, &req, params)
}

func shouldBufferOllamaGenerateStream(req *model.GenerateRequest) bool {
	return hasMeaningfulRawJSON(req.Format) || hasMeaningfulRawJSON(req.Think)
}

// Completion handles POST /v1/completions using OpenAI's legacy completions
// response shape. Unlike /api/generate, OpenAI completions are non-streaming
// by default and use SSE only when stream=true.
func (h *GenerateHandler) Completion(w http.ResponseWriter, r *http.Request) {
	var req model.CompletionRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	prompt, err := completionPrompt(req.Prompt)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	params := toEngineParamsFromCompletion(&req)
	if req.Stream {
		h.handleCompletionStream(w, r, &req, prompt, params)
		return
	}
	h.handleCompletion(w, r, &req, prompt, params)
}

func (h *GenerateHandler) handleCompletion(
	w http.ResponseWriter,
	r *http.Request,
	req *model.CompletionRequest,
	prompt string,
	params engine.CompletionParams,
) {
	result, err := h.inference.Generate(r.Context(), req.Model, prompt, "", nil, params)
	if err != nil {
		slog.Error("completion failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	resp := model.CompletionResponse{
		ID:      "cmpl-" + newShortID(),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []model.CompletionChoice{
			{Text: result.Text, Index: 0, FinishReason: &result.FinishReason},
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

func (h *GenerateHandler) handleCompletionStream(
	w http.ResponseWriter,
	r *http.Request,
	req *model.CompletionRequest,
	prompt string,
	params engine.CompletionParams,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	ch, err := h.inference.GenerateStream(r.Context(), req.Model, prompt, "", nil, params)
	if err != nil {
		slog.Error("completion stream failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	id := "cmpl-" + newShortID()
	created := time.Now().Unix()
	for chunk := range ch {
		if chunk.Err != nil {
			slog.Error("completion chunk error", "error", chunk.Err)
			errPayload := map[string]any{
				"error": map[string]any{
					"message": chunk.Err.Error(),
					"type":    "runtime_stream_error",
				},
			}
			data, _ := json.Marshal(errPayload)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			return
		}
		choice := model.CompletionChunkChoice{Index: 0, Text: chunk.Text}
		if chunk.Done {
			choice.Text = ""
			choice.FinishReason = &chunk.FinishReason
		}
		resp := model.CompletionChunk{
			ID:      id,
			Object:  "text_completion",
			Created: created,
			Model:   req.Model,
			Choices: []model.CompletionChunkChoice{choice},
		}
		data, _ := json.Marshal(resp)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		if chunk.Done {
			break
		}
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (h *GenerateHandler) handleComplete(
	w http.ResponseWriter, r *http.Request, req *model.GenerateRequest, params engine.CompletionParams,
) {
	prompt, system := req.Prompt, req.System
	if instruction, ok, err := structuredFormatInstruction(req.Format); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		system = appendInstruction(system, instruction)
	}
	result, err := h.inference.Generate(r.Context(), req.Model, prompt, system, req.Images, params)
	if err != nil {
		slog.Error("generate failed", "error", err)
		writeRuntimeError(w, err)
		return
	}
	content, thinking := applyThinkingOutput(req.Think, result.Text)
	if err := validateStructuredOutput(req.Format, content); err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	h.inference.ApplyKeepAlive(req.KeepAlive)

	resp := model.GenerateResponse{
		Model:                req.Model,
		Response:             content,
		Thinking:             thinking,
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

	ch, err := h.inference.GenerateStream(r.Context(), req.Model, req.Prompt, req.System, req.Images, params)
	if err != nil {
		slog.Error("generate stream failed", "error", err)
		writeRuntimeError(w, err)
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
			resp := model.GenerateResponse{
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

func (h *GenerateHandler) handleBufferedStream(
	w http.ResponseWriter, r *http.Request, req *model.GenerateRequest, params engine.CompletionParams,
) {
	prompt, system := req.Prompt, req.System
	if instruction, ok, err := structuredFormatInstruction(req.Format); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	} else if ok {
		system = appendInstruction(system, instruction)
	}

	ch, err := h.inference.GenerateStream(r.Context(), req.Model, prompt, system, req.Images, params)
	if err != nil {
		slog.Error("generate buffered stream failed", "error", err)
		writeRuntimeError(w, err)
		return
	}
	text, final, err := collectCompletionStream(ch)
	if err != nil {
		slog.Error("generate buffered stream chunk error", "error", err)
		writeRuntimeError(w, err)
		h.inference.ApplyKeepAlive(req.KeepAlive)
		return
	}

	content, thinking := applyThinkingOutput(req.Think, text)
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
	if thinking != "" {
		writeOllamaGenerateStreamResponse(w, flusher, model.GenerateResponse{
			Model:     req.Model,
			CreatedAt: createdAt,
			Thinking:  thinking,
		})
	}
	if content != "" {
		writeOllamaGenerateStreamResponse(w, flusher, model.GenerateResponse{
			Model:     req.Model,
			CreatedAt: createdAt,
			Response:  content,
		})
	}
	writeOllamaGenerateStreamResponse(w, flusher, model.GenerateResponse{
		Model:                req.Model,
		CreatedAt:            createdAt,
		Done:                 true,
		DoneReason:           final.FinishReason,
		TotalDurationNs:      final.Timings.TotalNs,
		PromptEvalDurationNs: final.Timings.PromptEvalNs,
		PromptEvalCount:      final.PromptTokens,
		EvalDurationNs:       final.Timings.EvalNs,
		EvalCount:            final.CompletionTokens,
	})
	h.inference.ApplyKeepAlive(req.KeepAlive)
}

func writeOllamaGenerateStreamResponse(w http.ResponseWriter, flusher http.Flusher, resp model.GenerateResponse) {
	if data, err := json.Marshal(resp); err == nil {
		fmt.Fprintf(w, "%s\n", data)
		flusher.Flush()
	}
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

func validateGenerateRequest(req *model.GenerateRequest, multimodalEnabled bool) error {
	if req.Template != "" {
		return fmt.Errorf("template is not supported yet")
	}
	if len(req.Context) > 0 {
		return fmt.Errorf("context reuse is not supported yet")
	}
	if req.Suffix != "" {
		return fmt.Errorf("suffix is not supported yet")
	}
	if len(req.Images) > 0 {
		if !multimodalEnabled {
			return fmt.Errorf("multimodal images require ORCHESTRA_MMPROJ_PATH")
		}
	}
	return nil
}

func toEngineParamsFromCompletion(req *model.CompletionRequest) engine.CompletionParams {
	p := engine.DefaultCompletionParams()
	if req.MaxTokens != nil {
		p.MaxTokens = *req.MaxTokens
	}
	if req.Temperature != nil {
		p.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		p.TopP = float32(*req.TopP)
	}
	if req.TopK != nil {
		p.TopK = *req.TopK
	}
	if req.MinP != nil {
		p.MinP = float32(*req.MinP)
	}
	if req.TypicalP != nil {
		p.TypicalP = float32(*req.TypicalP)
	}
	if req.RepeatPenalty != nil {
		p.RepeatPenalty = float32(*req.RepeatPenalty)
	}
	if req.RepeatLastN != nil {
		p.RepeatLastN = *req.RepeatLastN
	}
	if req.FrequencyPenalty != nil {
		p.FrequencyPenalty = float32(*req.FrequencyPenalty)
	}
	if req.PresencePenalty != nil {
		p.PresencePenalty = float32(*req.PresencePenalty)
	}
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	if req.Mirostat != nil {
		p.Mirostat = *req.Mirostat
	}
	if req.MirostatTau != nil {
		p.MirostatTau = float32(*req.MirostatTau)
	}
	if req.MirostatEta != nil {
		p.MirostatEta = float32(*req.MirostatEta)
	}
	if len(req.Stop) > 0 {
		p.Stop = req.Stop
	}
	return p
}

func completionPrompt(raw json.RawMessage) (string, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "", fmt.Errorf("prompt is required")
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return "", fmt.Errorf("prompt is required")
		}
		return s, nil
	}
	var parts []string
	if err := json.Unmarshal(raw, &parts); err == nil {
		if len(parts) == 0 {
			return "", fmt.Errorf("prompt is required")
		}
		return parts[0], nil
	}
	return "", fmt.Errorf("prompt must be a string or string array")
}

func newShortID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())[:8]
}

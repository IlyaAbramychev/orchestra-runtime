package handler

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
	"github.com/operium/orchestra-runtime/internal/storage"
)

type ModelsHandler struct {
	manager *service.ModelManager
	engine  engine.Backend
}

func NewModelsHandler(manager *service.ModelManager, eng engine.Backend) *ModelsHandler {
	return &ModelsHandler{manager: manager, engine: eng}
}

// ListOpenAI handles GET /v1/models (OpenAI-compatible format).
func (h *ModelsHandler) ListOpenAI(w http.ResponseWriter, r *http.Request) {
	entries := h.manager.List()
	data := make([]model.OpenAIModel, 0, len(entries))
	for _, e := range entries {
		if modelAvailableForInference(e.Status) {
			data = append(data, model.OpenAIModel{
				ID:      e.Name,
				Object:  "model",
				Created: e.DownloadedAt.Unix(),
				OwnedBy: "local",
			})
		}
	}
	writeJSON(w, http.StatusOK, model.OpenAIModelList{
		Object: "list",
		Data:   data,
	})
}

// List handles GET /api/models (extended format).
func (h *ModelsHandler) List(w http.ResponseWriter, r *http.Request) {
	entries := h.manager.List()
	result := make([]model.ModelInfo, 0, len(entries))
	for _, e := range entries {
		result = append(result, toModelInfo(e))
	}
	writeJSON(w, http.StatusOK, result)
}

// Import handles POST /api/models/import.
// Body: {"path": "/path/to/directory"} — scans and registers all .gguf files found.
func (h *ModelsHandler) Import(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Path == "" {
		writeError(w, http.StatusBadRequest, "path is required")
		return
	}

	entries, err := h.manager.ImportFromDirectory(req.Path)
	if err != nil {
		slog.Error("import models failed", "path", req.Path, "error", err)
		writeRuntimeError(w, err)
		return
	}

	result := make([]model.ModelInfo, 0, len(entries))
	for _, e := range entries {
		result = append(result, toModelInfo(e))
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"imported": len(result),
		"models":   result,
	})
}

// Pull handles POST /api/models/pull.
func (h *ModelsHandler) Pull(w http.ResponseWriter, r *http.Request) {
	var req model.PullModelRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if req.SourceURL == "" {
		writeError(w, http.StatusBadRequest, "source_url is required")
		return
	}
	if req.Name == "" {
		req.Name = "model"
	}

	id, err := h.manager.PullModelWithMetadata(req.Name, req.SourceURL, service.PullModelMetadata{
		Quantization:        req.Quantization,
		Family:              req.Family,
		Parameters:          req.Parameters,
		SHA256:              req.SHA256,
		Template:            req.Template,
		StopTokens:          append([]string(nil), req.StopTokens...),
		Capabilities:        toStorageCapabilitiesPtr(req.Capabilities),
		RecommendedSettings: toStorageRecommendedSettingsPtr(req.RecommendedSettings),
	})
	if err != nil {
		slog.Error("pull model failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	writeJSON(w, http.StatusAccepted, model.PullModelResponse{
		ID:     id,
		Status: "downloading",
	})
}

// PullOllama handles POST /api/pull (Ollama-compat).
//
// Ollama pulls from its own model registry by model name. Orchestra Runtime
// stores direct GGUF downloads today, so this endpoint accepts the Ollama
// fields plus an Orchestra extension: `source_url`. If `model` itself is an
// HTTP(S) URL, it is treated as the source URL and the filename stem is used as
// the display name.
func (h *ModelsHandler) PullOllama(w http.ResponseWriter, r *http.Request) {
	var req model.OllamaPullRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	name, sourceURL, err := normalizeOllamaPullRequest(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	id, err := h.manager.PullModel(name, sourceURL)
	if err != nil {
		slog.Error("ollama pull failed", "error", err)
		writeRuntimeError(w, err)
		return
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}
	if !stream {
		h.waitForOllamaPull(w, r, id)
		return
	}
	h.streamOllamaPull(w, r, id)
}

// Delete handles DELETE /api/models/{id}.
func (h *ModelsHandler) Delete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := h.manager.DeleteModel(id); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// DeleteOllama handles DELETE /api/delete (Ollama-compat).
// Body: {"model": "name-or-id"}.
func (h *ModelsHandler) DeleteOllama(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"` // permissive alias for older local clients
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	ref := req.Model
	if ref == "" {
		ref = req.Name
	}
	entry, err := h.manager.ResolveModel(ref)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	if err := h.manager.DeleteModel(entry.ID); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	w.WriteHeader(http.StatusOK)
}

func normalizeOllamaPullRequest(req *model.OllamaPullRequest) (string, string, error) {
	name := strings.TrimSpace(req.Model)
	sourceURL := strings.TrimSpace(req.SourceURL)
	if name == "" {
		return "", "", fmt.Errorf("model is required")
	}
	if sourceURL == "" && isHTTPURL(name) {
		sourceURL = name
		name = modelNameFromURL(name)
	}
	if sourceURL == "" {
		return "", "", fmt.Errorf("pull by Ollama library name is not supported yet; provide source_url")
	}
	if !isHTTPURL(sourceURL) {
		return "", "", fmt.Errorf("source_url must be an http or https URL")
	}
	if name == "" {
		name = "model"
	}
	return name, sourceURL, nil
}

func isHTTPURL(value string) bool {
	return strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://")
}

func modelNameFromURL(value string) string {
	value = strings.TrimRight(value, "/")
	if idx := strings.LastIndex(value, "/"); idx >= 0 {
		value = value[idx+1:]
	}
	value = strings.TrimSuffix(value, ".gguf")
	if value == "" {
		return "model"
	}
	return value
}

func (h *ModelsHandler) waitForOllamaPull(w http.ResponseWriter, r *http.Request, id string) {
	if ds := h.manager.GetDownloadState(id); ds != nil {
		select {
		case <-r.Context().Done():
			writeRuntimeError(w, r.Context().Err())
			return
		case <-ds.Done:
		}
	}

	entry := h.manager.Get(id)
	if entry != nil && entry.Status == model.StatusError {
		writeError(w, http.StatusInternalServerError, entry.ErrorMessage)
		return
	}
	writeJSON(w, http.StatusOK, model.OllamaPullResponse{Status: "success"})
}

func (h *ModelsHandler) streamOllamaPull(w http.ResponseWriter, r *http.Request, id string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{Status: "pulling manifest"})

	ds := h.manager.GetDownloadState(id)
	if ds == nil {
		writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{Status: "success"})
		return
	}

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ds.Done:
			entry := h.manager.Get(id)
			if entry != nil && entry.Status == model.StatusError {
				writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{
					Status: "error",
					Error:  entry.ErrorMessage,
				})
				return
			}
			writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{Status: "verifying sha256 digest"})
			writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{Status: "writing manifest"})
			writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{Status: "success"})
			return
		case <-ticker.C:
			writeOllamaPullChunk(w, flusher, model.OllamaPullResponse{
				Status:    "pulling gguf",
				Total:     ds.TotalBytes,
				Completed: ds.DownloadedBytes.Load(),
			})
		}
	}
}

func writeOllamaPullChunk(w http.ResponseWriter, flusher http.Flusher, resp model.OllamaPullResponse) {
	if data, err := json.Marshal(resp); err == nil {
		fmt.Fprintf(w, "%s\n", data)
		flusher.Flush()
	}
}

// Load handles POST /api/models/{id}/load.
//
// Body is optional — any omitted field falls back to the runtime's configured
// default load options.
// See model.LoadModelRequest for the full field list with semantics.
func (h *ModelsHandler) Load(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	var req model.LoadModelRequest
	readJSON(r, &req) // optional body

	opts := h.manager.DefaultLoadOptions()

	if req.GPULayers != nil {
		opts.GPULayers = *req.GPULayers
	}
	if req.Threads != nil {
		opts.Threads = *req.Threads
	}
	if req.ContextSize != nil {
		opts.CtxSize = *req.ContextSize
	}
	if req.BatchSize != nil {
		opts.BatchSize = *req.BatchSize
	}
	if req.RopeFreqBase != nil {
		opts.RopeFreqBase = float32(*req.RopeFreqBase)
	}
	if req.RopeFreqScale != nil {
		opts.RopeFreqScale = float32(*req.RopeFreqScale)
	}
	if req.FlashAttention != nil {
		if *req.FlashAttention {
			opts.FlashAttn = 1
		} else {
			opts.FlashAttn = 0
		}
	}
	if req.OffloadKQV != nil {
		opts.OffloadKQV = *req.OffloadKQV
	}
	if req.KVCacheQuantK != nil {
		opts.TypeK = *req.KVCacheQuantK
	}
	if req.KVCacheQuantV != nil {
		opts.TypeV = *req.KVCacheQuantV
	}
	if req.UseMmap != nil {
		opts.UseMmap = *req.UseMmap
	}
	if req.KeepModelInRAM != nil {
		opts.UseMlock = *req.KeepModelInRAM
	}

	if err := h.manager.LoadModel(id, opts); err != nil {
		slog.Error("load model failed", "id", id, "error", err)
		writeRuntimeError(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "loaded"})
}

// Unload handles POST /api/models/{id}/unload.
func (h *ModelsHandler) Unload(w http.ResponseWriter, r *http.Request) {
	h.manager.UnloadModel()
	writeJSON(w, http.StatusOK, map[string]string{"status": "idle"})
}

// Status handles GET /api/models/{id}/status.
func (h *ModelsHandler) Status(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	entry := h.manager.Get(id)
	if entry == nil {
		writeError(w, http.StatusNotFound, "model not found")
		return
	}

	snapshot := h.manager.RuntimeSnapshot()
	active := snapshot.ActiveModelID == entry.ID
	if !active && h.engine.LoadedModelID() == entry.ID && h.engine.IsLoaded() {
		active = true
	}
	resp := model.ModelStatusResponse{
		ID:           entry.ID,
		Name:         entry.Name,
		Status:       entry.Status,
		RuntimeState: snapshot.State,
		Active:       active,
		QueueDepth:   snapshot.QueueDepth,
	}

	if entry.Status == "downloading" {
		if ds := h.manager.GetDownloadState(id); ds != nil {
			resp.DownloadedBytes = ds.DownloadedBytes.Load()
			resp.TotalBytes = ds.TotalBytes
			resp.SpeedBPS = ds.SpeedBPS.Load()
			resp.DownloadAttempt = int(ds.Attempt.Load())
			resp.MaxAttempts = ds.MaxAttempts
			resp.ResumeFrom = ds.ResumeFrom.Load()
			resp.LastDownloadError = ds.LastErrorMessage()
			if ds.TotalBytes > 0 {
				resp.DownloadProgress = float64(resp.DownloadedBytes) / float64(ds.TotalBytes) * 100
			}
		}
	}

	if entry.Status == "error" {
		resp.ErrorMessage = entry.ErrorMessage
	}

	writeJSON(w, http.StatusOK, resp)
}

// ListOllamaTags handles GET /api/tags (Ollama-compat).
// Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
func (h *ModelsHandler) ListOllamaTags(w http.ResponseWriter, r *http.Request) {
	entries := h.manager.List()
	type ollamaDetails struct {
		Format            string   `json:"format"`
		Family            string   `json:"family"`
		Families          []string `json:"families,omitempty"`
		ParameterSize     string   `json:"parameter_size,omitempty"`
		QuantizationLevel string   `json:"quantization_level,omitempty"`
	}
	type ollamaTag struct {
		Name         string                  `json:"name"`
		Model        string                  `json:"model"`
		ModifiedAt   string                  `json:"modified_at"`
		Size         int64                   `json:"size"`
		Digest       string                  `json:"digest,omitempty"`
		Details      ollamaDetails           `json:"details"`
		Capabilities model.ModelCapabilities `json:"capabilities"`
	}
	tags := make([]ollamaTag, 0, len(entries))
	for _, e := range entries {
		if !modelAvailableForInference(e.Status) {
			continue
		}
		tags = append(tags, ollamaTag{
			Name:       e.Name,
			Model:      e.Name,
			ModifiedAt: e.DownloadedAt.UTC().Format("2006-01-02T15:04:05.999999999Z"),
			Size:       e.Size,
			Digest:     e.SHA256,
			Details: ollamaDetails{
				Format:            "gguf",
				Family:            e.Family,
				ParameterSize:     e.Parameters,
				QuantizationLevel: e.Quantization,
			},
			Capabilities: toModelCapabilities(e.Capabilities),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": tags})
}

// ListRunning handles GET /api/ps (Ollama-compat) — shows currently-loaded models.
func (h *ModelsHandler) ListRunning(w http.ResponseWriter, r *http.Request) {
	snapshot := h.manager.RuntimeSnapshot()
	id := snapshot.ActiveModelID
	if id == "" {
		id = h.engine.LoadedModelID()
	}
	type running struct {
		Name      string `json:"name"`
		Model     string `json:"model"`
		Size      int64  `json:"size"`
		SizeVRAM  int64  `json:"size_vram"`
		State     string `json:"state,omitempty"`
		ExpiresAt string `json:"expires_at,omitempty"`
	}
	items := []running{}
	if id != "" {
		if e := h.manager.Get(id); e != nil {
			items = append(items, running{
				Name:     e.Name,
				Model:    e.Name,
				Size:     e.Size,
				SizeVRAM: e.Size, // runtime doesn't separate; report as same
				State:    snapshot.State,
			})
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": items})
}

// Show handles POST /api/show (Ollama-compat) — model metadata.
func (h *ModelsHandler) Show(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"` // alias
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	id := req.Model
	if id == "" {
		id = req.Name
	}
	if id == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	entry := h.manager.Get(id)
	if entry == nil {
		// Also try by name — client side usually sends display name, we store by id.
		for _, e := range h.manager.List() {
			if e.Name == id {
				entry = e
				break
			}
		}
	}
	if entry == nil {
		writeError(w, http.StatusNotFound, "model not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"modelfile":  "", // placeholder — we don't support Modelfile yet
		"parameters": "", // future: surface load-time params
		"template":   entry.Template,
		"details": map[string]any{
			"format":             "gguf",
			"family":             entry.Family,
			"parameter_size":     entry.Parameters,
			"quantization_level": entry.Quantization,
		},
		"capabilities":         toModelCapabilities(entry.Capabilities),
		"stop_tokens":          entry.StopTokens,
		"recommended_settings": toRecommendedSettings(entry.RecommendedSettings),
		"model_info": map[string]any{
			"general.name":       entry.Name,
			"general.size_bytes": entry.Size,
			"general.file_path":  entry.FilePath,
			"general.sha256":     entry.SHA256,
		},
	})
}

func toModelInfo(e *storage.ModelEntry) model.ModelInfo {
	return model.ModelInfo{
		ID:                  e.ID,
		Name:                e.Name,
		Filename:            e.Filename,
		Size:                e.Size,
		SizeHuman:           model.HumanSize(e.Size),
		Quantization:        e.Quantization,
		Family:              e.Family,
		Parameters:          e.Parameters,
		Template:            e.Template,
		StopTokens:          e.StopTokens,
		Capabilities:        toModelCapabilities(e.Capabilities),
		RecommendedSettings: toRecommendedSettings(e.RecommendedSettings),
		SourceURL:           e.SourceURL,
		SHA256:              e.SHA256,
		Status:              e.Status,
		ErrorMessage:        e.ErrorMessage,
		FilePath:            e.FilePath,
		DownloadedAt:        e.DownloadedAt,
	}
}

func toModelCapabilities(c storage.ModelCapabilities) model.ModelCapabilities {
	return model.ModelCapabilities{
		Chat:       c.Chat,
		Embeddings: c.Embeddings,
		Rerank:     c.Rerank,
		Tools:      c.Tools,
		Thinking:   c.Thinking,
	}
}

func toStorageCapabilitiesPtr(c *model.ModelCapabilities) *storage.ModelCapabilities {
	if c == nil {
		return nil
	}
	return &storage.ModelCapabilities{
		Chat:       c.Chat,
		Embeddings: c.Embeddings,
		Rerank:     c.Rerank,
		Tools:      c.Tools,
		Thinking:   c.Thinking,
	}
}

func toRecommendedSettings(s storage.RecommendedModelSettings) model.RecommendedModelSettings {
	return model.RecommendedModelSettings{
		ContextSize: s.ContextSize,
	}
}

func toStorageRecommendedSettingsPtr(s *model.RecommendedModelSettings) *storage.RecommendedModelSettings {
	if s == nil {
		return nil
	}
	return &storage.RecommendedModelSettings{
		ContextSize: s.ContextSize,
	}
}

func modelAvailableForInference(status string) bool {
	switch status {
	case model.StatusReady, model.StatusLoaded,
		engine.StateLoading, engine.StateGenerating, engine.StateUnloading:
		return true
	default:
		return false
	}
}

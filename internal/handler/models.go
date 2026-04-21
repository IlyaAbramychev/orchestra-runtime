package handler

import (
	"log/slog"
	"net/http"

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
		if e.Status == "ready" || e.Status == "loaded" {
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
		writeError(w, http.StatusInternalServerError, err.Error())
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

	id, err := h.manager.PullModel(req.Name, req.SourceURL)
	if err != nil {
		slog.Error("pull model failed", "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusAccepted, model.PullModelResponse{
		ID:     id,
		Status: "downloading",
	})
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

// Load handles POST /api/models/{id}/load.
func (h *ModelsHandler) Load(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	var req model.LoadModelRequest
	readJSON(r, &req) // optional body

	gpuLayers := -1
	ctxSize := 4096
	threads := 0 // 0 = auto

	if req.GPULayers != nil {
		gpuLayers = *req.GPULayers
	}
	if req.ContextSize != nil {
		ctxSize = *req.ContextSize
	}

	if err := h.manager.LoadModel(id, gpuLayers, ctxSize, threads); err != nil {
		slog.Error("load model failed", "id", id, "error", err)
		writeError(w, http.StatusInternalServerError, err.Error())
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

	resp := model.ModelStatusResponse{
		ID:     entry.ID,
		Name:   entry.Name,
		Status: entry.Status,
	}

	if entry.Status == "downloading" {
		if ds := h.manager.GetDownloadState(id); ds != nil {
			resp.DownloadedBytes = ds.DownloadedBytes.Load()
			resp.TotalBytes = ds.TotalBytes
			resp.SpeedBPS = ds.SpeedBPS.Load()
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
		Name       string        `json:"name"`
		Model      string        `json:"model"`
		ModifiedAt string        `json:"modified_at"`
		Size       int64         `json:"size"`
		Digest     string        `json:"digest,omitempty"`
		Details    ollamaDetails `json:"details"`
	}
	tags := make([]ollamaTag, 0, len(entries))
	for _, e := range entries {
		if e.Status != "ready" && e.Status != "loaded" {
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
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": tags})
}

// ListRunning handles GET /api/ps (Ollama-compat) — shows currently-loaded models.
func (h *ModelsHandler) ListRunning(w http.ResponseWriter, r *http.Request) {
	id := h.engine.LoadedModelID()
	type running struct {
		Name      string `json:"name"`
		Model     string `json:"model"`
		Size      int64  `json:"size"`
		SizeVRAM  int64  `json:"size_vram"`
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
		"template":   "", // future: surface chat template
		"details": map[string]any{
			"format":             "gguf",
			"family":             entry.Family,
			"parameter_size":     entry.Parameters,
			"quantization_level": entry.Quantization,
		},
		"model_info": map[string]any{
			"general.name":        entry.Name,
			"general.size_bytes":  entry.Size,
			"general.file_path":   entry.FilePath,
			"general.sha256":      entry.SHA256,
		},
	})
}

func toModelInfo(e *storage.ModelEntry) model.ModelInfo {
	return model.ModelInfo{
		ID:           e.ID,
		Name:         e.Name,
		Filename:     e.Filename,
		Size:         e.Size,
		SizeHuman:    model.HumanSize(e.Size),
		Quantization: e.Quantization,
		Family:       e.Family,
		Parameters:   e.Parameters,
		SourceURL:    e.SourceURL,
		SHA256:       e.SHA256,
		Status:       e.Status,
		ErrorMessage: e.ErrorMessage,
		FilePath:     e.FilePath,
		DownloadedAt: e.DownloadedAt,
	}
}

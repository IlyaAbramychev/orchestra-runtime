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
	engine  *engine.Engine
}

func NewModelsHandler(manager *service.ModelManager, eng *engine.Engine) *ModelsHandler {
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

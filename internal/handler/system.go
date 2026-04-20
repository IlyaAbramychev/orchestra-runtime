package handler

import (
	"net/http"

	"github.com/operium/orchestra-runtime/internal/service"
)

type SystemHandler struct {
	sysInfo   *service.SystemInfo
	inference *service.InferenceService
}

func NewSystemHandler(sysInfo *service.SystemInfo) *SystemHandler {
	return &SystemHandler{sysInfo: sysInfo}
}

// SetInference sets the inference service for queue depth reporting.
func (h *SystemHandler) SetInference(inference *service.InferenceService) {
	h.inference = inference
}

// Health handles GET /health.
func (h *SystemHandler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"service": "orchestra-runtime",
	})
}

// Info handles GET /api/system.
func (h *SystemHandler) Info(w http.ResponseWriter, r *http.Request) {
	queueDepth := 0
	if h.inference != nil {
		queueDepth = h.inference.QueueDepth()
	}
	info := h.sysInfo.GetInfo(queueDepth)
	writeJSON(w, http.StatusOK, info)
}

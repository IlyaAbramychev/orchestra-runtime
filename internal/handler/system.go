package handler

import (
	"net/http"

	"github.com/operium/orchestra-runtime/internal/model"
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

// Version handles GET /api/version using Ollama's response shape.
func (h *SystemHandler) Version(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"version": service.Version,
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

// Status handles GET /api/status. It is a compact, read-only runtime state
// endpoint intended for clients that need the actual loaded model limits.
func (h *SystemHandler) Status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.sysInfo.GetStatus())
}

// Capabilities handles GET /api/capabilities. It is an Orchestra extension for
// clients that need feature detection instead of probing Ollama-shaped errors.
func (h *SystemHandler) Capabilities(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, model.RuntimeCapabilitiesResponse{
		Service: "orchestra-runtime",
		Version: service.Version,
		Ollama: model.OllamaCapabilities{
			Compatible: true,
			Endpoints: []string{
				"POST /api/chat",
				"POST /api/generate",
				"POST /api/embed",
				"POST /api/embeddings",
				"GET /api/tags",
				"GET /api/ps",
				"POST /api/show",
				"POST /api/copy",
				"POST /api/create",
				"DELETE /api/delete",
				"POST /api/pull",
				"GET /api/version",
			},
		},
		Features: []model.FeatureCapability{
			{Name: "chat", Status: "supported"},
			{Name: "generate", Status: "supported"},
			{Name: "embeddings", Status: "supported", Notes: "Supports encoding_format='float' (default), 'base64'; supports dimensions truncation; stable error.code available"},
			{Name: "registry_pull", Status: "supported"},
			{Name: "direct_gguf_pull", Status: "extension"},
			{Name: "metadata_create", Status: "partial", Notes: "metadata-only derived models share an existing GGUF artifact"},
			{Name: "structured_output_non_streaming", Status: "supported"},
			{Name: "json_schema_validation", Status: "supported"},
			{Name: "structured_output_streaming", Status: "partial", Notes: "buffered stream; validates final output before emitting NDJSON"},
			{Name: "tool_calls_non_streaming", Status: "partial", Notes: "model-scoped: requires a GGUF chat template with native tool/tool-call support", Details: map[string]any{"scope": "model_chat_template", "transport": "llama_cpp_native"}},
			{Name: "tool_calls_streaming", Status: "partial", Notes: "model-scoped and buffered; emits parsed tool_calls after generation completes", Details: map[string]any{"scope": "model_chat_template", "transport": "llama_cpp_native"}},
			{Name: "openai_tool_calls_non_streaming", Status: "partial", Notes: "model-scoped; tools and history are passed through llama.cpp's native chat-template API", Details: map[string]any{"scope": "model_chat_template", "malformedFinishReason": "tool_protocol_error"}},
			{Name: "openai_tool_calls_streaming", Status: "partial", Notes: "model-scoped buffered SSE; availability must not be applied globally to every model", Details: map[string]any{"scope": "model_chat_template", "malformedFinishReason": "tool_protocol_error"}},
			{Name: "thinking_non_streaming", Status: "partial", Notes: "model-scoped native thinking control; returns reasoning_content separately from content"},
			{Name: "thinking_streaming", Status: "partial", Notes: "model-scoped buffered stream; returns reasoning separately after native parsing"},
			{
				Name:   "multimodal_images",
				Status: "partial",
				Notes:  "supports Ollama images and OpenAI image_url data URIs; requires a compatible global or model-scoped mmproj",
				Details: map[string]any{
					"transports":          []string{"ollama_images", "openai_image_url_data_uri"},
					"mimeTypes":           []string{"image/jpeg", "image/png", "image/webp"},
					"maxImagesPerRequest": maxImagesPerRequest,
					"maxImageBytes":       maxDecodedImageBytes,
					"maxTotalImageBytes":  maxDecodedImageBytesTotal,
					"remoteURLs":          false,
				},
			},
			{Name: "grammar_constrained_decoding", Status: "supported", Notes: "Ollama format uses llama.cpp JSON Schema to GBNF conversion"},
		},
	})
}

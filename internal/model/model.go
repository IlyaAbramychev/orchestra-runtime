package model

import (
	"fmt"
	"time"
)

// ModelInfo represents a downloaded model in the registry.
type ModelInfo struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Filename     string    `json:"filename"`
	Size         int64     `json:"size"`
	SizeHuman    string    `json:"size_human"`
	Quantization string    `json:"quantization,omitempty"`
	Family       string    `json:"family,omitempty"`
	Parameters   string    `json:"parameters,omitempty"`
	SourceURL    string    `json:"source_url"`
	SHA256       string    `json:"sha256,omitempty"`
	Status       string    `json:"status"`
	ErrorMessage string    `json:"error_message,omitempty"`
	FilePath     string    `json:"file_path"`
	DownloadedAt time.Time `json:"downloaded_at"`
}

// Model statuses
const (
	StatusDownloading = "downloading"
	StatusReady       = "ready"
	StatusLoaded      = "loaded"
	StatusError       = "error"
)

type PullModelRequest struct {
	Name      string `json:"name"`
	SourceURL string `json:"source_url"`
}

type PullModelResponse struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

// LoadModelRequest is the payload for POST /api/models/{id}/load. All fields
// are optional — omit to inherit the runtime default. Flag names mirror
// llama.cpp CLI where possible so users familiar with llama-server recognise
// them.
type LoadModelRequest struct {
	// Hardware
	GPULayers *int `json:"gpu_layers,omitempty"` // -1 = all available (default)
	Threads   *int `json:"threads,omitempty"`    // 0 = auto

	// Context window
	ContextSize *int `json:"context_size,omitempty"` // n_ctx (default 4096)
	BatchSize   *int `json:"batch_size,omitempty"`   // n_batch (default 512)

	// RoPE (only override if you extrapolate context)
	RopeFreqBase  *float64 `json:"rope_freq_base,omitempty"`
	RopeFreqScale *float64 `json:"rope_freq_scale,omitempty"`

	// KV cache placement + quantisation
	FlashAttention *bool   `json:"flash_attention,omitempty"` // nil = auto; true/false override
	OffloadKQV     *bool   `json:"kv_cache_gpu_offload,omitempty"`
	KVCacheQuantK  *string `json:"kv_cache_quant_k,omitempty"` // f16|q8_0|q4_0|...
	KVCacheQuantV  *string `json:"kv_cache_quant_v,omitempty"`

	// Memory placement
	UseMmap          *bool `json:"use_mmap,omitempty"`           // default true
	KeepModelInRAM   *bool `json:"keep_model_in_memory,omitempty"` // use_mlock
}

type ModelStatusResponse struct {
	ID              string  `json:"id"`
	Name            string  `json:"name"`
	Status          string  `json:"status"`
	DownloadProgress float64 `json:"download_progress,omitempty"`
	DownloadedBytes int64   `json:"downloaded_bytes,omitempty"`
	TotalBytes      int64   `json:"total_bytes,omitempty"`
	SpeedBPS        int64   `json:"speed_bytes_per_sec,omitempty"`
	ErrorMessage    string  `json:"error_message,omitempty"`
}

// OpenAIModel is the OpenAI-compatible model list entry.
type OpenAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// HumanSize returns a human-readable file size string.
func HumanSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

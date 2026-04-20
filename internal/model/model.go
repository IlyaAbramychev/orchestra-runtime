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

type LoadModelRequest struct {
	GPULayers   *int `json:"gpu_layers,omitempty"`
	ContextSize *int `json:"context_size,omitempty"`
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

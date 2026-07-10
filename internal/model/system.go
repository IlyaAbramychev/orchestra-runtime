package model

type SystemInfoResponse struct {
	Service        string   `json:"service"`
	Version        string   `json:"version"`
	BuildCommit    string   `json:"buildCommit"`
	LlamaCppCommit string   `json:"llamaCppCommit"`
	Platform       string   `json:"platform"`
	OS             string   `json:"os"`
	Arch           string   `json:"arch"`
	CPUCount       int      `json:"cpu_count"`
	TotalRAM       int64    `json:"total_ram_bytes"`
	AvailableRAM   int64    `json:"available_ram_bytes"`
	GPU            *GPUInfo `json:"gpu,omitempty"`
	EngineState    string   `json:"engine_state"`
	CurrentModel   *string  `json:"current_model,omitempty"`
	ContextSize    int      `json:"context_size,omitempty"`
	QueueDepth     int      `json:"queue_depth"`
	// IdleTimeoutSeconds: auto-unload timer in seconds (0 = disabled).
	IdleTimeoutSeconds int `json:"idle_timeout_seconds"`
}

type RuntimeStatusResponse struct {
	State           string  `json:"state"`
	Model           *string `json:"model"`
	ContextSize     *int    `json:"contextSize"`
	MaxOutputTokens int     `json:"maxOutputTokens"`
	GPULayers       int     `json:"gpuLayers"`
	Threads         int     `json:"threads"`
	LoadedAt        *string `json:"loadedAt"`
	Error           *string `json:"error"`
}

type RuntimeCapabilitiesResponse struct {
	Service  string              `json:"service"`
	Version  string              `json:"version"`
	Ollama   OllamaCapabilities  `json:"ollama"`
	Features []FeatureCapability `json:"features"`
}

type OllamaCapabilities struct {
	Compatible bool     `json:"compatible"`
	Endpoints  []string `json:"endpoints"`
}

type FeatureCapability struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Notes  string `json:"notes,omitempty"`
	Details interface{} `json:"details,omitempty"`
}

type GPUInfo struct {
	Name      string `json:"name"`
	TotalVRAM int64  `json:"total_vram_bytes"`
	FreeVRAM  int64  `json:"free_vram_bytes"`
	Backend   string `json:"backend"` // metal, cuda, rocm, cpu
}

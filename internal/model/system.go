package model

type SystemInfoResponse struct {
	Service            string   `json:"service"`
	Version            string   `json:"version"`
	OS                 string   `json:"os"`
	Arch               string   `json:"arch"`
	CPUCount           int      `json:"cpu_count"`
	TotalRAM           int64    `json:"total_ram_bytes"`
	AvailableRAM       int64    `json:"available_ram_bytes"`
	GPU                *GPUInfo `json:"gpu,omitempty"`
	EngineState        string   `json:"engine_state"`
	CurrentModel       *string  `json:"current_model,omitempty"`
	QueueDepth         int      `json:"queue_depth"`
	// IdleTimeoutSeconds: auto-unload timer in seconds (0 = disabled).
	IdleTimeoutSeconds int `json:"idle_timeout_seconds"`
}

type GPUInfo struct {
	Name      string `json:"name"`
	TotalVRAM int64  `json:"total_vram_bytes"`
	FreeVRAM  int64  `json:"free_vram_bytes"`
	Backend   string `json:"backend"` // metal, cuda, rocm, cpu
}

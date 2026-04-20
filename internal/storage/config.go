package storage

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// UserConfig stores user preferences that persist across restarts.
type UserConfig struct {
	DefaultModel string `json:"default_model,omitempty"`
	ContextSize  int    `json:"context_size,omitempty"`
	GPULayers    int    `json:"gpu_layers,omitempty"`
	Threads      int    `json:"threads,omitempty"`
}

func LoadUserConfig(configDir string) (*UserConfig, error) {
	path := filepath.Join(configDir, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &UserConfig{}, nil
		}
		return nil, err
	}
	var cfg UserConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func SaveUserConfig(configDir string, cfg *UserConfig) error {
	path := filepath.Join(configDir, "config.json")
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

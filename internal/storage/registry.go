package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ModelEntry represents a downloaded model on disk.
type ModelEntry struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Filename     string    `json:"filename"`
	Size         int64     `json:"size"`
	Quantization string    `json:"quantization,omitempty"`
	Family       string    `json:"family,omitempty"`
	Parameters   string    `json:"parameters,omitempty"`
	SourceURL    string    `json:"source_url"`
	SHA256       string    `json:"sha256,omitempty"`
	Status       string    `json:"status"`
	ErrorMessage string    `json:"error_message,omitempty"`
	FilePath     string    `json:"file_path"`
	DownloadedAt time.Time `json:"downloaded_at"`
	External     bool      `json:"external,omitempty"` // true for imported models (e.g. from LM Studio)
}

type registryData struct {
	Models []*ModelEntry `json:"models"`
}

// ModelRegistry manages model metadata persisted as JSON.
type ModelRegistry struct {
	mu       sync.RWMutex
	path     string
	models   map[string]*ModelEntry
}

func NewModelRegistry(modelsDir string) (*ModelRegistry, error) {
	path := filepath.Join(modelsDir, "registry.json")
	r := &ModelRegistry{
		path:   path,
		models: make(map[string]*ModelEntry),
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return r, nil
		}
		return nil, fmt.Errorf("read registry: %w", err)
	}

	var rd registryData
	if err := json.Unmarshal(data, &rd); err != nil {
		return nil, fmt.Errorf("parse registry: %w", err)
	}

	for _, m := range rd.Models {
		r.models[m.ID] = m
	}
	return r, nil
}

func (r *ModelRegistry) List() []*ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*ModelEntry, 0, len(r.models))
	for _, m := range r.models {
		result = append(result, m)
	}
	return result
}

func (r *ModelRegistry) Get(id string) *ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.models[id]
}

func (r *ModelRegistry) Add(entry *ModelEntry) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models[entry.ID] = entry
	return r.save()
}

func (r *ModelRegistry) Update(entry *ModelEntry) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[entry.ID]; !ok {
		return fmt.Errorf("model %s not found", entry.ID)
	}
	r.models[entry.ID] = entry
	return r.save()
}

func (r *ModelRegistry) Delete(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[id]; !ok {
		return fmt.Errorf("model %s not found", id)
	}
	delete(r.models, id)
	return r.save()
}

// save writes the registry to disk atomically (write to .tmp, then rename).
func (r *ModelRegistry) save() error {
	entries := make([]*ModelEntry, 0, len(r.models))
	for _, m := range r.models {
		entries = append(entries, m)
	}
	rd := registryData{Models: entries}

	data, err := json.MarshalIndent(rd, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal registry: %w", err)
	}

	tmpPath := r.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("write registry tmp: %w", err)
	}
	if err := os.Rename(tmpPath, r.path); err != nil {
		return fmt.Errorf("rename registry: %w", err)
	}
	return nil
}

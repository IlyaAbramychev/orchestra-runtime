package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/storage"
)

// DownloadState tracks an in-progress download.
type DownloadState struct {
	ModelID         string
	TotalBytes      int64
	DownloadedBytes atomic.Int64
	SpeedBPS        atomic.Int64
	Error           error
	Cancel          context.CancelFunc
	Done            chan struct{}
}

// ModelManager handles model CRUD and lifecycle.
type ModelManager struct {
	registry  *storage.ModelRegistry
	engine    engine.Backend
	modelsDir string
	downloads sync.Map // id -> *DownloadState
}

func NewModelManager(registry *storage.ModelRegistry, eng engine.Backend, modelsDir string) *ModelManager {
	return &ModelManager{
		registry:  registry,
		engine:    eng,
		modelsDir: modelsDir,
	}
}

// List returns all models in the registry.
func (m *ModelManager) List() []*storage.ModelEntry {
	entries := m.registry.List()
	// Update status for the loaded model
	loadedID := m.engine.LoadedModelID()
	for _, e := range entries {
		if e.ID == loadedID && e.Status == "ready" {
			e.Status = "loaded"
		}
	}
	return entries
}

// Get returns a single model entry.
func (m *ModelManager) Get(id string) *storage.ModelEntry {
	entry := m.registry.Get(id)
	if entry != nil && entry.ID == m.engine.LoadedModelID() && entry.Status == "ready" {
		entry.Status = "loaded"
	}
	return entry
}

// PullModel downloads a model from a URL.
func (m *ModelManager) PullModel(name, sourceURL string) (string, error) {
	if sourceURL == "" {
		return "", fmt.Errorf("source_url is required")
	}

	id := uuid.New().String()
	filename := extractFilename(sourceURL)
	if filename == "" {
		filename = name + ".gguf"
	}

	meta := parseModelMetadata(filename)

	entry := &storage.ModelEntry{
		ID:           id,
		Name:         name,
		Filename:     filename,
		SourceURL:    sourceURL,
		Quantization: meta.quantization,
		Family:       meta.family,
		Parameters:   meta.parameters,
		Status:       "downloading",
		FilePath:     filepath.Join(m.modelsDir, filename),
		DownloadedAt: time.Now().UTC(),
	}

	if err := m.registry.Add(entry); err != nil {
		return "", fmt.Errorf("add to registry: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	ds := &DownloadState{
		ModelID: id,
		Cancel:  cancel,
		Done:    make(chan struct{}),
	}
	m.downloads.Store(id, ds)

	go m.downloadModel(ctx, entry, ds)

	return id, nil
}

func (m *ModelManager) downloadModel(ctx context.Context, entry *storage.ModelEntry, ds *DownloadState) {
	defer close(ds.Done)

	slog.Info("starting model download", "id", entry.ID, "url", entry.SourceURL, "path", entry.FilePath)

	partPath := entry.FilePath + ".part"
	out, err := os.Create(partPath)
	if err != nil {
		m.failDownload(entry, ds, fmt.Errorf("create file: %w", err))
		return
	}
	defer out.Close()

	req, err := http.NewRequestWithContext(ctx, "GET", entry.SourceURL, nil)
	if err != nil {
		m.failDownload(entry, ds, fmt.Errorf("create request: %w", err))
		return
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		m.failDownload(entry, ds, fmt.Errorf("download: %w", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		m.failDownload(entry, ds, fmt.Errorf("HTTP %d", resp.StatusCode))
		return
	}

	ds.TotalBytes = resp.ContentLength
	hasher := sha256.New()
	reader := io.TeeReader(resp.Body, hasher)

	buf := make([]byte, 256*1024) // 256KB buffer
	var downloaded int64
	lastSpeedUpdate := time.Now()
	var lastBytes int64

	for {
		n, readErr := reader.Read(buf)
		if n > 0 {
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				m.failDownload(entry, ds, fmt.Errorf("write: %w", writeErr))
				return
			}
			downloaded += int64(n)
			ds.DownloadedBytes.Store(downloaded)

			if time.Since(lastSpeedUpdate) > time.Second {
				speed := downloaded - lastBytes
				ds.SpeedBPS.Store(speed)
				lastBytes = downloaded
				lastSpeedUpdate = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			m.failDownload(entry, ds, fmt.Errorf("read: %w", readErr))
			return
		}
	}

	out.Close()

	// Rename .part to final
	if err := os.Rename(partPath, entry.FilePath); err != nil {
		m.failDownload(entry, ds, fmt.Errorf("rename: %w", err))
		return
	}

	// Update registry
	entry.Size = downloaded
	entry.SHA256 = hex.EncodeToString(hasher.Sum(nil))
	entry.Status = "ready"
	entry.DownloadedAt = time.Now().UTC()

	if err := m.registry.Update(entry); err != nil {
		slog.Error("update registry after download", "error", err)
	}

	m.downloads.Delete(entry.ID)
	slog.Info("model downloaded", "id", entry.ID, "size", downloaded, "sha256", entry.SHA256)
}

func (m *ModelManager) failDownload(entry *storage.ModelEntry, ds *DownloadState, err error) {
	slog.Error("model download failed", "id", entry.ID, "error", err)
	ds.Error = err
	entry.Status = "error"
	entry.ErrorMessage = err.Error()
	m.registry.Update(entry)
}

// GetDownloadState returns download progress for a model.
func (m *ModelManager) GetDownloadState(id string) *DownloadState {
	if v, ok := m.downloads.Load(id); ok {
		return v.(*DownloadState)
	}
	return nil
}

// DeleteModel removes a model from disk and registry.
func (m *ModelManager) DeleteModel(id string) error {
	entry := m.registry.Get(id)
	if entry == nil {
		return fmt.Errorf("model %s not found", id)
	}

	// Cancel active download if any
	if ds := m.GetDownloadState(id); ds != nil {
		ds.Cancel()
		<-ds.Done
		m.downloads.Delete(id)
	}

	// Unload if currently loaded
	if m.engine.LoadedModelID() == id {
		m.engine.UnloadModel()
	}

	// Remove file (only if not externally imported, e.g. from LM Studio)
	if !entry.External {
		os.Remove(entry.FilePath)
		os.Remove(entry.FilePath + ".part")
	}

	return m.registry.Delete(id)
}

// ImportFromDirectory scans a directory for .gguf files and registers them.
// Existing imports (by FilePath) are skipped.
func (m *ModelManager) ImportFromDirectory(dir string) ([]*storage.ModelEntry, error) {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve path: %w", err)
	}

	if info, err := os.Stat(absDir); err != nil {
		return nil, fmt.Errorf("stat dir: %w", err)
	} else if !info.IsDir() {
		return nil, fmt.Errorf("not a directory: %s", absDir)
	}

	// Build set of already-registered paths
	existing := make(map[string]bool)
	for _, e := range m.registry.List() {
		existing[e.FilePath] = true
	}

	var imported []*storage.ModelEntry

	err = filepath.Walk(absDir, func(path string, info os.FileInfo, werr error) error {
		if werr != nil {
			return nil // skip unreadable entries
		}
		if info.IsDir() {
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
			return nil
		}
		// Skip multimodal projection files — they are not standalone models
		if strings.HasPrefix(strings.ToLower(info.Name()), "mmproj") {
			return nil
		}
		if existing[path] {
			return nil
		}

		filename := info.Name()
		meta := parseModelMetadata(filename)

		// Derive a friendly name from the directory structure:
		// .../author/model-name/file.gguf  →  "author/model-name"
		name := deriveModelName(absDir, path, filename)

		entry := &storage.ModelEntry{
			ID:           uuid.New().String(),
			Name:         name,
			Filename:     filename,
			Size:         info.Size(),
			Quantization: meta.quantization,
			Family:       meta.family,
			Parameters:   meta.parameters,
			SourceURL:    "file://" + path,
			Status:       "ready",
			FilePath:     path,
			DownloadedAt: info.ModTime().UTC(),
			External:     true,
		}

		if err := m.registry.Add(entry); err != nil {
			slog.Warn("failed to register imported model", "path", path, "error", err)
			return nil
		}

		imported = append(imported, entry)
		slog.Info("imported model", "name", name, "path", path, "size", info.Size())
		return nil
	})

	if err != nil {
		return imported, fmt.Errorf("walk: %w", err)
	}
	return imported, nil
}

// deriveModelName extracts a sensible display name from the file path.
// For LM Studio structure (author/model/file.gguf), returns "author/model".
// Otherwise, returns the filename without extension.
func deriveModelName(rootDir, fullPath, filename string) string {
	rel, err := filepath.Rel(rootDir, fullPath)
	if err != nil {
		return strings.TrimSuffix(filename, ".gguf")
	}
	parts := strings.Split(rel, string(filepath.Separator))
	if len(parts) >= 3 {
		// author/model/file.gguf
		return parts[0] + "/" + parts[1]
	}
	if len(parts) == 2 {
		return parts[0]
	}
	return strings.TrimSuffix(filename, ".gguf")
}

// LoadModel loads a model into the engine.
//
// Safety net: on unified-memory systems (Apple Silicon, CPU-only) a load that
// would exceed available RAM causes swap-thrash and can hang the OS requiring
// a hard reboot. We refuse such loads unless caller explicitly bypasses via
// ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT=1. This is a backstop for the UI check;
// external HTTP clients and older extensions are also protected.
func (m *ModelManager) LoadModel(id string, gpuLayers, ctxSize, threads int) error {
	entry := m.registry.Get(id)
	if entry == nil {
		return fmt.Errorf("model %s not found", id)
	}
	if entry.Status != "ready" && entry.Status != "loaded" {
		return fmt.Errorf("model %s is not ready (status: %s)", id, entry.Status)
	}

	// Stat the file for its on-disk size; gguf mmap occupies roughly that
	// many bytes of RAM at inference time.
	if os.Getenv("ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT") != "1" {
		if st, err := os.Stat(entry.FilePath); err == nil {
			modelBytes := st.Size()
			// Conservative KV estimate: 0.2 MB/token (covers most 8B-32B GQA
			// models). Smaller than UI's tiered estimate but good enough as a
			// safety floor.
			kvBytes := int64(ctxSize) * 200 * 1024
			avail := getAvailableRAM()
			// Keep at least 2 GB headroom for the OS so userspace doesn't die.
			const headroom int64 = 2 * 1024 * 1024 * 1024
			budget := avail - headroom
			needed := modelBytes + kvBytes
			if budget > 0 && needed > budget {
				return fmt.Errorf(
					"load would exceed available RAM: model %.1f GB + KV ~%.1f GB = %.1f GB, "+
						"available %.1f GB (reserved 2 GB for OS). "+
						"Close other apps, lower n_ctx, or set ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT=1 to bypass.",
					float64(modelBytes)/1024/1024/1024,
					float64(kvBytes)/1024/1024/1024,
					float64(needed)/1024/1024/1024,
					float64(avail)/1024/1024/1024,
				)
			}
		}
	}

	return m.engine.LoadModel(id, entry.FilePath, gpuLayers, ctxSize, threads)
}

// UnloadModel unloads the current model.
func (m *ModelManager) UnloadModel() {
	m.engine.UnloadModel()
}

// --- Helpers ---

type modelMeta struct {
	family       string
	parameters   string
	quantization string
}

var (
	quantRe = regexp.MustCompile(`(?i)(q[0-9]+_[a-z0-9_]+|f16|f32|fp16|fp32)`)
	paramRe = regexp.MustCompile(`(?i)(\d+\.?\d*[bB])`)
)

func parseModelMetadata(filename string) modelMeta {
	name := strings.TrimSuffix(strings.ToLower(filename), ".gguf")
	meta := modelMeta{}

	if m := quantRe.FindString(name); m != "" {
		meta.quantization = strings.ToUpper(m)
	}
	if m := paramRe.FindString(name); m != "" {
		meta.parameters = strings.ToUpper(m)
	}

	// Extract family: everything before the first dash-separated parameter/quant token
	parts := strings.Split(name, "-")
	var familyParts []string
	for _, p := range parts {
		if paramRe.MatchString(p) || quantRe.MatchString(p) {
			break
		}
		familyParts = append(familyParts, p)
	}
	if len(familyParts) > 0 {
		meta.family = strings.Join(familyParts, "-")
	}

	return meta
}

func extractFilename(url string) string {
	parts := strings.Split(url, "/")
	if len(parts) == 0 {
		return ""
	}
	name := parts[len(parts)-1]
	if idx := strings.Index(name, "?"); idx >= 0 {
		name = name[:idx]
	}
	return name
}

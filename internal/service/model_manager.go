package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/storage"
)

const (
	modelDownloadMaxAttempts = 3
	modelDownloadBaseBackoff = 500 * time.Millisecond
)

// DownloadState tracks an in-progress download.
type DownloadState struct {
	ModelID         string
	TotalBytes      int64
	DownloadedBytes atomic.Int64
	SpeedBPS        atomic.Int64
	Attempt         atomic.Int32
	MaxAttempts     int
	ResumeFrom      atomic.Int64
	LastError       atomic.Value
	Error           error
	Cancel          context.CancelFunc
	Done            chan struct{}
}

type PullModelMetadata struct {
	Quantization        string
	Family              string
	Parameters          string
	SHA256              string
	MMProjFilename      string
	Template            string
	StopTokens          []string
	Capabilities        *storage.ModelCapabilities
	RecommendedSettings *storage.RecommendedModelSettings
}

type CreateModelMetadata struct {
	Template   string
	System     string
	Parameters map[string]any
	License    []string
}

// ModelManager handles model CRUD and lifecycle.
type ModelManager struct {
	registry           *storage.ModelRegistry
	engine             engine.Backend
	scheduler          *RuntimeScheduler
	modelsDir          string
	defaultLoadOptions engine.LoadOptions
	loadPlanner        *LoadPlanner
	loadMu             sync.Mutex
	pullMu             sync.Mutex
	downloads          sync.Map // id -> *DownloadState
	ollamaPulls        sync.Map // registry ref -> *ollamaPullState
}

func NewModelManager(registry *storage.ModelRegistry, eng engine.Backend, modelsDir string) *ModelManager {
	return &ModelManager{
		registry:           registry,
		engine:             eng,
		modelsDir:          modelsDir,
		defaultLoadOptions: engine.DefaultLoadOptions(),
		loadPlanner:        NewLoadPlanner(),
	}
}

func NewModelManagerWithScheduler(registry *storage.ModelRegistry, scheduler *RuntimeScheduler, modelsDir string) *ModelManager {
	return &ModelManager{
		registry:           registry,
		engine:             scheduler.Backend(),
		scheduler:          scheduler,
		modelsDir:          modelsDir,
		defaultLoadOptions: engine.DefaultLoadOptions(),
		loadPlanner:        NewLoadPlanner(),
	}
}

func (m *ModelManager) SetDefaultLoadOptions(opts engine.LoadOptions) {
	m.defaultLoadOptions = opts
}

func (m *ModelManager) DefaultLoadOptions() engine.LoadOptions {
	return m.defaultLoadOptions
}

// DefaultLoadOptionsForModel returns the zero-configuration profile used by
// the UI's one-click load and by request-scoped auto-load. A runtime-wide
// context setting is respected once the user has changed it; otherwise a
// model's recommended context provides a better out-of-the-box default.
func (m *ModelManager) DefaultLoadOptionsForModel(id string) engine.LoadOptions {
	opts := m.DefaultLoadOptions()
	entry := m.registry.Get(id)
	entry = cloneModelEntry(entry)
	if entry == nil || entry.RecommendedSettings.ContextSize <= 0 {
		return opts
	}
	if opts.CtxSize == engine.DefaultLoadOptions().CtxSize {
		opts.CtxSize = entry.RecommendedSettings.ContextSize
	}
	return opts
}

// List returns all models in the registry.
func (m *ModelManager) List() []*storage.ModelEntry {
	entries := m.registry.List()
	result := make([]*storage.ModelEntry, 0, len(entries))
	for _, entry := range entries {
		copy := cloneModelEntry(entry)
		if copy.Status == "ready" {
			if status := m.RuntimeStatus(copy.ID); status != "" {
				copy.Status = status
			}
		}
		result = append(result, copy)
	}
	return result
}

// Get returns a single model entry.
func (m *ModelManager) Get(id string) *storage.ModelEntry {
	entry := m.registry.Get(id)
	if entry == nil {
		return nil
	}
	copy := cloneModelEntry(entry)
	if copy.Status == "ready" {
		if status := m.RuntimeStatus(copy.ID); status != "" {
			copy.Status = status
		}
	}
	return copy
}

func (m *ModelManager) RuntimeStatus(id string) string {
	if m.scheduler != nil {
		if m.scheduler.ActiveModelID() == id {
			return m.scheduler.State()
		}
	}
	if id == m.engine.LoadedModelID() && m.engine.IsLoaded() {
		return "loaded"
	}
	return ""
}

func (m *ModelManager) RuntimeSnapshot() RuntimeSnapshot {
	if m.scheduler != nil {
		return m.scheduler.Snapshot()
	}
	return RuntimeSnapshot{
		State:         m.engine.State(),
		ActiveModelID: m.engine.LoadedModelID(),
	}
}

func cloneModelEntry(entry *storage.ModelEntry) *storage.ModelEntry {
	if entry == nil {
		return nil
	}
	copy := *entry
	normalizeModelMetadata(&copy)
	return &copy
}

func normalizeModelMetadata(entry *storage.ModelEntry) {
	if entry.MMProjFilename == "" {
		if discovery := discoverMMProjFilename(entry.FilePath); discovery.filename != "" {
			entry.MMProjFilename = discovery.filename
		}
	}
	metadataAvailable := enrichModelEntryFromGGUF(entry)
	if !metadataAvailable && entry.Capabilities == (storage.ModelCapabilities{}) {
		entry.Capabilities = inferModelCapabilities(entry.Name, entry.Filename)
	}
	if entry.MMProjFilename != "" {
		entry.Capabilities.Vision = true
	}
	if entry.RecommendedSettings == (storage.RecommendedModelSettings{}) {
		entry.RecommendedSettings = inferRecommendedSettings(modelMeta{parameters: entry.Parameters})
	}
	if entry.TrainingContext > 0 && entry.RecommendedSettings.ContextSize > entry.TrainingContext {
		entry.RecommendedSettings.ContextSize = entry.TrainingContext
	}
}

// ResolveModel finds a model by registry id, display name, filename, or
// filename stem. This mirrors how Ollama/OpenAI clients usually address
// models while keeping registry IDs stable for internal management.
func (m *ModelManager) ResolveModel(ref string) (*storage.ModelEntry, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return nil, fmt.Errorf("model is required")
	}
	if entry := m.registry.Get(ref); entry != nil {
		return entry, nil
	}
	for _, entry := range m.registry.List() {
		if entry.Name == ref || entry.Filename == ref {
			return entry, nil
		}
		if strings.TrimSuffix(entry.Filename, filepath.Ext(entry.Filename)) == ref {
			return entry, nil
		}
	}
	return nil, fmt.Errorf("model %s not found", ref)
}

// ResolveModelID returns the stable registry ID for a request-facing model
// reference. Inference uses it to verify that concurrent auto-loads did not
// switch the backend to a different model before the runtime slot was acquired.
func (m *ModelManager) ResolveModelID(ref string) (string, error) {
	entry, err := m.ResolveModel(ref)
	if err != nil {
		return "", err
	}
	return entry.ID, nil
}

// CopyModel creates a new registry entry that points at the same model artifact
// as source. This mirrors Ollama's local copy/tag behavior without duplicating
// large GGUF files on disk.
func (m *ModelManager) CopyModel(source, destination string) (string, error) {
	source = strings.TrimSpace(source)
	destination = strings.TrimSpace(destination)
	if source == "" {
		return "", fmt.Errorf("source is required")
	}
	if destination == "" {
		return "", fmt.Errorf("destination is required")
	}

	sourceEntry, err := m.ResolveModel(source)
	if err != nil {
		return "", err
	}
	if _, err := m.ResolveModel(destination); err == nil {
		return "", fmt.Errorf("model %s already exists", destination)
	}

	entry := cloneModelEntry(sourceEntry)
	entry.ID = uuid.New().String()
	entry.Name = destination
	entry.StopTokens = append([]string(nil), sourceEntry.StopTokens...)
	entry.License = append([]string(nil), sourceEntry.License...)
	entry.MMProjFilename = sourceEntry.MMProjFilename
	entry.DownloadedAt = time.Now().UTC()
	if entry.Status == "loaded" {
		entry.Status = "ready"
	}
	if err := m.registry.Add(entry); err != nil {
		return "", fmt.Errorf("add copied model: %w", err)
	}
	return entry.ID, nil
}

// CreateModelFromBase creates a derived local model entry from an existing model.
// It updates registry metadata only; the underlying GGUF artifact is shared with
// the base model because Orchestra Runtime does not rebuild GGUF files.
func (m *ModelManager) CreateModelFromBase(name, from string, metadata CreateModelMetadata) (string, error) {
	name = strings.TrimSpace(name)
	from = strings.TrimSpace(from)
	if name == "" {
		return "", fmt.Errorf("model is required")
	}
	if from == "" {
		return "", fmt.Errorf("from is required")
	}

	base, err := m.ResolveModel(from)
	if err != nil {
		return "", err
	}
	if _, err := m.ResolveModel(name); err == nil {
		return "", fmt.Errorf("model %s already exists", name)
	}

	entry := cloneModelEntry(base)
	entry.ID = uuid.New().String()
	entry.Name = name
	entry.SourceURL = "ollama://" + name
	entry.StopTokens = append([]string(nil), base.StopTokens...)
	entry.License = append([]string(nil), base.License...)
	entry.MMProjFilename = base.MMProjFilename
	entry.DownloadedAt = time.Now().UTC()
	if entry.Status == "loaded" {
		entry.Status = "ready"
	}
	if metadata.Template != "" {
		entry.Template = metadata.Template
	}
	if metadata.System != "" {
		entry.System = metadata.System
	}
	if len(metadata.Parameters) > 0 {
		entry.OllamaParameters = renderOllamaParameters(metadata.Parameters)
		if stopTokens := extractStopTokens(metadata.Parameters["stop"]); len(stopTokens) > 0 {
			entry.StopTokens = stopTokens
		}
	}
	if len(metadata.License) > 0 {
		entry.License = append([]string(nil), metadata.License...)
	}
	entry.Modelfile = buildOllamaModelfile(ollamaRegistryRef{Name: from}, entry.Template, entry.System, entry.OllamaParameters, entry.License)

	if err := m.registry.Add(entry); err != nil {
		return "", fmt.Errorf("add created model: %w", err)
	}
	return entry.ID, nil
}

// EnsureLoaded resolves a request model reference and loads it if necessary.
// Concurrent first requests are serialized so the same model is not loaded
// twice under burst traffic.
func (m *ModelManager) EnsureLoaded(ctx context.Context, ref string) error {
	return m.EnsureLoadedFor(ctx, ref, "")
}

func (m *ModelManager) EnsureLoadedFor(ctx context.Context, ref, capability string) error {
	entry, err := m.ResolveModel(ref)
	if err != nil {
		return err
	}
	if err := m.requireModelCapability(entry, capability); err != nil {
		return err
	}
	if m.engine.LoadedModelID() == entry.ID && m.engine.IsLoaded() {
		return nil
	}

	m.loadMu.Lock()
	defer m.loadMu.Unlock()

	entry, err = m.ResolveModel(ref)
	if err != nil {
		return err
	}
	if err := m.requireModelCapability(entry, capability); err != nil {
		return err
	}
	if m.engine.LoadedModelID() == entry.ID && m.engine.IsLoaded() {
		return nil
	}
	return m.LoadModelWithContext(ctx, entry.ID, m.DefaultLoadOptionsForModel(entry.ID))
}

func (m *ModelManager) DefaultsForModel(ref string) (ModelRequestDefaults, error) {
	entry, err := m.ResolveModel(ref)
	if err != nil {
		return ModelRequestDefaults{}, err
	}
	normalized := cloneModelEntry(entry)
	return ModelRequestDefaults{
		StopTokens:   append([]string(nil), normalized.StopTokens...),
		ChatTemplate: normalized.Template,
	}, nil
}

func requireModelCapability(entry *storage.ModelEntry, capability string) error {
	if capability == "" {
		return nil
	}
	normalized := cloneModelEntry(entry)
	switch capability {
	case "chat":
		if normalized.Capabilities.Chat {
			return nil
		}
	case "embeddings":
		if normalized.Capabilities.Embeddings {
			return nil
		}
	case "rerank":
		if normalized.Capabilities.Rerank {
			return nil
		}
	case "tools":
		if normalized.Capabilities.Tools {
			return nil
		}
	case "vision":
		if normalized.Capabilities.Chat && normalized.Capabilities.Vision {
			return nil
		}
	default:
		return fmt.Errorf("unsupported model capability %q", capability)
	}
	return fmt.Errorf("model %s does not support %s", normalized.Name, capability)
}

func (m *ModelManager) requireModelCapability(entry *storage.ModelEntry, capability string) error {
	if capability != "vision" {
		return requireModelCapability(entry, capability)
	}
	if err := requireModelCapability(entry, "chat"); err != nil {
		return err
	}
	if strings.TrimSpace(m.defaultLoadOptions.MMProjPath) != "" {
		return nil
	}
	resolved, err := resolveEntryMMProjPath(entry)
	if err != nil {
		return err
	}
	if resolved == "" {
		return fmt.Errorf("multimodal images require a loaded mmproj")
	}
	return nil
}

// PullModel downloads a model from a URL.
func (m *ModelManager) PullModel(name, sourceURL string) (string, error) {
	return m.PullModelWithMetadata(name, sourceURL, PullModelMetadata{})
}

func (m *ModelManager) PullModelWithMetadata(name, sourceURL string, metadata PullModelMetadata) (string, error) {
	sourceURL = strings.TrimSpace(sourceURL)
	if sourceURL == "" {
		return "", fmt.Errorf("source_url is required")
	}

	filename := extractFilename(sourceURL)
	if filename == "" {
		filename = name + ".gguf"
	}
	filePath := filepath.Join(m.modelsDir, filename)

	m.pullMu.Lock()
	defer m.pullMu.Unlock()

	if existing := m.findExistingPull(sourceURL, filePath); existing != nil {
		return existing.ID, nil
	}

	id := uuid.New().String()
	if failed := m.findFailedPull(sourceURL, filePath); failed != nil {
		id = failed.ID
	}

	meta := parseModelMetadata(filename)
	quantization := firstNonEmpty(metadata.Quantization, meta.quantization)
	family := firstNonEmpty(metadata.Family, meta.family)
	parameters := firstNonEmpty(metadata.Parameters, meta.parameters)
	capabilities := inferModelCapabilities(name, filename)
	if metadata.Capabilities != nil {
		capabilities = *metadata.Capabilities
	}
	settings := inferRecommendedSettings(modelMeta{parameters: parameters})
	if metadata.RecommendedSettings != nil {
		settings = *metadata.RecommendedSettings
	}

	entry := &storage.ModelEntry{
		ID:                  id,
		Name:                name,
		Filename:            filename,
		SourceURL:           sourceURL,
		Quantization:        quantization,
		Family:              family,
		Parameters:          parameters,
		SHA256:              strings.ToLower(strings.TrimSpace(metadata.SHA256)),
		Template:            metadata.Template,
		StopTokens:          append([]string(nil), metadata.StopTokens...),
		Capabilities:        capabilities,
		RecommendedSettings: settings,
		MMProjFilename:      strings.TrimSpace(metadata.MMProjFilename),
		Status:              "downloading",
		FilePath:            filePath,
		DownloadedAt:        time.Now().UTC(),
	}

	if failed := m.registry.Get(id); failed != nil {
		if err := m.registry.Update(entry); err != nil {
			return "", fmt.Errorf("update registry: %w", err)
		}
	} else {
		if err := m.registry.Add(entry); err != nil {
			return "", fmt.Errorf("add to registry: %w", err)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	ds := &DownloadState{
		ModelID:     id,
		MaxAttempts: modelDownloadMaxAttempts,
		Cancel:      cancel,
		Done:        make(chan struct{}),
	}
	m.downloads.Store(id, ds)

	go m.downloadModel(ctx, entry, ds)

	return id, nil
}

func (m *ModelManager) findExistingPull(sourceURL, filePath string) *storage.ModelEntry {
	for _, entry := range m.registry.List() {
		if entry.Status == "error" {
			continue
		}
		if entry.SourceURL == sourceURL || entry.FilePath == filePath {
			return entry
		}
	}
	return nil
}

func (m *ModelManager) findFailedPull(sourceURL, filePath string) *storage.ModelEntry {
	for _, entry := range m.registry.List() {
		if entry.Status != "error" {
			continue
		}
		if entry.SourceURL == sourceURL || entry.FilePath == filePath {
			return entry
		}
	}
	return nil
}

func (m *ModelManager) downloadModel(ctx context.Context, entry *storage.ModelEntry, ds *DownloadState) {
	defer close(ds.Done)

	slog.Info("starting model download", "id", entry.ID, "url", entry.SourceURL, "path", entry.FilePath)

	var err error
	for attempt := 1; attempt <= modelDownloadMaxAttempts; attempt++ {
		ds.Attempt.Store(int32(attempt))
		err = m.downloadModelAttempt(ctx, entry, ds)
		if err == nil {
			return
		}
		ds.setLastError(err)
		if ctx.Err() != nil || !isRetryableDownloadError(err) || attempt == modelDownloadMaxAttempts {
			m.failDownload(entry, ds, err)
			return
		}

		backoff := modelDownloadBaseBackoff * time.Duration(attempt)
		slog.Warn("model download attempt failed; retrying", "id", entry.ID, "attempt", attempt, "backoff", backoff, "error", err)
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			m.failDownload(entry, ds, ctx.Err())
			return
		case <-timer.C:
		}
	}
}

func (m *ModelManager) downloadModelAttempt(ctx context.Context, entry *storage.ModelEntry, ds *DownloadState) error {
	partPath := entry.FilePath + ".part"
	var resumeFrom int64
	if st, err := os.Stat(partPath); err == nil && st.Size() > 0 {
		resumeFrom = st.Size()
	}
	ds.ResumeFrom.Store(resumeFrom)

	req, err := http.NewRequestWithContext(ctx, "GET", entry.SourceURL, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	if resumeFrom > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeFrom))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return retryableDownloadError{err: fmt.Errorf("download: %w", err)}
	}

	if resumeFrom > 0 {
		switch resp.StatusCode {
		case http.StatusPartialContent:
			// Server accepted the range; keep the existing .part bytes.
		case http.StatusOK:
			// Server ignored Range; restart from scratch.
			resumeFrom = 0
		case http.StatusRequestedRangeNotSatisfiable:
			// Stale/corrupt .part size. Restart cleanly.
			resp.Body.Close()
			_ = os.Remove(partPath)
			resumeFrom = 0
			req, err = http.NewRequestWithContext(ctx, "GET", entry.SourceURL, nil)
			if err != nil {
				return fmt.Errorf("create request: %w", err)
			}
			resp, err = http.DefaultClient.Do(req)
			if err != nil {
				return retryableDownloadError{err: fmt.Errorf("download: %w", err)}
			}
			if resp.StatusCode != http.StatusOK {
				resp.Body.Close()
				return downloadHTTPError(resp.StatusCode)
			}
		default:
			resp.Body.Close()
			return downloadHTTPError(resp.StatusCode)
		}
	} else if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return downloadHTTPError(resp.StatusCode)
	}
	defer resp.Body.Close()

	hasher := sha256.New()
	if resumeFrom > 0 {
		existing, err := os.Open(partPath)
		if err != nil {
			return fmt.Errorf("open partial file: %w", err)
		}
		if _, err := io.Copy(hasher, existing); err != nil {
			existing.Close()
			return fmt.Errorf("hash partial file: %w", err)
		}
		existing.Close()
	}

	out, err := os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open partial file: %w", err)
	}
	if resumeFrom == 0 {
		if err := out.Truncate(0); err != nil {
			out.Close()
			return fmt.Errorf("truncate partial file: %w", err)
		}
	}
	defer out.Close()

	if resp.ContentLength > 0 {
		ds.TotalBytes = resp.ContentLength + resumeFrom
	}
	reader := io.TeeReader(resp.Body, hasher)

	buf := make([]byte, 256*1024) // 256KB buffer
	downloaded := resumeFrom
	ds.DownloadedBytes.Store(downloaded)
	lastSpeedUpdate := time.Now()
	lastBytes := downloaded

	for {
		n, readErr := reader.Read(buf)
		if n > 0 {
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("write: %w", writeErr)
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
			return retryableDownloadError{err: fmt.Errorf("read: %w", readErr)}
		}
	}

	out.Close()

	actualSHA := hex.EncodeToString(hasher.Sum(nil))
	if entry.SHA256 != "" && !strings.EqualFold(entry.SHA256, actualSHA) {
		os.Remove(partPath)
		return fmt.Errorf("sha256 mismatch: expected %s, got %s", entry.SHA256, actualSHA)
	}

	// Rename .part to final
	if err := os.Rename(partPath, entry.FilePath); err != nil {
		return fmt.Errorf("rename: %w", err)
	}

	// Update registry
	entry.Size = downloaded
	entry.SHA256 = actualSHA
	entry.Status = "ready"
	entry.DownloadedAt = time.Now().UTC()
	normalizeModelMetadata(entry)

	if err := m.registry.Update(entry); err != nil {
		slog.Error("update registry after download", "error", err)
	}

	m.downloads.Delete(entry.ID)
	slog.Info("model downloaded", "id", entry.ID, "size", downloaded, "sha256", entry.SHA256)
	return nil
}

type retryableDownloadError struct {
	err error
}

func (e retryableDownloadError) Error() string {
	return e.err.Error()
}

func (e retryableDownloadError) Unwrap() error {
	return e.err
}

type downloadHTTPError int

func (e downloadHTTPError) Error() string {
	return fmt.Sprintf("HTTP %d", int(e))
}

func isRetryableDownloadError(err error) bool {
	var retryable retryableDownloadError
	if errors.As(err, &retryable) {
		return true
	}
	var httpErr downloadHTTPError
	if errors.As(err, &httpErr) {
		statusCode := int(httpErr)
		return statusCode == http.StatusRequestTimeout || statusCode == http.StatusTooManyRequests || statusCode >= http.StatusInternalServerError
	}
	return false
}

func (ds *DownloadState) setLastError(err error) {
	if err != nil {
		ds.LastError.Store(err.Error())
	}
}

func (ds *DownloadState) LastErrorMessage() string {
	if value := ds.LastError.Load(); value != nil {
		if msg, ok := value.(string); ok {
			return msg
		}
	}
	return ""
}

func (m *ModelManager) failDownload(entry *storage.ModelEntry, ds *DownloadState, err error) {
	slog.Error("model download failed", "id", entry.ID, "error", err)
	ds.setLastError(err)
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
		if err := m.unloadCurrentModel(context.Background()); err != nil {
			return err
		}
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
		if isMMProjFilename(info.Name()) {
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

		discovery := discoverMMProjFilename(path)
		if len(discovery.ambiguous) > 0 {
			slog.Warn("multiple mmproj files found while importing model; manual selection required",
				"name", name,
				"model_path", path,
				"mmproj_candidates", discovery.ambiguous,
			)
		}

		entry := &storage.ModelEntry{
			ID:                  uuid.New().String(),
			Name:                name,
			Filename:            filename,
			Size:                info.Size(),
			Quantization:        meta.quantization,
			Family:              meta.family,
			Parameters:          meta.parameters,
			Capabilities:        inferModelCapabilities(name, filename),
			RecommendedSettings: inferRecommendedSettings(meta),
			MMProjFilename:      discovery.filename,
			SourceURL:           "file://" + path,
			Status:              "ready",
			FilePath:            path,
			DownloadedAt:        info.ModTime().UTC(),
			External:            true,
		}
		normalizeModelMetadata(entry)

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

// LoadModel loads a model into the engine with the given options.
//
// Safety net: on unified-memory systems (Apple Silicon, CPU-only) a load that
// would exceed available RAM causes swap-thrash and can hang the OS requiring
// a hard reboot. We refuse such loads unless caller explicitly bypasses via
// ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT=1. This is a backstop for the UI check;
// external HTTP clients and older extensions are also protected.
func (m *ModelManager) LoadModel(id string, opts engine.LoadOptions) error {
	return m.LoadModelWithContext(context.Background(), id, opts)
}

func (m *ModelManager) LoadModelWithContext(ctx context.Context, id string, opts engine.LoadOptions) error {
	entry := m.registry.Get(id)
	if entry == nil {
		return fmt.Errorf("model %s not found", id)
	}
	if entry.Status != "ready" && entry.Status != "loaded" {
		return fmt.Errorf("model %s is not ready (status: %s)", id, entry.Status)
	}
	var err error
	opts, err = m.resolveModelLoadOptions(entry, opts)
	if err != nil {
		return err
	}
	projectorBytes, err := mmprojSize(opts.MMProjPath)
	if err != nil {
		return err
	}
	normalizedEntry := cloneModelEntry(entry)

	modelBytes := int64(0)
	if st, statErr := os.Stat(entry.FilePath); statErr == nil {
		modelBytes = st.Size()
	}
	planner := m.loadPlanner
	if planner == nil {
		planner = NewLoadPlanner()
	}
	plan, err := planner.Plan(LoadPlanRequest{
		Options:         opts,
		ModelBytes:      modelBytes,
		ProjectorBytes:  projectorBytes,
		Family:          modelFamily(normalizedEntry),
		Vision:          projectorBytes > 0 || normalizedEntry.Capabilities.Vision,
		TrainingContext: normalizedEntry.TrainingContext,
		AllowOvercommit: os.Getenv("ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT") == "1",
	})
	if err != nil {
		return err
	}
	selected := plan.Attempts[0]
	slog.Info("model load plan",
		"id", id,
		"attempts", len(plan.Attempts),
		"adjustment", selected.Adjustment,
		"ctx_size", selected.Options.CtxSize,
		"batch_size", selected.Options.BatchSize,
		"type_k", normalizeKVName(selected.Options.TypeK),
		"type_v", normalizeKVName(selected.Options.TypeV),
		"estimated_gb", bytesInGiB(selected.Estimate.TotalBytes),
		"available_gb", bytesInGiB(plan.AvailableMemory),
		"budget_gb", bytesInGiB(plan.SafetyBudget),
	)

	attempts := make([]engine.LoadOptions, 0, len(plan.Attempts))
	for _, attempt := range plan.Attempts {
		attempts = append(attempts, attempt.Options)
	}
	if m.scheduler != nil {
		_, err = m.scheduler.LoadModelAttempts(ctx, id, entry.FilePath, attempts)
		return err
	}
	_, err = loadModelWithAttempts(ctx, m.engine, id, entry.FilePath, attempts)
	return err
}

func mmprojSize(path string) (int64, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return 0, nil
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0, fmt.Errorf("configured mmproj %q was not found: %w", path, err)
	}
	if info.IsDir() {
		return 0, fmt.Errorf("configured mmproj %q is a directory", path)
	}
	if !strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
		return 0, fmt.Errorf("configured mmproj %q is not a GGUF file", path)
	}
	return info.Size(), nil
}

func modelFamily(entry *storage.ModelEntry) string {
	if entry == nil {
		return ""
	}
	return firstNonEmpty(entry.Family, entry.Name, entry.Filename)
}

// UnloadModel unloads the current model.
func (m *ModelManager) UnloadModel() {
	_ = m.unloadCurrentModel(context.Background())
}

func (m *ModelManager) unloadCurrentModel(ctx context.Context) error {
	if m.scheduler != nil {
		return m.scheduler.UnloadModel(ctx)
	}
	m.engine.UnloadModel()
	return nil
}

func (m *ModelManager) resolveModelLoadOptions(entry *storage.ModelEntry, opts engine.LoadOptions) (engine.LoadOptions, error) {
	if strings.TrimSpace(opts.MMProjPath) != "" || entry == nil {
		return opts, nil
	}
	resolved, err := resolveEntryMMProjPath(entry)
	if err != nil {
		return opts, err
	}
	if resolved != "" {
		opts.MMProjPath = resolved
	}
	return opts, nil
}

func resolveEntryMMProjPath(entry *storage.ModelEntry) (string, error) {
	if entry == nil || strings.TrimSpace(entry.FilePath) == "" {
		return "", nil
	}
	modelDir := filepath.Dir(entry.FilePath)
	configured := strings.TrimSpace(entry.MMProjFilename)
	if configured != "" {
		resolved := configured
		if !filepath.IsAbs(resolved) {
			resolved = filepath.Join(modelDir, configured)
		}
		if info, err := os.Stat(resolved); err == nil && !info.IsDir() {
			return resolved, nil
		}
		return "", fmt.Errorf("configured mmproj %q for model %s was not found", configured, entry.Name)
	}
	discovery := discoverMMProjFilename(entry.FilePath)
	if discovery.filename != "" {
		return filepath.Join(modelDir, discovery.filename), nil
	}
	if len(discovery.ambiguous) > 0 {
		return "", fmt.Errorf(
			"multiple mmproj files found for model %s: %s; set mmproj_filename in registry metadata or ORCHESTRA_MMPROJ_PATH explicitly",
			entry.Name,
			strings.Join(discovery.ambiguous, ", "),
		)
	}
	return "", nil
}

type mmprojDiscovery struct {
	filename  string
	ambiguous []string
}

func discoverMMProjFilename(modelPath string) mmprojDiscovery {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return mmprojDiscovery{}
	}
	dir := filepath.Dir(modelPath)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return mmprojDiscovery{}
	}
	modelBase := strings.ToLower(filepath.Base(modelPath))
	modelStem := strings.TrimSuffix(modelBase, filepath.Ext(modelBase))

	type candidate struct {
		name  string
		score int
	}
	var candidates []candidate
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		lower := strings.ToLower(name)
		if lower == modelBase || !strings.HasSuffix(lower, ".gguf") || !isMMProjFilename(name) {
			continue
		}
		score := 1
		stem := strings.TrimSuffix(lower, filepath.Ext(lower))
		if strings.Contains(stem, modelStem) || strings.Contains(modelStem, stem) {
			score = 3
		}
		candidates = append(candidates, candidate{name: name, score: score})
	}
	if len(candidates) == 0 {
		return mmprojDiscovery{}
	}
	best := candidates[0]
	bestNames := []string{best.name}
	for _, candidate := range candidates[1:] {
		if candidate.score > best.score {
			best = candidate
			bestNames = []string{candidate.name}
			continue
		}
		if candidate.score == best.score {
			bestNames = append(bestNames, candidate.name)
		}
	}
	if len(bestNames) > 1 {
		sort.Strings(bestNames)
		return mmprojDiscovery{ambiguous: bestNames}
	}
	return mmprojDiscovery{filename: best.name}
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

func inferModelCapabilities(name, filename string) storage.ModelCapabilities {
	combined := strings.ToLower(name + " " + filename)
	isEmbedding := strings.Contains(combined, "embed") ||
		strings.Contains(combined, "bge-") ||
		strings.Contains(combined, "e5-") ||
		strings.Contains(combined, "gte-")
	return storage.ModelCapabilities{
		Chat:       !isEmbedding,
		Embeddings: isEmbedding,
		Rerank:     strings.Contains(combined, "rerank"),
		Tools:      false,
		Thinking:   strings.Contains(combined, "qwen3") || strings.Contains(combined, "deepseek-r1"),
	}
}

func isMMProjFilename(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	return strings.HasSuffix(lower, ".gguf") && strings.Contains(lower, "mmproj")
}

func inferRecommendedSettings(meta modelMeta) storage.RecommendedModelSettings {
	settings := storage.RecommendedModelSettings{ContextSize: 4096}
	switch strings.ToUpper(meta.parameters) {
	case "1B", "1.5B", "2B", "3B", "4B":
		settings.ContextSize = 8192
	case "7B", "8B", "9B":
		settings.ContextSize = 8192
	case "14B", "27B", "32B":
		settings.ContextSize = 4096
	}
	return settings
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
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

// kvBytesPerTokenForModel estimates KV-cache bytes per token in fp16.
//
// File-size tiers are only a fallback: quantization changes file size without
// changing KV dimensions, and hybrid architectures may allocate KV on only a
// subset of layers. Qwen3.5 uses full attention every fourth layer; measured
// llama.cpp allocations are 32 KiB/token for 9B and 64 KiB/token for 27B,
// exactly half of their respective fallback tiers.
func kvBytesPerTokenForModel(modelBytes int64, family string) int {
	const GB = int64(1024) * 1024 * 1024
	var estimate int
	switch {
	case modelBytes < 8*GB:
		estimate = 64 * 1024 // 7-9B class
	case modelBytes < 20*GB:
		estimate = 128 * 1024 // 14-27B quantized class
	case modelBytes < 45*GB:
		estimate = 256 * 1024 // 32B class
	default:
		estimate = 384 * 1024 // 70B+
	}

	if strings.Contains(normalizeModelFamily(family), "qwen35") {
		estimate /= 2
	}
	return estimate
}

func normalizeModelFamily(family string) string {
	value := strings.ToLower(strings.TrimSpace(family))
	value = strings.ReplaceAll(value, ".", "")
	value = strings.ReplaceAll(value, "-", "")
	value = strings.ReplaceAll(value, "_", "")
	return value
}

func kvQuantFactor(kind string) float64 {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "q8_0":
		return 0.5
	case "q5_0", "q5_1":
		return 5.0 / 16.0
	case "q4_0", "q4_1":
		return 0.25
	default:
		return 1.0 // fp16 / unknown
	}
}

package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/storage"
)

const ollamaDefaultRegistry = "registry.ollama.ai"

type OllamaPullProgress struct {
	Status    string
	Digest    string
	Total     int64
	Completed int64
}

type ollamaPullState struct {
	done chan struct{}

	mu        sync.Mutex
	callbacks []func(OllamaPullProgress)
	id        string
	err       error
}

type ollamaRegistryRef struct {
	Original string
	Scheme   string
	Host     string
	Repo     string
	Tag      string
	Name     string
}

func (r ollamaRegistryRef) cacheKey() string {
	return r.Scheme + "://" + r.Host + "/" + r.Repo + ":" + r.Tag
}

type ollamaManifest struct {
	SchemaVersion int           `json:"schemaVersion"`
	MediaType     string        `json:"mediaType"`
	Config        ollamaLayer   `json:"config"`
	Layers        []ollamaLayer `json:"layers"`
}

type ollamaLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
}

// PullOllamaLibraryModel pulls an Ollama registry model by name. It implements
// the registry manifest/blob path used by Ollama while registering the resolved
// GGUF model layer in Orchestra's local model registry.
func (m *ModelManager) PullOllamaLibraryModel(
	ctx context.Context,
	ref string,
	insecure bool,
	progress func(OllamaPullProgress),
) (string, error) {
	parsed, err := parseOllamaRegistryRef(ref, insecure)
	if err != nil {
		return "", err
	}
	if progress == nil {
		progress = func(OllamaPullProgress) {}
	}

	key := parsed.cacheKey()
	m.pullMu.Lock()
	if existing := m.findExistingOllamaPull(parsed.Name); existing != nil {
		m.pullMu.Unlock()
		progress(OllamaPullProgress{Status: "success"})
		return existing.ID, nil
	}
	if active, ok := m.ollamaPulls.Load(key); ok {
		state := active.(*ollamaPullState)
		state.subscribe(progress)
		m.pullMu.Unlock()

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-state.done:
			return state.result()
		}
	}

	state := &ollamaPullState{done: make(chan struct{})}
	state.subscribe(progress)
	m.ollamaPulls.Store(key, state)
	m.pullMu.Unlock()
	defer func() {
		m.ollamaPulls.Delete(key)
		close(state.done)
	}()

	state.emit(OllamaPullProgress{Status: "pulling manifest"})
	manifest, err := m.fetchOllamaManifest(ctx, parsed)
	if err != nil {
		state.err = fmt.Errorf("pull model manifest: %w", err)
		return "", state.err
	}

	layers := append([]ollamaLayer(nil), manifest.Layers...)
	if manifest.Config.Digest != "" {
		layers = append(layers, manifest.Config)
	}

	var modelLayer *ollamaLayer
	layerPaths := make(map[string]string, len(layers))
	for i := range layers {
		layer := layers[i]
		path, err := m.downloadOllamaBlob(ctx, parsed, layer, state.emit)
		if err != nil {
			state.err = err
			return "", err
		}
		layerPaths[layer.Digest] = path
		if layer.MediaType == "application/vnd.ollama.image.model" {
			copyLayer := layer
			modelLayer = &copyLayer
		}
	}
	if modelLayer == nil {
		state.err = fmt.Errorf("manifest does not contain a GGUF model layer")
		return "", state.err
	}

	state.emit(OllamaPullProgress{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if err := verifyDigestFile(layer.Digest, layerPaths[layer.Digest]); err != nil {
			state.err = err
			return "", err
		}
	}

	entry := m.ollamaManifestEntry(parsed, manifest, *modelLayer, layerPaths)
	if existing := m.findExistingOllamaPull(parsed.Name); existing != nil {
		entry.ID = existing.ID
		if err := m.registry.Update(entry); err != nil {
			state.err = fmt.Errorf("update registry: %w", err)
			return "", state.err
		}
	} else if err := m.registry.Add(entry); err != nil {
		state.err = fmt.Errorf("add to registry: %w", err)
		return "", state.err
	}

	state.id = entry.ID
	state.emit(OllamaPullProgress{Status: "writing manifest"})
	state.emit(OllamaPullProgress{Status: "success"})
	return entry.ID, nil
}

func (s *ollamaPullState) subscribe(fn func(OllamaPullProgress)) {
	if fn == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.callbacks = append(s.callbacks, fn)
}

func (s *ollamaPullState) emit(progress OllamaPullProgress) {
	s.mu.Lock()
	callbacks := append([]func(OllamaPullProgress){}, s.callbacks...)
	s.mu.Unlock()
	for _, fn := range callbacks {
		fn(progress)
	}
}

func (s *ollamaPullState) result() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.id, s.err
}

func (m *ModelManager) findExistingOllamaPull(name string) *storage.ModelEntry {
	for _, entry := range m.registry.List() {
		if entry.Name == name && strings.HasPrefix(entry.SourceURL, "ollama://") && entry.Status != "error" {
			return entry
		}
	}
	return nil
}

func parseOllamaRegistryRef(ref string, insecure bool) (ollamaRegistryRef, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return ollamaRegistryRef{}, fmt.Errorf("model is required")
	}

	scheme := "https"
	if insecure {
		scheme = "http"
	}
	if strings.Contains(ref, "://") {
		u, err := url.Parse(ref)
		if err != nil {
			return ollamaRegistryRef{}, fmt.Errorf("parse model reference: %w", err)
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return ollamaRegistryRef{}, fmt.Errorf("unsupported model reference scheme %q", u.Scheme)
		}
		scheme = u.Scheme
		ref = strings.TrimPrefix(strings.Trim(u.Host+u.Path, "/"), "/")
	}

	refNoTag := ref
	tag := "latest"
	lastSlash := strings.LastIndex(refNoTag, "/")
	if lastColon := strings.LastIndex(refNoTag, ":"); lastColon > lastSlash {
		tag = refNoTag[lastColon+1:]
		refNoTag = refNoTag[:lastColon]
	}
	if tag == "" {
		return ollamaRegistryRef{}, fmt.Errorf("model tag is empty")
	}

	parts := strings.Split(refNoTag, "/")
	host := ollamaDefaultRegistry
	repoParts := parts
	if len(parts) > 1 && isOllamaRegistryHost(parts[0]) {
		host = parts[0]
		repoParts = parts[1:]
	}
	if len(repoParts) == 0 || repoParts[0] == "" {
		return ollamaRegistryRef{}, fmt.Errorf("model name is required")
	}
	if len(repoParts) == 1 {
		repoParts = []string{"library", repoParts[0]}
	}

	repo := strings.Join(repoParts, "/")
	name := repoParts[len(repoParts)-1]
	if host != ollamaDefaultRegistry || len(repoParts) > 2 {
		name = strings.Join(repoParts, "/")
	}
	if tag != "latest" {
		name += ":" + tag
	}

	return ollamaRegistryRef{
		Original: ref,
		Scheme:   scheme,
		Host:     host,
		Repo:     repo,
		Tag:      tag,
		Name:     name,
	}, nil
}

func isOllamaRegistryHost(value string) bool {
	return value == "localhost" || strings.Contains(value, ".") || strings.Contains(value, ":")
}

func (m *ModelManager) fetchOllamaManifest(ctx context.Context, ref ollamaRegistryRef) (*ollamaManifest, error) {
	requestURL := url.URL{Scheme: ref.Scheme, Host: ref.Host}
	requestURL.Path = filepath.ToSlash(filepath.Join("v2", ref.Repo, "manifests", ref.Tag))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var manifest ollamaManifest
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return nil, err
	}
	return &manifest, nil
}

func (m *ModelManager) downloadOllamaBlob(
	ctx context.Context,
	ref ollamaRegistryRef,
	layer ollamaLayer,
	progress func(OllamaPullProgress),
) (string, error) {
	path, err := m.ollamaBlobPath(layer.Digest)
	if err != nil {
		return "", err
	}
	if info, err := os.Stat(path); err == nil && info.Size() == layer.Size {
		progress(OllamaPullProgress{
			Status:    "pulling " + shortDigest(layer.Digest),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: layer.Size,
		})
		return path, nil
	}

	partPath := path + ".part"
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", err
	}

	var resumeFrom int64
	if info, err := os.Stat(partPath); err == nil {
		resumeFrom = info.Size()
	}

	requestURL := url.URL{Scheme: ref.Scheme, Host: ref.Host}
	requestURL.Path = filepath.ToSlash(filepath.Join("v2", ref.Repo, "blobs", layer.Digest))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		return "", err
	}
	if resumeFrom > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeFrom))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resumeFrom > 0 && resp.StatusCode == http.StatusOK {
		resumeFrom = 0
	}
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("pull blob %s: HTTP %d: %s", layer.Digest, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	out, err := os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return "", err
	}
	if resumeFrom == 0 {
		if err := out.Truncate(0); err != nil {
			out.Close()
			return "", err
		}
	}

	buf := make([]byte, 256*1024)
	completed := resumeFrom
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				out.Close()
				return "", err
			}
			completed += int64(n)
			progress(OllamaPullProgress{
				Status:    "pulling " + shortDigest(layer.Digest),
				Digest:    layer.Digest,
				Total:     layer.Size,
				Completed: completed,
			})
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			out.Close()
			return "", readErr
		}
	}
	if err := out.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(partPath, path); err != nil {
		return "", err
	}
	return path, nil
}

func (m *ModelManager) ollamaBlobPath(digest string) (string, error) {
	algorithm, encoded, ok := strings.Cut(digest, ":")
	if !ok || algorithm != "sha256" || encoded == "" {
		return "", fmt.Errorf("unsupported digest %q", digest)
	}
	return filepath.Join(m.modelsDir, "blobs", algorithm+"-"+encoded), nil
}

func verifyDigestFile(digest, path string) error {
	_, encoded, ok := strings.Cut(digest, ":")
	if !ok {
		return fmt.Errorf("invalid digest %q", digest)
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return err
	}
	actual := hex.EncodeToString(hasher.Sum(nil))
	if !strings.EqualFold(actual, encoded) {
		return fmt.Errorf("sha256 mismatch for %s: expected %s, got %s", path, encoded, actual)
	}
	return nil
}

func (m *ModelManager) ollamaManifestEntry(
	ref ollamaRegistryRef,
	manifest *ollamaManifest,
	modelLayer ollamaLayer,
	layerPaths map[string]string,
) *storage.ModelEntry {
	filename := safeOllamaModelFilename(ref.Name)
	meta := parseModelMetadata(filename)
	template := readTextLayer(layerPaths, manifest.Layers, "application/vnd.ollama.image.template")
	if template == "" {
		template = readTextLayer(layerPaths, manifest.Layers, "application/vnd.ollama.image.prompt")
	}
	stopTokens := readStopTokens(layerPaths, manifest.Layers)
	capabilities := inferModelCapabilities(ref.Name, filename)
	settings := inferRecommendedSettings(meta)

	return &storage.ModelEntry{
		ID:                  uuid.New().String(),
		Name:                ref.Name,
		Filename:            filename,
		Size:                modelLayer.Size,
		Quantization:        meta.quantization,
		Family:              meta.family,
		Parameters:          meta.parameters,
		Template:            template,
		StopTokens:          stopTokens,
		Capabilities:        capabilities,
		RecommendedSettings: settings,
		SourceURL:           "ollama://" + ref.Original,
		SHA256:              strings.TrimPrefix(modelLayer.Digest, "sha256:"),
		Status:              "ready",
		FilePath:            layerPaths[modelLayer.Digest],
		DownloadedAt:        time.Now().UTC(),
	}
}

func safeOllamaModelFilename(name string) string {
	replacer := strings.NewReplacer("/", "-", ":", "-")
	name = strings.Trim(replacer.Replace(name), "-")
	if name == "" {
		name = "model"
	}
	return name + ".gguf"
}

func readTextLayer(paths map[string]string, layers []ollamaLayer, mediaType string) string {
	for _, layer := range layers {
		if layer.MediaType != mediaType {
			continue
		}
		data, err := os.ReadFile(paths[layer.Digest])
		if err == nil {
			return string(data)
		}
	}
	return ""
}

func readStopTokens(paths map[string]string, layers []ollamaLayer) []string {
	for _, layer := range layers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}
		data, err := os.ReadFile(paths[layer.Digest])
		if err != nil {
			return nil
		}
		var params map[string]any
		if err := json.Unmarshal(data, &params); err != nil {
			return nil
		}
		raw, ok := params["stop"]
		if !ok {
			return nil
		}
		switch value := raw.(type) {
		case string:
			return []string{value}
		case []any:
			stops := make([]string, 0, len(value))
			for _, item := range value {
				if stop, ok := item.(string); ok {
					stops = append(stops, stop)
				}
			}
			return stops
		default:
			return nil
		}
	}
	return nil
}

func shortDigest(digest string) string {
	_, encoded, ok := strings.Cut(digest, ":")
	if !ok {
		return digest
	}
	if len(encoded) > 12 {
		return encoded[:12]
	}
	return encoded
}

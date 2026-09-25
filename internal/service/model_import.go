package service

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/storage"
)

// ImportIssue describes an entry that could not be scanned during import,
// e.g. a folder without read permission or a broken symlink.
type ImportIssue struct {
	Path  string `json:"path"`
	Error string `json:"error"`
}

// ImportResult is the outcome of scanning a directory for models.
type ImportResult struct {
	Imported []*storage.ModelEntry
	Issues   []ImportIssue
}

// ImportFromDirectory scans a directory for .gguf files and registers them.
// Already registered files are skipped.
func (m *ModelManager) ImportFromDirectory(dir string) ([]*storage.ModelEntry, error) {
	res, err := m.ImportFromDirectoryDetailed(dir)
	if res == nil {
		return nil, err
	}
	return res.Imported, err
}

// ScanModelsDir registers .gguf files that were placed into the runtime's own
// models directory by hand (copied, moved, symlinked) and are not yet in
// registry.json. It is called on startup; a missing directory is not an error.
func (m *ModelManager) ScanModelsDir() (*ImportResult, error) {
	if m.modelsDir == "" {
		return &ImportResult{}, nil
	}
	if _, err := os.Stat(m.modelsDir); errors.Is(err, fs.ErrNotExist) {
		return &ImportResult{}, nil
	}
	return m.ImportFromDirectoryDetailed(m.modelsDir)
}

// ImportFromDirectoryDetailed scans dir recursively for .gguf files and
// registers the new ones. Symlinked files and folders are followed (with
// cycle protection). Entries that cannot be read are reported in Issues
// instead of being silently skipped.
func (m *ModelManager) ImportFromDirectoryDetailed(dir string) (*ImportResult, error) {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve path: %w", err)
	}

	info, err := os.Stat(absDir)
	if err != nil {
		return nil, fmt.Errorf("stat dir: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("not a directory: %s", absDir)
	}

	// Already registered files, keyed by their canonical path so that
	// symlinked or differently-cased (Windows, macOS) paths do not duplicate.
	existing := make(map[string]bool)
	for _, e := range m.registry.List() {
		if e.FilePath != "" {
			existing[canonicalPathKey(e.FilePath)] = true
		}
	}

	managedRoot := ""
	if m.modelsDir != "" {
		managedRoot = canonicalPathKey(m.modelsDir)
	}

	res := &ImportResult{}
	report := func(path string, err error) {
		res.Issues = append(res.Issues, ImportIssue{Path: path, Error: describeFSError(err)})
		slog.Warn("model import: skipped entry", "path", path, "error", err)
	}

	walkModelTree(absDir, report, func(path string, info os.FileInfo) {
		name := info.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			return
		}
		// Skip multimodal projection files — they are not standalone models
		if isMMProjFilename(name) {
			return
		}
		key := canonicalPathKey(path)
		if existing[key] {
			return
		}

		meta := parseModelMetadata(name)

		// Derive a friendly name from the directory structure:
		// .../author/model-name/file.gguf  →  "author/model-name"
		modelName := deriveModelName(absDir, path, name)

		discovery := discoverMMProjFilename(path)
		if len(discovery.ambiguous) > 0 {
			slog.Warn("multiple mmproj files found while importing model; manual selection required",
				"name", modelName,
				"model_path", path,
				"mmproj_candidates", discovery.ambiguous,
			)
		}

		entry := &storage.ModelEntry{
			ID:                  uuid.New().String(),
			Name:                modelName,
			Filename:            name,
			Size:                info.Size(),
			Quantization:        meta.quantization,
			Family:              meta.family,
			Parameters:          meta.parameters,
			Capabilities:        inferModelCapabilities(modelName, name),
			RecommendedSettings: inferRecommendedSettings(meta),
			MMProjFilename:      discovery.filename,
			SourceURL:           "file://" + path,
			Status:              "ready",
			FilePath:            path,
			DownloadedAt:        info.ModTime().UTC(),
			// Files that physically live in the runtime's models directory
			// are managed by the runtime: deleting the model removes the
			// file, so it does not come back on the next startup scan.
			// Anything outside (LM Studio folders, symlink targets) is
			// external and never deleted.
			External: managedRoot == "" || !isWithinDir(key, managedRoot),
		}
		normalizeModelMetadata(entry)

		if err := m.registry.Add(entry); err != nil {
			report(path, fmt.Errorf("register model: %w", err))
			return
		}
		existing[key] = true

		res.Imported = append(res.Imported, entry)
		slog.Info("imported model", "name", modelName, "path", path, "size", info.Size())
	})

	return res, nil
}

// walkModelTree walks root recursively, following symlinks to files and
// folders. Each real folder is visited once, so symlink loops terminate.
// visit receives regular files (after resolving symlinks); report receives
// entries that could not be read.
func walkModelTree(root string, report func(string, error), visit func(string, os.FileInfo)) {
	visited := make(map[string]bool)

	var walk func(dir string)
	walk = func(dir string) {
		key := canonicalPathKey(dir)
		if visited[key] {
			return
		}
		visited[key] = true

		entries, err := os.ReadDir(dir)
		if err != nil {
			report(dir, err)
			// ReadDir may return a partial listing together with the error.
		}
		for _, e := range entries {
			path := filepath.Join(dir, e.Name())
			info, err := e.Info()
			if err != nil {
				report(path, err)
				continue
			}
			if info.Mode()&fs.ModeSymlink != 0 {
				info, err = os.Stat(path)
				if err != nil {
					report(path, fmt.Errorf("broken symlink: %w", err))
					continue
				}
			}
			if info.IsDir() {
				walk(path)
				continue
			}
			if info.Mode().IsRegular() {
				visit(path, info)
			}
		}
	}
	walk(root)
}

// canonicalPathKey returns a comparison key for a path: absolute, symlinks
// resolved when possible, and case-folded on case-insensitive platforms.
func canonicalPathKey(path string) string {
	p := path
	if abs, err := filepath.Abs(p); err == nil {
		p = abs
	}
	if real, err := filepath.EvalSymlinks(p); err == nil {
		p = real
	}
	p = filepath.Clean(p)
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		p = strings.ToLower(p)
	}
	return p
}

// isWithinDir reports whether path is dir itself or inside it. Both must be
// canonical keys.
func isWithinDir(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

func describeFSError(err error) string {
	switch {
	case errors.Is(err, fs.ErrPermission):
		return "permission denied"
	case errors.Is(err, fs.ErrNotExist) && strings.HasPrefix(err.Error(), "broken symlink"):
		return "broken symlink"
	}
	var pe *fs.PathError
	if errors.As(err, &pe) {
		return pe.Err.Error()
	}
	return err.Error()
}

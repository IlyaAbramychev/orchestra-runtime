package service

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/operium/orchestra-runtime/internal/storage"
)

func newImportTestManager(t *testing.T, modelsDir string) (*ModelManager, *storage.ModelRegistry) {
	t.Helper()
	registry, err := storage.NewModelRegistry(modelsDir)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	return NewModelManager(registry, &autoLoadBackend{}, modelsDir), registry
}

func writeTestFile(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestScanModelsDirRegistersHandPlacedModels(t *testing.T) {
	modelsDir := t.TempDir()
	writeTestFile(t, filepath.Join(modelsDir, "qwen2.5-0.5b-instruct-q4_k_m.gguf"))
	writeTestFile(t, filepath.Join(modelsDir, "author", "model", "model-q8_0.gguf"))

	manager, registry := newImportTestManager(t, modelsDir)
	res, err := manager.ScanModelsDir()
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(res.Imported) != 2 {
		t.Fatalf("expected 2 imported models, got %d", len(res.Imported))
	}
	for _, e := range res.Imported {
		if e.External {
			t.Fatalf("model inside models dir should be managed, got external: %s", e.FilePath)
		}
	}

	// A second scan (next startup) must not duplicate entries.
	res, err = manager.ScanModelsDir()
	if err != nil {
		t.Fatalf("rescan: %v", err)
	}
	if len(res.Imported) != 0 || len(registry.List()) != 2 {
		t.Fatalf("rescan duplicated models: imported=%d total=%d", len(res.Imported), len(registry.List()))
	}

	// Persisted: a fresh registry sees the models.
	reloaded, err := storage.NewModelRegistry(modelsDir)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(reloaded.List()) != 2 {
		t.Fatalf("expected 2 persisted models, got %d", len(reloaded.List()))
	}
}

func TestScanModelsDirMissingDirIsNoop(t *testing.T) {
	manager, _ := newImportTestManager(t, t.TempDir())
	manager.modelsDir = filepath.Join(t.TempDir(), "does-not-exist")
	res, err := manager.ScanModelsDir()
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(res.Imported) != 0 {
		t.Fatalf("expected nothing imported")
	}
}

func TestImportFollowsSymlinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks need privileges on Windows")
	}
	modelsDir := t.TempDir()
	external := t.TempDir()
	writeTestFile(t, filepath.Join(external, "lmstudio", "author", "linked-model", "linked-q4_k_m.gguf"))
	writeTestFile(t, filepath.Join(external, "single-q5_k_m.gguf"))

	importRoot := t.TempDir()
	if err := os.Symlink(filepath.Join(external, "lmstudio"), filepath.Join(importRoot, "lmstudio")); err != nil {
		t.Fatalf("symlink dir: %v", err)
	}
	if err := os.Symlink(filepath.Join(external, "single-q5_k_m.gguf"), filepath.Join(importRoot, "single-q5_k_m.gguf")); err != nil {
		t.Fatalf("symlink file: %v", err)
	}
	// A loop must not hang the walk.
	if err := os.Symlink(importRoot, filepath.Join(importRoot, "loop")); err != nil {
		t.Fatalf("symlink loop: %v", err)
	}

	manager, _ := newImportTestManager(t, modelsDir)
	res, err := manager.ImportFromDirectoryDetailed(importRoot)
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if len(res.Imported) != 2 {
		t.Fatalf("expected 2 models through symlinks, got %d (issues: %+v)", len(res.Imported), res.Issues)
	}
	for _, e := range res.Imported {
		if !e.External {
			t.Fatalf("model outside models dir must be external: %s", e.FilePath)
		}
	}

	// Importing the real folder afterwards must not duplicate the same files.
	res, err = manager.ImportFromDirectoryDetailed(external)
	if err != nil {
		t.Fatalf("import real dir: %v", err)
	}
	if len(res.Imported) != 0 {
		t.Fatalf("expected symlink targets to be deduplicated, got %d new", len(res.Imported))
	}
}

func TestImportReportsBrokenSymlinkAndUnreadableDir(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permissions and symlinks")
	}
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "ok-q4_0.gguf"))
	if err := os.Symlink(filepath.Join(root, "missing.gguf"), filepath.Join(root, "broken.gguf")); err != nil {
		t.Fatalf("symlink: %v", err)
	}
	locked := filepath.Join(root, "locked")
	writeTestFile(t, filepath.Join(locked, "hidden-q4_0.gguf"))
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	t.Cleanup(func() { os.Chmod(locked, 0755) })
	if _, err := os.ReadDir(locked); err == nil {
		t.Skip("running as a user that ignores directory permissions (root)")
	}

	manager, _ := newImportTestManager(t, t.TempDir())
	res, err := manager.ImportFromDirectoryDetailed(root)
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if len(res.Imported) != 1 {
		t.Fatalf("expected the readable model to import, got %d", len(res.Imported))
	}
	got := map[string]string{}
	for _, issue := range res.Issues {
		got[filepath.Base(issue.Path)] = issue.Error
	}
	if got["broken.gguf"] != "broken symlink" {
		t.Fatalf("expected broken symlink issue, got %+v", res.Issues)
	}
	if got["locked"] != "permission denied" {
		t.Fatalf("expected permission denied issue, got %+v", res.Issues)
	}
}

func TestIsWithinDir(t *testing.T) {
	base := filepath.Join(string(filepath.Separator), "a", "models")
	cases := map[string]bool{
		base:                                   true,
		filepath.Join(base, "x.gguf"):          true,
		filepath.Join(base, "sub", "x.gguf"):   true,
		filepath.Join(base+"-other", "x.gguf"): false,
		filepath.Join(string(filepath.Separator), "a", "x.gguf"): false,
	}
	for path, want := range cases {
		if got := isWithinDir(path, base); got != want {
			t.Errorf("isWithinDir(%q) = %v, want %v", path, got, want)
		}
	}
}

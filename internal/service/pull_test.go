package service

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/operium/orchestra-runtime/internal/storage"
)

func TestPullModelWithMetadataPersistsManifestFields(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)
	body := []byte("model")
	sum := sha256.Sum256(body)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer server.Close()

	id, err := manager.PullModelWithMetadata("qwen3-tool", server.URL+"/model.gguf", PullModelMetadata{
		Family:       "qwen",
		Parameters:   "8B",
		Quantization: "Q4_K_M",
		SHA256:       hex.EncodeToString(sum[:]),
		Template:     "{{ .Prompt }}",
		StopTokens:   []string{"<|im_end|>"},
		Capabilities: &storage.ModelCapabilities{
			Chat:     true,
			Tools:    true,
			Thinking: true,
		},
		RecommendedSettings: &storage.RecommendedModelSettings{
			ContextSize: 32768,
		},
	})
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
	}

	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	if entry.Family != "qwen" || entry.Parameters != "8B" || entry.Quantization != "Q4_K_M" {
		t.Fatalf("metadata not persisted: %+v", entry)
	}
	if entry.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("sha256 = %q", entry.SHA256)
	}
	if entry.Template != "{{ .Prompt }}" {
		t.Fatalf("template = %q", entry.Template)
	}
	if len(entry.StopTokens) != 1 || entry.StopTokens[0] != "<|im_end|>" {
		t.Fatalf("stop_tokens = %#v", entry.StopTokens)
	}
	if !entry.Capabilities.Chat || !entry.Capabilities.Tools || !entry.Capabilities.Thinking {
		t.Fatalf("capabilities = %+v", entry.Capabilities)
	}
	if entry.RecommendedSettings.ContextSize != 32768 {
		t.Fatalf("recommended settings = %+v", entry.RecommendedSettings)
	}
}

func TestPullModelFailsOnSHA256Mismatch(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("model"))
	}))
	defer server.Close()

	id, err := manager.PullModelWithMetadata("bad-sha", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: "0000000000000000000000000000000000000000000000000000000000000000",
	})
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
		if ds.Error == nil {
			t.Fatal("expected download error")
		}
	}

	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	if entry.Status != "error" {
		t.Fatalf("expected error status, got %s", entry.Status)
	}
	if _, err := os.Stat(entry.FilePath); !os.IsNotExist(err) {
		t.Fatalf("expected final file removed, stat err=%v", err)
	}
}

package service

import (
	"net/http"
	"net/http/httptest"
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
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("model"))
	}))
	defer server.Close()

	id, err := manager.PullModelWithMetadata("qwen3-tool", server.URL+"/model.gguf", PullModelMetadata{
		Family:       "qwen",
		Parameters:   "8B",
		Quantization: "Q4_K_M",
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

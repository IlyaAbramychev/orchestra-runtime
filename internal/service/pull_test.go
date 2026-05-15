package service

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

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

func TestPullModelResumesPartialDownload(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)
	body := []byte("hello world")
	sum := sha256.Sum256(body)
	rangeSeen := make(chan string, 1)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/model.gguf" {
			http.NotFound(w, r)
			return
		}
		rangeHeader := r.Header.Get("Range")
		if rangeHeader != "" {
			rangeSeen <- rangeHeader
			if rangeHeader != "bytes=5-" {
				http.Error(w, "bad range", http.StatusRequestedRangeNotSatisfiable)
				return
			}
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 5-%d/%d", len(body)-1, len(body)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(body[5:])
			return
		}
		_, _ = w.Write(body)
	}))
	defer server.Close()

	partPath := tmp + "/model.gguf.part"
	if err := os.WriteFile(partPath, body[:5], 0644); err != nil {
		t.Fatalf("write partial: %v", err)
	}

	id, err := manager.PullModelWithMetadata("resume", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: hex.EncodeToString(sum[:]),
	})
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
		if ds.Error != nil {
			t.Fatalf("download error: %v", ds.Error)
		}
	}

	select {
	case got := <-rangeSeen:
		if got != "bytes=5-" {
			t.Fatalf("range = %q", got)
		}
	default:
		t.Fatal("expected range request")
	}

	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	data, err := os.ReadFile(entry.FilePath)
	if err != nil {
		t.Fatalf("read final file: %v", err)
	}
	if string(data) != string(body) {
		t.Fatalf("final file = %q", data)
	}
	if entry.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("sha256 = %q", entry.SHA256)
	}
}

func TestPullModelRetriesTransientHTTPError(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)
	body := []byte("model")
	sum := sha256.Sum256(body)
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			http.Error(w, "temporary failure", http.StatusBadGateway)
			return
		}
		_, _ = w.Write(body)
	}))
	defer server.Close()

	id, err := manager.PullModelWithMetadata("retry", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: hex.EncodeToString(sum[:]),
	})
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
		if ds.Error != nil {
			t.Fatalf("download error: %v", ds.Error)
		}
	}

	if attempts.Load() != 2 {
		t.Fatalf("attempts = %d", attempts.Load())
	}
	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	if entry.Status != "ready" {
		t.Fatalf("expected ready status, got %s", entry.Status)
	}
}

func TestPullModelDeduplicatesActiveDownload(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)
	body := []byte("model")
	sum := sha256.Sum256(body)
	var requests atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) == 1 {
			close(started)
		}
		<-release
		_, _ = w.Write(body)
	}))
	defer server.Close()

	id, err := manager.PullModelWithMetadata("dedupe", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: hex.EncodeToString(sum[:]),
	})
	if err != nil {
		t.Fatalf("first pull: %v", err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for download")
	}

	duplicateID, err := manager.PullModelWithMetadata("dedupe", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: hex.EncodeToString(sum[:]),
	})
	if err != nil {
		t.Fatalf("duplicate pull: %v", err)
	}
	if duplicateID != id {
		t.Fatalf("duplicate id = %s, want %s", duplicateID, id)
	}
	if got := len(registry.List()); got != 1 {
		t.Fatalf("registry entries = %d", got)
	}

	close(release)
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
		if ds.Error != nil {
			t.Fatalf("download error: %v", ds.Error)
		}
	}
	if requests.Load() != 1 {
		t.Fatalf("requests = %d", requests.Load())
	}
}

func TestPullModelRetriesFailedEntryWithSameID(t *testing.T) {
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

	id, err := manager.PullModelWithMetadata("retry-failed", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: "0000000000000000000000000000000000000000000000000000000000000000",
	})
	if err != nil {
		t.Fatalf("first pull: %v", err)
	}
	if ds := manager.GetDownloadState(id); ds != nil {
		<-ds.Done
		if ds.Error == nil {
			t.Fatal("expected first download error")
		}
	}
	if entry := registry.Get(id); entry == nil || entry.Status != "error" {
		t.Fatalf("expected failed entry, got %+v", entry)
	}

	retryID, err := manager.PullModelWithMetadata("retry-failed", server.URL+"/model.gguf", PullModelMetadata{
		SHA256: hex.EncodeToString(sum[:]),
	})
	if err != nil {
		t.Fatalf("retry pull: %v", err)
	}
	if retryID != id {
		t.Fatalf("retry id = %s, want %s", retryID, id)
	}
	if ds := manager.GetDownloadState(retryID); ds != nil {
		<-ds.Done
		if ds.Error != nil {
			t.Fatalf("retry download error: %v", ds.Error)
		}
	}

	if got := len(registry.List()); got != 1 {
		t.Fatalf("registry entries = %d", got)
	}
	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	if entry.Status != "ready" {
		t.Fatalf("expected ready status, got %s", entry.Status)
	}
	if entry.ErrorMessage != "" {
		t.Fatalf("expected cleared error message, got %q", entry.ErrorMessage)
	}
	if entry.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("sha256 = %q", entry.SHA256)
	}
}

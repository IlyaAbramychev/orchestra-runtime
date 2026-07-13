package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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

func TestPullOllamaLibraryModelSharesConcurrentProgress(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)

	body := []byte("registry-model")
	digest := testSHA256Digest(body)
	var manifestHits atomic.Int32
	var blobHits atomic.Int32
	blobStarted := make(chan struct{})
	releaseBlob := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/shared/manifests/latest":
			manifestHits.Add(1)
			writeManifestResponse(t, w, digest, len(body))
		case "/v2/library/shared/blobs/" + digest:
			if blobHits.Add(1) == 1 {
				close(blobStarted)
				<-releaseBlob
			}
			_, _ = w.Write(body)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	ref := strings.TrimPrefix(server.URL, "http://") + "/library/shared:latest"
	firstDone := make(chan struct {
		id  string
		err error
	}, 1)
	go func() {
		id, err := manager.PullOllamaLibraryModel(context.Background(), ref, true, nil)
		firstDone <- struct {
			id  string
			err error
		}{id: id, err: err}
	}()

	select {
	case <-blobStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first blob request")
	}

	secondProgress := make(chan string, 8)
	secondDone := make(chan struct {
		id  string
		err error
	}, 1)
	go func() {
		id, err := manager.PullOllamaLibraryModel(context.Background(), ref, true, func(progress OllamaPullProgress) {
			secondProgress <- progress.Status
		})
		secondDone <- struct {
			id  string
			err error
		}{id: id, err: err}
	}()

	close(releaseBlob)

	first := <-firstDone
	second := <-secondDone
	if first.err != nil {
		t.Fatalf("first pull failed: %v", first.err)
	}
	if second.err != nil {
		t.Fatalf("second pull failed: %v", second.err)
	}
	if first.id == "" || first.id != second.id {
		t.Fatalf("expected shared model id, got first=%q second=%q", first.id, second.id)
	}
	if manifestHits.Load() != 1 {
		t.Fatalf("manifest hits = %d", manifestHits.Load())
	}
	if blobHits.Load() != 1 {
		t.Fatalf("blob hits = %d", blobHits.Load())
	}
	close(secondProgress)
	seenSuccess := false
	for status := range secondProgress {
		if status == "success" {
			seenSuccess = true
		}
	}
	if !seenSuccess {
		t.Fatal("expected second subscriber to receive success progress")
	}
}

func TestPullOllamaLibraryModelPersistsManifestMetadata(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)

	modelBlob := []byte("registry-model")
	templateBlob := []byte("{{ .System }}\n{{ .Prompt }}")
	systemBlob := []byte("You are concise.")
	paramsBlob := []byte(`{"temperature":0.2,"stop":["<|eot_id|>","<|end|>"]}`)
	licenseBlob := []byte("MIT")
	blobs := map[string][]byte{
		testSHA256Digest(modelBlob):    modelBlob,
		testSHA256Digest(templateBlob): templateBlob,
		testSHA256Digest(systemBlob):   systemBlob,
		testSHA256Digest(paramsBlob):   paramsBlob,
		testSHA256Digest(licenseBlob):  licenseBlob,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/meta/manifests/latest":
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(map[string]any{
				"schemaVersion": 2,
				"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
				"layers": []map[string]any{
					{"mediaType": "application/vnd.ollama.image.model", "digest": testSHA256Digest(modelBlob), "size": len(modelBlob)},
					{"mediaType": "application/vnd.ollama.image.template", "digest": testSHA256Digest(templateBlob), "size": len(templateBlob)},
					{"mediaType": "application/vnd.ollama.image.system", "digest": testSHA256Digest(systemBlob), "size": len(systemBlob)},
					{"mediaType": "application/vnd.ollama.image.params", "digest": testSHA256Digest(paramsBlob), "size": len(paramsBlob)},
					{"mediaType": "application/vnd.ollama.image.license", "digest": testSHA256Digest(licenseBlob), "size": len(licenseBlob)},
				},
			}); err != nil {
				t.Fatalf("write manifest: %v", err)
			}
		default:
			prefix := "/v2/library/meta/blobs/"
			digest := strings.TrimPrefix(r.URL.Path, prefix)
			if digest == r.URL.Path {
				http.NotFound(w, r)
				return
			}
			body, ok := blobs[digest]
			if !ok {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(body)
		}
	}))
	defer server.Close()

	ref := strings.TrimPrefix(server.URL, "http://") + "/library/meta:latest"
	id, err := manager.PullOllamaLibraryModel(context.Background(), ref, true, nil)
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	if entry.Template != string(templateBlob) {
		t.Fatalf("template = %q", entry.Template)
	}
	if entry.System != string(systemBlob) {
		t.Fatalf("system = %q", entry.System)
	}
	if len(entry.StopTokens) != 2 || entry.StopTokens[0] != "<|eot_id|>" || entry.StopTokens[1] != "<|end|>" {
		t.Fatalf("stop tokens = %#v", entry.StopTokens)
	}
	if !strings.Contains(entry.OllamaParameters, `PARAMETER stop "<|eot_id|>"`) {
		t.Fatalf("ollama parameters = %q", entry.OllamaParameters)
	}
	if !strings.Contains(entry.OllamaParameters, "PARAMETER temperature 0.2") {
		t.Fatalf("ollama parameters = %q", entry.OllamaParameters)
	}
	if len(entry.License) != 1 || entry.License[0] != "MIT" {
		t.Fatalf("license = %#v", entry.License)
	}
	if !strings.Contains(entry.Modelfile, "SYSTEM") || !strings.Contains(entry.Modelfile, "LICENSE") {
		t.Fatalf("modelfile = %q", entry.Modelfile)
	}
}

func TestPullOllamaLibraryModelRejectsLegacyGPTOSSBeforeWeights(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)

	configBlob := []byte(`{"model_format":"gguf","model_family":"gptoss","model_families":["gptoss"],"model_type":"20.9B","file_type":"MXFP4"}`)
	modelBlob := []byte("legacy provider-specific weights")
	configDigest := testSHA256Digest(configBlob)
	modelDigest := testSHA256Digest(modelBlob)
	var modelRequests atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/gpt-oss/manifests/20b":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"schemaVersion": 2,
				"config": map[string]any{
					"mediaType": "application/vnd.docker.container.image.v1+json",
					"digest":    configDigest,
					"size":      len(configBlob),
				},
				"layers": []map[string]any{{
					"mediaType": "application/vnd.ollama.image.model",
					"digest":    modelDigest,
					"size":      len(modelBlob),
				}},
			})
		case "/v2/library/gpt-oss/blobs/" + configDigest:
			_, _ = w.Write(configBlob)
		case "/v2/library/gpt-oss/blobs/" + modelDigest:
			modelRequests.Add(1)
			_, _ = w.Write(modelBlob)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	ref := strings.TrimPrefix(server.URL, "http://") + "/library/gpt-oss:20b"
	_, err = manager.PullOllamaLibraryModel(context.Background(), ref, true, nil)
	var incompatible *IncompatibleModelArtifactError
	if !errors.As(err, &incompatible) {
		t.Fatalf("pull error = %v; want IncompatibleModelArtifactError", err)
	}
	if got := modelRequests.Load(); got != 0 {
		t.Fatalf("model blob requests = %d; want 0", got)
	}
	if got := len(registry.List()); got != 0 {
		t.Fatalf("registry entries = %d; want 0", got)
	}
}

func TestPullOllamaLibraryModelWithRecordedRegistryFixture(t *testing.T) {
	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	manager := NewModelManager(registry, &autoLoadBackend{}, tmp)

	fixtureDir := filepath.Join("testdata", "ollama_registry")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/recorded/manifests/latest":
			w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
			http.ServeFile(w, r, filepath.Join(fixtureDir, "manifest.json"))
		default:
			prefix := "/v2/library/recorded/blobs/"
			digest := strings.TrimPrefix(r.URL.Path, prefix)
			if digest == r.URL.Path {
				http.NotFound(w, r)
				return
			}
			blobName := strings.Replace(digest, ":", "-", 1)
			http.ServeFile(w, r, filepath.Join(fixtureDir, "blobs", blobName))
		}
	}))
	defer server.Close()

	ref := strings.TrimPrefix(server.URL, "http://") + "/library/recorded:latest"
	id, err := manager.PullOllamaLibraryModel(context.Background(), ref, true, nil)
	if err != nil {
		t.Fatalf("pull: %v", err)
	}
	entry := registry.Get(id)
	if entry == nil {
		t.Fatal("expected registry entry")
	}
	if entry.SHA256 != "987c23397402954bc7db2283b33ab16036fd67a68d6224ddf8078e9fbb2767b8" {
		t.Fatalf("sha256 = %q", entry.SHA256)
	}
	if entry.Template != "{{ .System }}\n{{ .Prompt }}\n" {
		t.Fatalf("template = %q", entry.Template)
	}
	if entry.System != "You are a recorded fixture.\n" {
		t.Fatalf("system = %q", entry.System)
	}
	if len(entry.StopTokens) != 1 || entry.StopTokens[0] != "<|stop|>" {
		t.Fatalf("stop tokens = %#v", entry.StopTokens)
	}
	if entry.OllamaParameters != "PARAMETER stop \"<|stop|>\"\nPARAMETER temperature 0.1" {
		t.Fatalf("ollama parameters = %q", entry.OllamaParameters)
	}
	if len(entry.License) != 1 || entry.License[0] != "Apache-2.0" {
		t.Fatalf("license = %#v", entry.License)
	}
	data, err := os.ReadFile(entry.FilePath)
	if err != nil {
		t.Fatalf("read pulled model: %v", err)
	}
	if string(data) != "gguf fixture\n" {
		t.Fatalf("model blob = %q", data)
	}
}

func testSHA256Digest(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func writeManifestResponse(t *testing.T, w http.ResponseWriter, digest string, size int) {
	t.Helper()
	w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	_, err := fmt.Fprintf(w, `{
		"schemaVersion": 2,
		"mediaType": "application/vnd.docker.distribution.manifest.v2+json",
		"layers": [
			{"mediaType": "application/vnd.ollama.image.model", "digest": %q, "size": %d}
		]
	}`, digest, size)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}
}

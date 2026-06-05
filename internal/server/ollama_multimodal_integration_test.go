package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/config"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/handler"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/service"
	"github.com/operium/orchestra-runtime/internal/storage"
)

const tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0a8AAAAASUVORK5CYII="
const tinyPNGDataURI = "data:image/png;base64," + tinyPNGBase64

func TestOllamaMultimodalRealVisionIntegration(t *testing.T) {
	modelPath := strings.TrimSpace(os.Getenv("ORCHESTRA_TEST_VISION_MODEL_PATH"))
	mmprojPath := strings.TrimSpace(os.Getenv("ORCHESTRA_TEST_VISION_MMPROJ_PATH"))
	if modelPath == "" || mmprojPath == "" {
		t.Skip("set ORCHESTRA_TEST_VISION_MODEL_PATH and ORCHESTRA_TEST_VISION_MMPROJ_PATH to run real multimodal integration")
	}

	server := httptest.NewServer(newVisionIntegrationRouter(t, modelPath, mmprojPath))
	t.Cleanup(server.Close)

	t.Run("chat", func(t *testing.T) {
		var resp model.OllamaChatResponse
		visionJSON(
			t,
			server.URL,
			http.MethodPost,
			"/api/chat",
			`{"model":"vision-smoke:latest","stream":false,"messages":[{"role":"user","content":"describe the image briefly","images":["`+tinyPNGBase64+`"]}]}`,
			http.StatusOK,
			&resp,
		)
		if !resp.Done {
			t.Fatalf("expected final response, got %+v", resp)
		}
		if resp.Error != "" {
			t.Fatalf("unexpected chat error: %+v", resp)
		}
		if resp.PromptEvalCount <= 0 {
			t.Fatalf("expected prompt eval count, got %+v", resp)
		}
	})

	t.Run("chat multiple images mixed encodings", func(t *testing.T) {
		var resp model.OllamaChatResponse
		visionJSON(
			t,
			server.URL,
			http.MethodPost,
			"/api/chat",
			`{"model":"vision-smoke:latest","stream":false,"messages":[{"role":"user","content":"compare both images briefly","images":["`+tinyPNGBase64+`","`+tinyPNGDataURI+`"]}]}`,
			http.StatusOK,
			&resp,
		)
		if !resp.Done {
			t.Fatalf("expected final response, got %+v", resp)
		}
		if resp.Error != "" {
			t.Fatalf("unexpected chat error: %+v", resp)
		}
	})

	t.Run("generate", func(t *testing.T) {
		var resp model.GenerateResponse
		visionJSON(
			t,
			server.URL,
			http.MethodPost,
			"/api/generate",
			`{"model":"vision-smoke:latest","stream":false,"prompt":"describe the image briefly","images":["`+tinyPNGBase64+`"]}`,
			http.StatusOK,
			&resp,
		)
		if !resp.Done {
			t.Fatalf("expected final response, got %+v", resp)
		}
		if resp.Error != "" {
			t.Fatalf("unexpected generate error: %+v", resp)
		}
		if resp.PromptEvalCount <= 0 {
			t.Fatalf("expected prompt eval count, got %+v", resp)
		}
	})

	t.Run("generate multiple images mixed encodings", func(t *testing.T) {
		var resp model.GenerateResponse
		visionJSON(
			t,
			server.URL,
			http.MethodPost,
			"/api/generate",
			`{"model":"vision-smoke:latest","stream":false,"prompt":"compare both images briefly","images":["`+tinyPNGBase64+`","`+tinyPNGDataURI+`"]}`,
			http.StatusOK,
			&resp,
		)
		if !resp.Done {
			t.Fatalf("expected final response, got %+v", resp)
		}
		if resp.Error != "" {
			t.Fatalf("unexpected generate error: %+v", resp)
		}
	})
}

func TestOllamaMultimodalRejectsIncompatibleProjector(t *testing.T) {
	modelPath := strings.TrimSpace(os.Getenv("ORCHESTRA_TEST_VISION_MODEL_PATH"))
	badMMProjPath := strings.TrimSpace(os.Getenv("ORCHESTRA_TEST_BAD_MMPROJ_PATH"))
	if modelPath == "" || badMMProjPath == "" {
		t.Skip("set ORCHESTRA_TEST_VISION_MODEL_PATH and ORCHESTRA_TEST_BAD_MMPROJ_PATH to run incompatible projector fixture")
	}

	server := httptest.NewServer(newVisionIntegrationRouter(t, modelPath, badMMProjPath))
	t.Cleanup(server.Close)

	var resp struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	visionJSON(
		t,
		server.URL,
		http.MethodPost,
		"/api/chat",
		`{"model":"vision-smoke:latest","stream":false,"messages":[{"role":"user","content":"describe the image briefly","images":["`+tinyPNGBase64+`"]}]}`,
		http.StatusBadRequest,
		&resp,
	)
	if msg := strings.ToLower(strings.TrimSpace(resp.Error.Message)); msg == "" {
		t.Fatalf("expected error payload, got %+v", resp)
	} else if !containsAny(msg, "mmproj", "projector", "vision", "clip") {
		t.Fatalf("expected projector-related error, got %q", msg)
	}
	if resp.Error.Code == "" {
		t.Fatalf("expected classified error code, got %+v", resp)
	}
}

func newVisionIntegrationRouter(t *testing.T, modelPath, mmprojPath string) http.Handler {
	t.Helper()

	tmp := t.TempDir()
	registry, err := storage.NewModelRegistry(tmp)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}

	modelLink := filepath.Join(tmp, filepath.Base(modelPath))
	mmprojLink := filepath.Join(tmp, filepath.Base(mmprojPath))
	if err := os.Symlink(modelPath, modelLink); err != nil {
		t.Fatalf("symlink model: %v", err)
	}
	if err := os.Symlink(mmprojPath, mmprojLink); err != nil {
		t.Fatalf("symlink mmproj: %v", err)
	}

	if err := registry.Add(&storage.ModelEntry{
		ID:             "vision-smoke",
		Name:           "vision-smoke:latest",
		Filename:       filepath.Base(modelLink),
		Size:           1,
		Family:         "vision",
		Capabilities:   storage.ModelCapabilities{Chat: true},
		Status:         model.StatusReady,
		FilePath:       modelLink,
		MMProjFilename: filepath.Base(mmprojLink),
		DownloadedAt:   time.Now().UTC(),
		External:       true,
	}); err != nil {
		t.Fatalf("add model: %v", err)
	}

	backend := engine.New()
	backend.InitBackend()
	t.Cleanup(func() {
		_ = backend.Close()
		backend.FreeBackend()
	})

	scheduler := service.NewRuntimeScheduler(backend, 1)
	modelMgr := service.NewModelManagerWithScheduler(registry, scheduler, tmp)
	modelMgr.SetDefaultLoadOptions(engine.DefaultLoadOptions())

	inference := service.NewInferenceServiceWithScheduler(scheduler)
	inference.SetModelLoader(modelMgr)
	embedding := service.NewEmbeddingServiceWithScheduler(scheduler)
	embedding.SetModelLoader(modelMgr)
	sysInfo := service.NewSystemInfo(backend)
	sysInfo.SetScheduler(scheduler)

	chatH := handler.NewChatHandler(inference)
	genH := handler.NewGenerateHandler(inference)
	embH := handler.NewEmbedHandler(embedding, inference)
	modelsH := handler.NewModelsHandler(modelMgr, backend)
	systemH := handler.NewSystemHandler(sysInfo)
	systemH.SetInference(inference)

	s := &Server{cfg: &config.Config{CORSOrigins: []string{"*"}}}
	return s.buildRouter(chatH, genH, embH, modelsH, systemH, handler.NewAdminHandler(nil, func() {}))
}

func visionJSON(t *testing.T, baseURL, method, path, body string, wantStatus int, out any) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, method, baseURL+path, bytes.NewBufferString(body))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request %s %s: %v", method, path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != wantStatus {
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(resp.Body)
		t.Fatalf("%s %s status = %d, want %d: %s", method, path, resp.StatusCode, wantStatus, buf.String())
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		t.Fatalf("decode %s %s: %v", method, path, err)
	}
}

func containsAny(s string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

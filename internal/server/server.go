package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/operium/orchestra-runtime/internal/config"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/handler"
	mw "github.com/operium/orchestra-runtime/internal/middleware"
	"github.com/operium/orchestra-runtime/internal/service"
	"github.com/operium/orchestra-runtime/internal/storage"
	"github.com/operium/orchestra-runtime/internal/supervisor"
)

type Server struct {
	cfg        *config.Config
	router     *chi.Mux
	httpServer *http.Server
	backend    engine.Backend
	// Retained so Shutdown can call Close on the right owner.
	ownsEngine bool
}

func New(cfg *config.Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) Start() error {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: parseLogLevel(s.cfg.LogLevel),
	})))

	// Pick the inference backend. Subprocess mode isolates llama.cpp in a
	// worker — a native crash no longer kills the host. In-process is the
	// default for now (lower latency, one fewer binary to ship), with a
	// feature flag to opt in during the rollout.
	//   ORCHESTRA_USE_SUBPROCESS=1 → out-of-process worker
	//   (unset / "0")              → in-process (CGo in host)
	if os.Getenv("ORCHESTRA_USE_SUBPROCESS") == "1" {
		workerBin := os.Getenv("ORCHESTRA_WORKER_BINARY")
		w := supervisor.NewWorker(supervisor.Options{
			WorkerBinary: workerBin,
		})
		s.backend = supervisor.NewRemote(w)
		slog.Info("inference backend: subprocess (worker)",
			"binary", workerBin,
		)
		// Lazy spawn — worker starts on first LoadModel.
	} else {
		eng := engine.New()
		eng.InitBackend()
		s.backend = eng
		s.ownsEngine = true
		slog.Info("inference backend: in-process")
	}
	s.backend.SetIdleTimeout(s.cfg.IdleTimeout)
	if s.cfg.IdleTimeout > 0 {
		slog.Info("idle auto-unload enabled", "timeout", s.cfg.IdleTimeout)
	}

	// Initialize storage
	registry, err := storage.NewModelRegistry(s.cfg.ModelsDir)
	if err != nil {
		return fmt.Errorf("init model registry: %w", err)
	}

	// Initialize services
	modelMgr := service.NewModelManager(registry, s.backend, s.cfg.ModelsDir)
	inferSvc := service.NewInferenceService(s.backend, s.cfg.MaxQueueSize)
	sysInfo := service.NewSystemInfo(s.backend)

	embedSvc := service.NewEmbeddingService(s.backend, inferSvc)

	// Initialize handlers
	chatH := handler.NewChatHandler(inferSvc)
	genH := handler.NewGenerateHandler(inferSvc)
	embH := handler.NewEmbedHandler(embedSvc, inferSvc)
	modelsH := handler.NewModelsHandler(modelMgr, s.backend)
	systemH := handler.NewSystemHandler(sysInfo)
	systemH.SetInference(inferSvc)

	// Build router
	s.router = s.buildRouter(chatH, genH, embH, modelsH, systemH)

	addr := fmt.Sprintf(":%s", s.cfg.Port)
	slog.Info("starting Orchestra Runtime", "addr", addr, "models_dir", s.cfg.ModelsDir)

	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      s.router,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 10 * time.Minute,
		IdleTimeout:  60 * time.Second,
	}

	return s.httpServer.ListenAndServe()
}

func (s *Server) buildRouter(
	chatH *handler.ChatHandler,
	genH *handler.GenerateHandler,
	embH *handler.EmbedHandler,
	modelsH *handler.ModelsHandler,
	systemH *handler.SystemHandler,
) *chi.Mux {
	r := chi.NewRouter()

	r.Use(chimw.RequestID)
	r.Use(chimw.RealIP)
	r.Use(mw.Logging)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   s.cfg.CORSOrigins,
		AllowedMethods:   []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// Health check (no auth, no recoverer)
	r.Get("/health", systemH.Health)

	r.Group(func(r chi.Router) {
		r.Use(chimw.Recoverer)

		if s.cfg.APIKey != "" {
			r.Use(mw.OptionalAPIKey(s.cfg.APIKey))
		}

		// OpenAI-compatible endpoints
		r.Post("/v1/chat/completions", chatH.ChatCompletion)
		r.Post("/v1/completions", genH.Generate) // OpenAI legacy completions → Ollama-style raw prompt
		r.Post("/v1/embeddings", embH.EmbedOpenAI)
		r.Get("/v1/models", modelsH.ListOpenAI)

		// Ollama-compatible endpoints
		r.Post("/api/chat", chatH.ChatCompletion)
		r.Post("/api/generate", genH.Generate)
		r.Post("/api/embed", embH.Embed)
		r.Post("/api/embeddings", embH.Embed) // legacy Ollama alias
		r.Get("/api/tags", modelsH.ListOllamaTags)
		r.Get("/api/ps", modelsH.ListRunning)
		r.Post("/api/show", modelsH.Show)

		// Model management
		r.Route("/api/models", func(r chi.Router) {
			r.Get("/", modelsH.List)
			r.Post("/pull", modelsH.Pull)
			r.Post("/import", modelsH.Import)
			r.Delete("/{id}", modelsH.Delete)
			r.Post("/{id}/load", modelsH.Load)
			r.Post("/{id}/unload", modelsH.Unload)
			r.Get("/{id}/status", modelsH.Status)
		})

		// System info
		r.Get("/api/system", systemH.Info)
	})

	return r
}

func (s *Server) Shutdown() {
	if s.backend != nil {
		s.backend.UnloadModel()
		if s.ownsEngine {
			s.backend.FreeBackend()
		} else {
			// Remote backend — this will send a graceful shutdown to the worker.
			_ = s.backend.Close()
		}
	}
	if s.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), s.cfg.ShutdownTimeout)
		defer cancel()
		s.httpServer.Shutdown(ctx)
	}
}

func parseLogLevel(level string) slog.Level {
	switch level {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

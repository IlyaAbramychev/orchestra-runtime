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
)

type Server struct {
	cfg        *config.Config
	router     *chi.Mux
	httpServer *http.Server
	eng        *engine.Engine
}

func New(cfg *config.Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) Start() error {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: parseLogLevel(s.cfg.LogLevel),
	})))

	// Initialize engine
	s.eng = engine.New()
	s.eng.InitBackend()

	// Initialize storage
	registry, err := storage.NewModelRegistry(s.cfg.ModelsDir)
	if err != nil {
		return fmt.Errorf("init model registry: %w", err)
	}

	// Initialize services
	modelMgr := service.NewModelManager(registry, s.eng, s.cfg.ModelsDir)
	inferSvc := service.NewInferenceService(s.eng, s.cfg.MaxQueueSize)
	sysInfo := service.NewSystemInfo(s.eng)

	// Initialize handlers
	chatH := handler.NewChatHandler(inferSvc)
	modelsH := handler.NewModelsHandler(modelMgr, s.eng)
	systemH := handler.NewSystemHandler(sysInfo)
	systemH.SetInference(inferSvc)

	// Build router
	s.router = s.buildRouter(chatH, modelsH, systemH)

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

func (s *Server) buildRouter(chatH *handler.ChatHandler, modelsH *handler.ModelsHandler, systemH *handler.SystemHandler) *chi.Mux {
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
		r.Get("/v1/models", modelsH.ListOpenAI)

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
	if s.eng != nil {
		s.eng.UnloadModel()
		s.eng.FreeBackend()
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

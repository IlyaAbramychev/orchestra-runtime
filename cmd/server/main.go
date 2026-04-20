package main

import (
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/operium/orchestra-runtime/internal/config"
	"github.com/operium/orchestra-runtime/internal/server"
)

func main() {
	cfg := config.Load()
	srv := server.New(cfg)

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("shutting down...")
		srv.Shutdown()
		os.Exit(0)
	}()

	if err := srv.Start(); err != nil {
		slog.Error("server failed", "error", err)
		os.Exit(1)
	}
}

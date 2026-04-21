package main

import (
	"log/slog"
	"os"
	"os/signal"
	"runtime/debug"
	"syscall"

	"github.com/operium/orchestra-runtime/internal/config"
	"github.com/operium/orchestra-runtime/internal/server"
)

func main() {
	// Promote memory faults in Go code to recoverable panics — still won't
	// rescue C-thread SIGSEGVs from llama.cpp / Metal (those bypass Go's
	// signal machinery entirely), but at least Go-side bugs won't take the
	// process down silently.
	debug.SetPanicOnFault(true)

	// Quieter crash marker: when llama.cpp blows up in a C thread, Go's
	// default GOTRACEBACK=all dumps every goroutine — noisy and hides the
	// cause. `single` limits to the crashing goroutine, making the actual
	// fatal line visible near the top of the log.
	if os.Getenv("GOTRACEBACK") == "" {
		os.Setenv("GOTRACEBACK", "single")
	}

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

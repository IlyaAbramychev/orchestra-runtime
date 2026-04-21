package handler

import (
	"bytes"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/operium/orchestra-runtime/internal/logbuf"
)

// AdminHandler serves operational endpoints for the extension's Stop / Logs
// buttons and anyone else managing the process from outside.
type AdminHandler struct {
	logs       *logbuf.Handler
	shutdownFn func() // optional; if nil, we just os.Exit(0)
}

func NewAdminHandler(logs *logbuf.Handler, shutdownFn func()) *AdminHandler {
	return &AdminHandler{logs: logs, shutdownFn: shutdownFn}
}

// Logs handles GET /api/logs?n=500. Returns the last N JSON log lines
// joined by newlines. `n` defaults to 500 and is capped at the buffer size.
func (h *AdminHandler) Logs(w http.ResponseWriter, r *http.Request) {
	n := 500
	if v := r.URL.Query().Get("n"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			n = parsed
		}
	}
	lines := [][]byte{}
	if h.logs != nil {
		lines = h.logs.Tail(n)
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	// Each entry already ends with \n from slog.JSONHandler; bytes.Join adds
	// nothing extra.
	_, _ = w.Write(bytes.Join(lines, nil))
}

// Shutdown handles POST /api/shutdown. Replies 202 then exits the process a
// moment later so the HTTP response has time to flush. The extension uses
// this instead of OS-level kill when the runtime was started externally.
func (h *AdminHandler) Shutdown(w http.ResponseWriter, r *http.Request) {
	slog.Info("shutdown requested via HTTP", "remote", r.RemoteAddr)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write([]byte(`{"status":"shutting_down"}`))

	// Flush the response body on its way out, then exit in a goroutine.
	// The delay lets in-flight HTTP writes drain; on modern kernels 100 ms
	// is plenty for a loopback connection.
	go func() {
		time.Sleep(150 * time.Millisecond)
		if h.shutdownFn != nil {
			h.shutdownFn()
		}
		os.Exit(0)
	}()
}

// Package logbuf is a tiny in-memory ring buffer for slog records. We wrap
// it around our primary handler so /api/logs can serve recent output without
// requiring disk files or external log aggregation.
//
// The buffer is bounded — each new record evicts the oldest once the cap is
// hit. Formatted output matches the primary JSON handler so lines in /api/logs
// look identical to what appears on stderr.
package logbuf

import (
	"bytes"
	"context"
	"log/slog"
	"sync"
)

// Handler wraps an inner slog.Handler and keeps the last N rendered records.
type Handler struct {
	inner   slog.Handler
	mu      sync.Mutex
	entries []entry
	cap     int
	head    int // index of oldest; when full we overwrite starting here
	full    bool
}

type entry struct {
	rendered []byte
}

// New wraps `inner` and records up to `capacity` most-recent lines.
func New(inner slog.Handler, capacity int) *Handler {
	if capacity <= 0 {
		capacity = 500
	}
	return &Handler{
		inner:   inner,
		entries: make([]entry, capacity),
		cap:     capacity,
	}
}

func (h *Handler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.inner.Enabled(ctx, level)
}

// Handle records the formatted line into the ring buffer, then delegates to
// the inner handler so stderr/stdout output is unchanged.
func (h *Handler) Handle(ctx context.Context, r slog.Record) error {
	// Render to a temporary buffer using a fresh JSON handler — this way the
	// line we store in memory matches exactly what an operator would see on
	// the primary sink.
	var buf bytes.Buffer
	jh := slog.NewJSONHandler(&buf, nil)
	_ = jh.Handle(ctx, r)
	rendered := append([]byte(nil), buf.Bytes()...)

	h.mu.Lock()
	if h.full {
		h.entries[h.head] = entry{rendered: rendered}
		h.head = (h.head + 1) % h.cap
	} else {
		h.entries[h.head] = entry{rendered: rendered}
		h.head++
		if h.head == h.cap {
			h.head = 0
			h.full = true
		}
	}
	h.mu.Unlock()

	return h.inner.Handle(ctx, r)
}

func (h *Handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &Handler{
		inner:   h.inner.WithAttrs(attrs),
		cap:     h.cap,
		entries: h.entries, // share buffer across children so /api/logs sees everything
	}
}

func (h *Handler) WithGroup(name string) slog.Handler {
	return &Handler{
		inner:   h.inner.WithGroup(name),
		cap:     h.cap,
		entries: h.entries,
	}
}

// Tail returns up to `n` most-recent lines in chronological order. Pass 0 to
// get everything currently in the buffer.
func (h *Handler) Tail(n int) [][]byte {
	h.mu.Lock()
	defer h.mu.Unlock()

	var size int
	if h.full {
		size = h.cap
	} else {
		size = h.head
	}
	if n <= 0 || n > size {
		n = size
	}

	out := make([][]byte, 0, n)
	// Walk chronologically: oldest first is at `head` when full, at 0 when not.
	start := 0
	if h.full {
		start = h.head
	}
	for i := 0; i < size; i++ {
		idx := (start + i) % h.cap
		out = append(out, h.entries[idx].rendered)
	}
	if len(out) > n {
		out = out[len(out)-n:]
	}
	return out
}

// Package rpc defines the wire format between orchestra-runtime HOST and its
// llama.cpp-hosting WORKER subprocess. Both speak newline-delimited JSON over
// a single Unix socket (or named pipe on Windows); each line is one Envelope.
//
// Why a custom protocol rather than gRPC/stdio:
//   - We want ANY frame (request/response/chunk/cancel) interleaved on the
//     same connection so streaming works without extra ports.
//   - NDJSON is trivially debuggable with `socat` / `netcat`.
//   - gRPC adds a .proto + codegen step; value-per-line-of-code is low here.
//
// If the worker crashes mid-stream, the host sees a closed connection and
// cancels all pending calls with ErrWorkerCrashed.
package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// ── Method names (kept stable across versions) ───────────────────────────────

const (
	MethodPing           = "ping"
	MethodStatus         = "status"
	MethodLoadModel      = "load_model"
	MethodUnloadModel    = "unload_model"
	MethodComplete       = "complete"
	MethodCompleteStream = "complete_stream"
	MethodEmbed          = "embed"
	MethodSetIdleTimeout = "set_idle_timeout"
	MethodApplyKeepAlive = "apply_keep_alive"
	MethodCancel         = "cancel"  // cancels another request by id
	MethodShutdown       = "shutdown" // graceful worker exit
)

// ── Error codes ──────────────────────────────────────────────────────────────

const (
	ErrCodeNotReady      = "not_ready"
	ErrCodeInvalid       = "invalid_request"
	ErrCodeLoadFailed    = "load_failed"
	ErrCodeInference     = "inference_failed"
	ErrCodeContextFull   = "context_overflow"
	ErrCodeCancelled     = "cancelled"
	ErrCodeUnsupported   = "unsupported_method"
	ErrCodeInternal      = "internal"
)

// ErrWorkerCrashed is returned on the host side when the worker subprocess
// died while a call was in flight.
var ErrWorkerCrashed = errors.New("worker process crashed")

// ── Envelope: one line = one frame ───────────────────────────────────────────

// Kind describes the role of a frame.
//
// Request lifecycle:
//
//	host                                 worker
//	 ── {id:X, kind:request, ...} ──→
//	                                  ←── {id:X, kind:final, result:...}
//
// Streaming:
//
//	 ── {id:X, kind:request, method:complete_stream} ──→
//	                                  ←── {id:X, kind:chunk, result:{...}}
//	                                  ←── {id:X, kind:chunk, result:{...}}
//	                                  ←── {id:X, kind:final, result:{...}}
//
// Cancellation:
//
//	 ── {id:Y, kind:request, method:cancel, params:{target:X}} ──→
//	 (worker stops pushing chunks for X and sends a final chunk for X)
type Kind string

const (
	KindRequest Kind = "request"
	KindChunk   Kind = "chunk"
	KindFinal   Kind = "final"
)

type Envelope struct {
	ID     string          `json:"id"`
	Kind   Kind            `json:"kind"`
	Method string          `json:"method,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *Error          `json:"error,omitempty"`
}

type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Unwrap() error { return nil }
func (e *Error) Error() string { return fmt.Sprintf("rpc error [%s]: %s", e.Code, e.Message) }

// ── Codec: reads/writes Envelopes over a single byte stream ──────────────────

// Codec wraps a reader + writer with mutex-protected framing. Use one Codec
// per connection. Safe to call Write concurrently; Read is single-reader.
type Codec struct {
	r      *bufio.Reader
	w      io.Writer
	closer io.Closer
	mu     sync.Mutex // serialises writes — multiple goroutines may push chunks
}

func NewCodec(rwc io.ReadWriteCloser) *Codec {
	return &Codec{
		// 2 MB max line — embedding responses for large hidden sizes can be
		// big (4096 dims * 4 bytes * base64 overhead ≈ 22 KB; plenty of room).
		r:      bufio.NewReaderSize(rwc, 2*1024*1024),
		w:      rwc,
		closer: rwc,
	}
}

// Read blocks until the next frame arrives, or returns error on EOF/corruption.
// Not goroutine-safe — only one reader per Codec.
func (c *Codec) Read() (*Envelope, error) {
	line, err := c.r.ReadBytes('\n')
	if len(line) == 0 && err != nil {
		return nil, err
	}
	// Strip trailing \n (and \r on Windows — just in case).
	for len(line) > 0 && (line[len(line)-1] == '\n' || line[len(line)-1] == '\r') {
		line = line[:len(line)-1]
	}
	if len(line) == 0 {
		return nil, fmt.Errorf("rpc: empty frame")
	}
	var env Envelope
	if err := json.Unmarshal(line, &env); err != nil {
		return nil, fmt.Errorf("rpc: decode frame: %w (raw: %q)", err, truncate(line, 200))
	}
	return &env, nil
}

// Write serialises a single frame followed by '\n'. Safe to call concurrently.
func (c *Codec) Write(env *Envelope) error {
	data, err := json.Marshal(env)
	if err != nil {
		return fmt.Errorf("rpc: encode frame: %w", err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, err := c.w.Write(data); err != nil {
		return err
	}
	if _, err := c.w.Write([]byte{'\n'}); err != nil {
		return err
	}
	return nil
}

// Close tears down the underlying transport. Safe to call multiple times.
func (c *Codec) Close() error {
	if c.closer == nil {
		return nil
	}
	err := c.closer.Close()
	c.closer = nil
	return err
}

// ── Typed request helpers (shared by host and worker) ────────────────────────

// ReadWithContext wraps Read so a caller's context can cancel the blocking
// read by closing the connection. Worker uses this in its main loop so the
// shutdown RPC can unblock a hung reader.
func ReadWithContext(ctx context.Context, c *Codec) (*Envelope, error) {
	done := make(chan struct{})
	var env *Envelope
	var err error
	go func() {
		env, err = c.Read()
		close(done)
	}()
	select {
	case <-ctx.Done():
		_ = c.Close() // abort the read by tearing down the conn
		return nil, ctx.Err()
	case <-done:
		return env, err
	}
}

// ── Payloads ─────────────────────────────────────────────────────────────────
//
// These are the JSON shapes embedded in Envelope.Params / .Result. Kept in
// this package so both host and worker import one source of truth.

type LoadParams struct {
	ModelID       string  `json:"model_id"`
	Path          string  `json:"path"`
	GPULayers     int     `json:"gpu_layers"`
	CtxSize       int     `json:"ctx_size"`
	Threads       int     `json:"threads"`
	BatchSize     int     `json:"batch_size,omitempty"`
	RopeFreqBase  float32 `json:"rope_freq_base,omitempty"`
	RopeFreqScale float32 `json:"rope_freq_scale,omitempty"`
	FlashAttn     int     `json:"flash_attn,omitempty"` // -1 auto, 0 off, 1 on
	OffloadKQV    bool    `json:"offload_kqv,omitempty"`
	UseMmap       bool    `json:"use_mmap"`
	UseMlock      bool    `json:"use_mlock,omitempty"`
	TypeK         string  `json:"type_k,omitempty"`
	TypeV         string  `json:"type_v,omitempty"`
}

type StatusResult struct {
	State         string `json:"state"`
	ModelID       string `json:"model_id,omitempty"`
	IsLoaded      bool   `json:"is_loaded"`
	IdleTimeoutNs int64  `json:"idle_timeout_ns"`
}

// ChatMessage mirrors engine.ChatMessage but lives here so the rpc package
// doesn't import engine (which pulls in CGo on the host side).
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionParams struct {
	MaxTokens        int      `json:"max_tokens"`
	Temperature      float32  `json:"temperature"`
	TopK             int      `json:"top_k"`
	TopP             float32  `json:"top_p"`
	MinP             float32  `json:"min_p"`
	TypicalP         float32  `json:"typical_p"`
	RepeatPenalty    float32  `json:"repeat_penalty"`
	RepeatLastN      int      `json:"repeat_last_n"`
	FrequencyPenalty float32  `json:"frequency_penalty"`
	PresencePenalty  float32  `json:"presence_penalty"`
	Seed             int64    `json:"seed"`
	Mirostat         int      `json:"mirostat"`
	MirostatTau      float32  `json:"mirostat_tau"`
	MirostatEta      float32  `json:"mirostat_eta"`
	Stop             []string `json:"stop,omitempty"`
	RawPrompt        bool     `json:"raw_prompt,omitempty"`
}

type CompleteParams struct {
	Messages []ChatMessage    `json:"messages"`
	Params   CompletionParams `json:"params"`
}

type Timings struct {
	TotalNs      int64 `json:"total_ns"`
	PromptEvalNs int64 `json:"prompt_eval_ns"`
	EvalNs       int64 `json:"eval_ns"`
}

type CompleteResult struct {
	Text             string  `json:"text"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	FinishReason     string  `json:"finish_reason"`
	Timings          Timings `json:"timings"`
}

type StreamChunk struct {
	Text             string  `json:"text,omitempty"`
	Done             bool    `json:"done,omitempty"`
	FinishReason     string  `json:"finish_reason,omitempty"`
	PromptTokens     int     `json:"prompt_tokens,omitempty"`
	CompletionTokens int     `json:"completion_tokens,omitempty"`
	Timings          Timings `json:"timings,omitempty"`
	Err              string  `json:"err,omitempty"` // non-fatal: propagated to caller
}

type EmbedParams struct {
	Text      string `json:"text"`
	Normalize bool   `json:"normalize"`
}

type EmbedResult struct {
	Vector       []float32 `json:"vector"`
	PromptTokens int       `json:"prompt_tokens"`
}

type CancelParams struct {
	Target string `json:"target"` // id of the in-flight request to cancel
}

type SetIdleTimeoutParams struct {
	Seconds int64 `json:"seconds"`
}

type ApplyKeepAliveParams struct {
	Seconds int64 `json:"seconds"`
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…"
}

// Package supervisor owns the worker subprocess lifecycle on the HOST side.
// It spawns the worker, multiplexes concurrent RPC calls over one Unix socket,
// detects crashes, and optionally auto-respawns with a quota.
//
// One worker per host: all HTTP requests funnel through a single inference
// process. Multi-worker (per-model) is a follow-up — architecture is already
// laid out for it (Pool would own a map of Worker by model id).
package supervisor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

// Options controls how the supervisor spawns and supervises the worker.
type Options struct {
	// WorkerBinary is the path to the orchestra-worker executable. If empty,
	// we look for `orchestra-worker` next to the running binary.
	WorkerBinary string
	// SocketDir holds the Unix socket file. Defaults to os.TempDir().
	SocketDir string
	// Env vars injected into the worker process.
	ExtraEnv []string
	// RespawnQuota limits automatic restarts per RespawnWindow. When the quota
	// is exhausted, further crashes require manual re-Spawn.
	RespawnQuota  int
	RespawnWindow time.Duration
	// BootTimeout bounds how long we wait for the worker to accept our dial.
	BootTimeout time.Duration
}

func defaultOptions() Options {
	return Options{
		RespawnQuota:  3,
		RespawnWindow: 10 * time.Minute,
		BootTimeout:   10 * time.Second,
	}
}

// ── Worker: one subprocess, bidirectional RPC ────────────────────────────────

type Worker struct {
	opts Options

	mu       sync.RWMutex
	cmd      *exec.Cmd
	conn     net.Conn
	codec    *rpc.Codec
	ready    atomic.Bool
	pending  map[string]*pendingCall
	socket   string
	stderr   strings.Builder
	crashErr error
	// Crash history for the respawn quota.
	crashes []time.Time
	// Listeners get notified when the worker exits (either cleanly or on crash).
	onExit []func(err error)
}

type pendingCall struct {
	// Final is closed when a KindFinal frame arrives (or the worker dies).
	// Receivers read a single *rpc.Envelope or nil (on crash).
	final chan *rpc.Envelope
	// Chunks carries KindChunk frames for streaming calls. Nil for single-shot.
	chunks chan *rpc.Envelope
}

func NewWorker(opts Options) *Worker {
	def := defaultOptions()
	if opts.RespawnQuota == 0 {
		opts.RespawnQuota = def.RespawnQuota
	}
	if opts.RespawnWindow == 0 {
		opts.RespawnWindow = def.RespawnWindow
	}
	if opts.BootTimeout == 0 {
		opts.BootTimeout = def.BootTimeout
	}
	return &Worker{
		opts:    opts,
		pending: make(map[string]*pendingCall),
	}
}

// IsReady reports whether the worker is currently spawned and connected.
func (w *Worker) IsReady() bool { return w.ready.Load() }

// OnExit registers a callback invoked when the worker process exits. The
// passed error is nil for graceful shutdown, rpc.ErrWorkerCrashed-like for
// abrupt deaths.
func (w *Worker) OnExit(fn func(err error)) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.onExit = append(w.onExit, fn)
}

// Spawn starts the worker subprocess and dials its Unix socket. Returns once
// a ping RPC has succeeded (so callers can safely issue requests after).
func (w *Worker) Spawn() error {
	w.mu.Lock()
	if w.ready.Load() {
		w.mu.Unlock()
		return nil
	}
	bin := w.opts.WorkerBinary
	if bin == "" {
		selfDir, _ := os.Executable()
		bin = filepath.Join(filepath.Dir(selfDir), "orchestra-worker")
	}
	if _, err := os.Stat(bin); err != nil {
		w.mu.Unlock()
		return fmt.Errorf("worker binary not found at %s: %w", bin, err)
	}

	sockDir := w.opts.SocketDir
	if sockDir == "" {
		sockDir = os.TempDir()
	}
	socketPath := filepath.Join(sockDir, fmt.Sprintf("orchestra-worker-%d-%s.sock", os.Getpid(), uuid.New().String()[:8]))

	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(),
		"ORCHESTRA_WORKER_SOCKET="+socketPath,
	)
	cmd.Env = append(cmd.Env, w.opts.ExtraEnv...)
	// Capture stderr so we can include it in crash reports.
	stderr, err := cmd.StderrPipe()
	if err != nil {
		w.mu.Unlock()
		return err
	}
	cmd.Stdout = os.Stdout // worker logs to stderr, so stdout is rarely used

	if err := cmd.Start(); err != nil {
		w.mu.Unlock()
		return fmt.Errorf("start worker: %w", err)
	}
	w.cmd = cmd
	w.socket = socketPath
	w.mu.Unlock()

	// Drain stderr into our buffer so it appears in crash reports.
	go w.drainStderr(stderr)

	// Watch the process — when it exits, we need to clean up and notify listeners.
	go w.waitAndReport()

	// Dial the socket. The worker needs a moment to create it.
	conn, err := dialWithRetry(socketPath, w.opts.BootTimeout)
	if err != nil {
		_ = cmd.Process.Kill()
		return fmt.Errorf("dial worker: %w", err)
	}

	w.mu.Lock()
	w.conn = conn
	w.codec = rpc.NewCodec(conn)
	w.ready.Store(true)
	w.mu.Unlock()

	go w.readLoop()

	// Sanity ping — ensures the codec + dispatch loop are alive before we
	// let callers in.
	pingCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := w.Call(pingCtx, rpc.MethodPing, nil); err != nil {
		_ = w.Shutdown(context.Background())
		return fmt.Errorf("worker ping failed: %w", err)
	}
	slog.Info("worker spawned", "pid", cmd.Process.Pid, "socket", socketPath)
	return nil
}

func dialWithRetry(path string, timeout time.Duration) (net.Conn, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		c, err := net.Dial("unix", path)
		if err == nil {
			return c, nil
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	return nil, lastErr
}

func (w *Worker) drainStderr(r io.Reader) {
	buf := make([]byte, 4096)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			// Forward to our own stderr so `View → Output` in VS Code shows it.
			os.Stderr.Write(buf[:n])
			w.mu.Lock()
			// Cap the buffer to avoid unbounded growth during normal logging.
			if w.stderr.Len() < 64*1024 {
				w.stderr.Write(buf[:n])
			}
			w.mu.Unlock()
		}
		if err != nil {
			return
		}
	}
}

// waitAndReport blocks on cmd.Wait and then tears down pending calls.
func (w *Worker) waitAndReport() {
	err := w.cmd.Wait()
	w.ready.Store(false)
	slog.Warn("worker exited", "error", err, "pid", w.cmd.ProcessState.Pid())

	// Classify: clean vs crash. We treat anything other than exit 0 / SIGTERM
	// as a crash.
	var exitErr error
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			if ws, ok := ee.ProcessState.Sys().(syscall.WaitStatus); ok {
				if ws.Signaled() && (ws.Signal() == syscall.SIGTERM || ws.Signal() == syscall.SIGINT) {
					// Graceful — we asked for it.
				} else {
					exitErr = fmt.Errorf("%w: %v", rpc.ErrWorkerCrashed, err)
					w.recordCrash()
				}
			} else {
				exitErr = fmt.Errorf("%w: %v", rpc.ErrWorkerCrashed, err)
				w.recordCrash()
			}
		} else {
			exitErr = err
		}
	}

	// Fail all pending calls.
	w.mu.Lock()
	w.crashErr = exitErr
	pend := w.pending
	w.pending = make(map[string]*pendingCall)
	listeners := make([]func(error), len(w.onExit))
	copy(listeners, w.onExit)
	w.mu.Unlock()
	for _, pc := range pend {
		close(pc.final)
		if pc.chunks != nil {
			close(pc.chunks)
		}
	}
	for _, fn := range listeners {
		fn(exitErr)
	}
}

func (w *Worker) recordCrash() {
	w.mu.Lock()
	defer w.mu.Unlock()
	now := time.Now()
	cutoff := now.Add(-w.opts.RespawnWindow)
	pruned := w.crashes[:0]
	for _, t := range w.crashes {
		if t.After(cutoff) {
			pruned = append(pruned, t)
		}
	}
	w.crashes = append(pruned, now)
}

// CanRespawn returns true if the worker has crashed fewer times than the
// quota in the current window — callers can decide whether to auto-restart.
func (w *Worker) CanRespawn() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.crashes) < w.opts.RespawnQuota
}

// LastStderr returns the tail of the worker's stderr (up to 64 KB). Useful
// to surface "fatal error: ..." from the Go runtime crash dump.
func (w *Worker) LastStderr() string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.stderr.String()
}

// ── readLoop: demultiplexes frames back to their pending calls ──────────────

func (w *Worker) readLoop() {
	for {
		env, err := w.codec.Read()
		if err != nil {
			if !errors.Is(err, io.EOF) && !errors.Is(err, net.ErrClosed) {
				slog.Warn("worker read loop error", "error", err)
			}
			return
		}
		w.mu.Lock()
		pc, ok := w.pending[env.ID]
		if ok && env.Kind == rpc.KindFinal {
			delete(w.pending, env.ID)
		}
		w.mu.Unlock()

		if !ok {
			slog.Warn("frame for unknown id", "id", env.ID, "kind", env.Kind)
			continue
		}
		switch env.Kind {
		case rpc.KindChunk:
			if pc.chunks != nil {
				pc.chunks <- env
			}
		case rpc.KindFinal:
			pc.final <- env
			close(pc.final)
			if pc.chunks != nil {
				close(pc.chunks)
			}
		default:
			slog.Warn("unexpected kind", "kind", env.Kind, "id", env.ID)
		}
	}
}

// ── Public RPC calls ─────────────────────────────────────────────────────────

// Call sends a single-shot RPC and waits for the final envelope. Returns the
// decoded Result (caller unmarshals) or a typed error.
func (w *Worker) Call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	if !w.ready.Load() {
		w.mu.RLock()
		err := w.crashErr
		w.mu.RUnlock()
		if err != nil {
			return nil, err
		}
		return nil, errors.New("worker not ready")
	}
	id := uuid.New().String()
	pc := &pendingCall{final: make(chan *rpc.Envelope, 1)}
	w.mu.Lock()
	w.pending[id] = pc
	w.mu.Unlock()

	var rawParams json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return nil, err
		}
		rawParams = b
	}
	if err := w.codec.Write(&rpc.Envelope{ID: id, Kind: rpc.KindRequest, Method: method, Params: rawParams}); err != nil {
		w.mu.Lock()
		delete(w.pending, id)
		w.mu.Unlock()
		return nil, err
	}

	select {
	case env, ok := <-pc.final:
		if !ok {
			// Channel closed without a frame → worker died mid-call.
			w.mu.RLock()
			err := w.crashErr
			w.mu.RUnlock()
			if err == nil {
				err = rpc.ErrWorkerCrashed
			}
			return nil, err
		}
		if env.Error != nil {
			return nil, env.Error
		}
		return env.Result, nil
	case <-ctx.Done():
		// Ask the worker to cancel the call — otherwise a decoding loop keeps
		// burning CPU. Fire-and-forget; the cleanup happens when the final
		// frame eventually arrives (or the worker dies).
		_ = w.codec.Write(&rpc.Envelope{
			ID:     uuid.New().String(),
			Kind:   rpc.KindRequest,
			Method: rpc.MethodCancel,
			Params: mustJSON(rpc.CancelParams{Target: id}),
		})
		return nil, ctx.Err()
	}
}

// CallStream sends a streaming RPC and returns a channel of envelopes. The
// last envelope has Kind=KindFinal; intermediate are KindChunk.
func (w *Worker) CallStream(ctx context.Context, method string, params any) (<-chan *rpc.Envelope, error) {
	if !w.ready.Load() {
		return nil, errors.New("worker not ready")
	}
	id := uuid.New().String()
	pc := &pendingCall{
		final:  make(chan *rpc.Envelope, 1),
		chunks: make(chan *rpc.Envelope, 32),
	}
	w.mu.Lock()
	w.pending[id] = pc
	w.mu.Unlock()

	rawParams, _ := json.Marshal(params)
	if err := w.codec.Write(&rpc.Envelope{ID: id, Kind: rpc.KindRequest, Method: method, Params: rawParams}); err != nil {
		w.mu.Lock()
		delete(w.pending, id)
		w.mu.Unlock()
		return nil, err
	}

	out := make(chan *rpc.Envelope, 32)
	go func() {
		defer close(out)
		for {
			select {
			case ch, ok := <-pc.chunks:
				if !ok {
					return
				}
				out <- ch
			case env, ok := <-pc.final:
				if !ok {
					return
				}
				out <- env
				return
			case <-ctx.Done():
				_ = w.codec.Write(&rpc.Envelope{
					ID:     uuid.New().String(),
					Kind:   rpc.KindRequest,
					Method: rpc.MethodCancel,
					Params: mustJSON(rpc.CancelParams{Target: id}),
				})
				// Keep draining until the worker sends its final so we clean
				// up the pending map. If worker dies, channels close.
			}
		}
	}()
	return out, nil
}

// Shutdown asks the worker to exit gracefully, then kills if it doesn't.
func (w *Worker) Shutdown(ctx context.Context) error {
	if !w.ready.Load() {
		return nil
	}
	shutdownCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	_, _ = w.Call(shutdownCtx, rpc.MethodShutdown, nil)

	// Give it a moment to exit on its own.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if !w.ready.Load() {
			return nil
		}
		time.Sleep(50 * time.Millisecond)
	}
	w.mu.RLock()
	proc := w.cmd.Process
	w.mu.RUnlock()
	if proc != nil {
		_ = proc.Kill()
	}
	return nil
}

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

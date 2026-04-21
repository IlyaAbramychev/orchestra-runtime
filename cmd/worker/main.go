// Package main is the orchestra-runtime WORKER process. It hosts the
// CGo-linked llama.cpp engine and receives RPC calls from the host over a
// Unix socket. When the host decides it needs inference, it spawns this
// binary with env vars telling it which socket to listen on.
//
// Crash isolation: if llama.cpp SIGSEGVs (e.g. unstable quant on Metal), only
// this process dies. The host's `cmd.Wait()` returns, it notifies pending
// callers with rpc.ErrWorkerCrashed, and can respawn if the user asks.
//
// Env vars:
//
//	ORCHESTRA_WORKER_SOCKET  path to Unix socket (required)
//	ORCHESTRA_LOG_LEVEL      debug|info|warn|error (default info)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"runtime/debug"
	"sync"
	"syscall"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

func main() {
	debug.SetPanicOnFault(true)

	socketPath := os.Getenv("ORCHESTRA_WORKER_SOCKET")
	if socketPath == "" {
		fmt.Fprintln(os.Stderr, "ORCHESTRA_WORKER_SOCKET is required")
		os.Exit(2)
	}

	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: parseLogLevel(os.Getenv("ORCHESTRA_LOG_LEVEL")),
	})))

	// Clean up any stale socket from a previous crashed run.
	_ = os.Remove(socketPath)

	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		slog.Error("listen failed", "socket", socketPath, "error", err)
		os.Exit(1)
	}
	defer ln.Close()
	defer os.Remove(socketPath)

	eng := engine.New()
	eng.InitBackend()
	defer eng.Close()

	w := &worker{
		engine:    eng,
		streamCtx: make(map[string]context.CancelFunc),
	}

	// Graceful shutdown on SIGTERM from supervisor.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("worker shutting down")
		cancel()
		ln.Close()
	}()

	slog.Info("worker ready", "socket", socketPath, "pid", os.Getpid())

	// We only ever expect ONE connection (from our supervisor). Accept in a
	// loop anyway so reconnects work after a supervisor restart.
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			slog.Error("accept failed", "error", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		slog.Info("supervisor connected")
		w.serve(ctx, conn)
		slog.Info("supervisor disconnected")
	}
}

// ── worker state ─────────────────────────────────────────────────────────────

type worker struct {
	engine *engine.Engine

	mu        sync.Mutex
	streamCtx map[string]context.CancelFunc // id → cancel fn for in-flight streams
}

func (w *worker) serve(ctx context.Context, conn net.Conn) {
	codec := rpc.NewCodec(conn)
	defer codec.Close()

	// Read requests on the main goroutine; spawn a goroutine per request so
	// long streams don't block new requests (mostly used for cancel).
	for {
		env, err := codec.Read()
		if err != nil {
			if errors.Is(err, net.ErrClosed) || ctx.Err() != nil {
				return
			}
			// EOF from supervisor is normal termination of the connection.
			slog.Info("read done", "error", err)
			return
		}
		if env.Kind != rpc.KindRequest {
			slog.Warn("ignoring non-request frame", "kind", env.Kind)
			continue
		}
		go w.dispatch(ctx, codec, env)
	}
}

// dispatch routes a single incoming request to the right handler. Streaming
// methods emit multiple chunks + a final; others emit exactly one final.
func (w *worker) dispatch(ctx context.Context, c *rpc.Codec, env *rpc.Envelope) {
	// Safety net: if a handler panics (Go-level), don't take the worker
	// down — reply with an error and let the supervisor decide. C-level
	// faults still crash the process; that's the whole point of this worker.
	defer func() {
		if r := recover(); r != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeInternal, fmt.Sprintf("worker panic: %v", r)))
		}
	}()

	switch env.Method {
	case rpc.MethodPing:
		_ = c.Write(finalOK(env.ID, map[string]any{"pong": true}))

	case rpc.MethodStatus:
		_ = c.Write(finalOK(env.ID, rpc.StatusResult{
			State:         w.engine.State(),
			ModelID:       w.engine.LoadedModelID(),
			IsLoaded:      w.engine.IsLoaded(),
			IdleTimeoutNs: int64(w.engine.IdleTimeout()),
		}))

	case rpc.MethodLoadModel:
		var p rpc.LoadParams
		if err := json.Unmarshal(env.Params, &p); err != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
			return
		}
		opts := engine.LoadOptions{
			GPULayers:     p.GPULayers,
			CtxSize:       p.CtxSize,
			Threads:       p.Threads,
			BatchSize:     p.BatchSize,
			RopeFreqBase:  p.RopeFreqBase,
			RopeFreqScale: p.RopeFreqScale,
			FlashAttn:     p.FlashAttn,
			OffloadKQV:    p.OffloadKQV,
			UseMmap:       p.UseMmap,
			UseMlock:      p.UseMlock,
			TypeK:         p.TypeK,
			TypeV:         p.TypeV,
		}
		if err := w.engine.LoadModel(p.ModelID, p.Path, opts); err != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeLoadFailed, err.Error()))
			return
		}
		_ = c.Write(finalOK(env.ID, map[string]string{"model_id": p.ModelID}))

	case rpc.MethodUnloadModel:
		w.engine.UnloadModel()
		_ = c.Write(finalOK(env.ID, map[string]bool{"ok": true}))

	case rpc.MethodComplete:
		w.handleComplete(ctx, c, env)

	case rpc.MethodCompleteStream:
		w.handleCompleteStream(ctx, c, env)

	case rpc.MethodEmbed:
		w.handleEmbed(ctx, c, env)

	case rpc.MethodSetIdleTimeout:
		var p rpc.SetIdleTimeoutParams
		if err := json.Unmarshal(env.Params, &p); err != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
			return
		}
		w.engine.SetIdleTimeout(time.Duration(p.Seconds) * time.Second)
		_ = c.Write(finalOK(env.ID, map[string]bool{"ok": true}))

	case rpc.MethodApplyKeepAlive:
		var p rpc.ApplyKeepAliveParams
		if err := json.Unmarshal(env.Params, &p); err != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
			return
		}
		secs := p.Seconds
		w.engine.ApplyKeepAlive(&secs)
		_ = c.Write(finalOK(env.ID, map[string]bool{"ok": true}))

	case rpc.MethodCancel:
		var p rpc.CancelParams
		if err := json.Unmarshal(env.Params, &p); err != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
			return
		}
		w.mu.Lock()
		cancel := w.streamCtx[p.Target]
		w.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		_ = c.Write(finalOK(env.ID, map[string]bool{"ok": true}))

	case rpc.MethodShutdown:
		_ = c.Write(finalOK(env.ID, map[string]bool{"ok": true}))
		// Exit the process cleanly. defer'd teardown in main() runs.
		go func() {
			time.Sleep(50 * time.Millisecond) // let the reply flush
			os.Exit(0)
		}()

	default:
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeUnsupported, "unknown method: "+env.Method))
	}
}

func (w *worker) handleComplete(ctx context.Context, c *rpc.Codec, env *rpc.Envelope) {
	var p rpc.CompleteParams
	if err := json.Unmarshal(env.Params, &p); err != nil {
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
		return
	}
	msgs := toEngineMessages(p.Messages)
	params := toEngineParams(p.Params)

	callCtx, cancel := context.WithCancel(ctx)
	w.mu.Lock()
	w.streamCtx[env.ID] = cancel
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		delete(w.streamCtx, env.ID)
		w.mu.Unlock()
		cancel()
	}()

	result, err := w.engine.Complete(callCtx, msgs, params)
	if err != nil {
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeInference, err.Error()))
		return
	}
	_ = c.Write(finalOK(env.ID, rpc.CompleteResult{
		Text:             result.Text,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
		FinishReason:     result.FinishReason,
		Timings: rpc.Timings{
			TotalNs:      result.Timings.TotalNs,
			PromptEvalNs: result.Timings.PromptEvalNs,
			EvalNs:       result.Timings.EvalNs,
		},
	}))
}

func (w *worker) handleCompleteStream(ctx context.Context, c *rpc.Codec, env *rpc.Envelope) {
	var p rpc.CompleteParams
	if err := json.Unmarshal(env.Params, &p); err != nil {
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
		return
	}
	msgs := toEngineMessages(p.Messages)
	params := toEngineParams(p.Params)

	callCtx, cancel := context.WithCancel(ctx)
	w.mu.Lock()
	w.streamCtx[env.ID] = cancel
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		delete(w.streamCtx, env.ID)
		w.mu.Unlock()
		cancel()
	}()

	ch, err := w.engine.CompleteStream(callCtx, msgs, params)
	if err != nil {
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeInference, err.Error()))
		return
	}
	for chunk := range ch {
		if chunk.Err != nil {
			_ = c.Write(finalErr(env.ID, rpc.ErrCodeInference, chunk.Err.Error()))
			return
		}
		payload := rpc.StreamChunk{
			Text:             chunk.Text,
			Done:             chunk.Done,
			FinishReason:     chunk.FinishReason,
			PromptTokens:     chunk.PromptTokens,
			CompletionTokens: chunk.CompletionTokens,
			Timings: rpc.Timings{
				TotalNs:      chunk.Timings.TotalNs,
				PromptEvalNs: chunk.Timings.PromptEvalNs,
				EvalNs:       chunk.Timings.EvalNs,
			},
		}
		kind := rpc.KindChunk
		if chunk.Done {
			kind = rpc.KindFinal
		}
		raw, _ := json.Marshal(payload)
		_ = c.Write(&rpc.Envelope{ID: env.ID, Kind: kind, Result: raw})
		if chunk.Done {
			return
		}
	}
	// Channel closed without a done chunk — emit a synthetic final so the
	// supervisor never hangs waiting.
	_ = c.Write(finalOK(env.ID, rpc.StreamChunk{Done: true, FinishReason: "stop"}))
}

func (w *worker) handleEmbed(ctx context.Context, c *rpc.Codec, env *rpc.Envelope) {
	var p rpc.EmbedParams
	if err := json.Unmarshal(env.Params, &p); err != nil {
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeInvalid, err.Error()))
		return
	}
	res, err := w.engine.Embed(ctx, p.Text, p.Normalize)
	if err != nil {
		_ = c.Write(finalErr(env.ID, rpc.ErrCodeInference, err.Error()))
		return
	}
	_ = c.Write(finalOK(env.ID, rpc.EmbedResult{
		Vector:       res.Vector,
		PromptTokens: res.PromptTokens,
	}))
}

// ── helpers ──────────────────────────────────────────────────────────────────

func finalOK(id string, payload any) *rpc.Envelope {
	raw, _ := json.Marshal(payload)
	return &rpc.Envelope{ID: id, Kind: rpc.KindFinal, Result: raw}
}

func finalErr(id, code, msg string) *rpc.Envelope {
	return &rpc.Envelope{ID: id, Kind: rpc.KindFinal, Error: &rpc.Error{Code: code, Message: msg}}
}

func toEngineMessages(in []rpc.ChatMessage) []engine.ChatMessage {
	out := make([]engine.ChatMessage, len(in))
	for i, m := range in {
		out[i] = engine.ChatMessage{Role: m.Role, Content: m.Content}
	}
	return out
}

func toEngineParams(p rpc.CompletionParams) engine.CompletionParams {
	return engine.CompletionParams{
		MaxTokens:        p.MaxTokens,
		Temperature:      p.Temperature,
		TopK:             p.TopK,
		TopP:             p.TopP,
		MinP:             p.MinP,
		TypicalP:         p.TypicalP,
		RepeatPenalty:    p.RepeatPenalty,
		RepeatLastN:      p.RepeatLastN,
		FrequencyPenalty: p.FrequencyPenalty,
		PresencePenalty:  p.PresencePenalty,
		Seed:             p.Seed,
		Mirostat:         p.Mirostat,
		MirostatTau:      p.MirostatTau,
		MirostatEta:      p.MirostatEta,
		Stop:             p.Stop,
		RawPrompt:        p.RawPrompt,
	}
}

func parseLogLevel(s string) slog.Level {
	switch s {
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

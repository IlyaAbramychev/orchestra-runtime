package supervisor

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

// Remote implements engine.Backend by forwarding every call to an
// orchestra-worker subprocess via the Worker supervisor. The host process
// itself is CGo-free — when llama.cpp crashes, only the worker dies and we
// can respawn.
//
// Local bookkeeping vs. remote truth:
//   - `IsLoaded`, `LoadedModelID`, `State` are cached locally so we don't
//     RPC on every HTTP request. Updated on LoadModel/UnloadModel or via
//     refreshStatus().
//   - `IdleTimeout` is mirrored locally too — set at startup and whenever
//     SetIdleTimeout / ApplyKeepAlive are called.
type Remote struct {
	worker *Worker

	// Cached local state. Updated inside LoadModel / UnloadModel / status
	// refresh; read by hot paths (State/IsLoaded) without RPC.
	modelID     atomic.Value // string
	loaded      atomic.Bool
	state       atomic.Value // string
	idleTimeout atomic.Int64 // ns
}

// NewRemote wraps a Worker supervisor. The Worker can be in any state; Remote
// defers spawning to either an explicit Spawn call or the first Load.
func NewRemote(w *Worker) *Remote {
	r := &Remote{worker: w}
	r.modelID.Store("")
	r.state.Store(engine.StateIdle)
	// When the worker exits (crash or graceful), flip local state to "idle"
	// so callers can re-LoadModel and it will auto-respawn.
	w.OnExit(func(err error) {
		r.loaded.Store(false)
		r.modelID.Store("")
		if err != nil {
			r.state.Store(engine.StateError)
		} else {
			r.state.Store(engine.StateIdle)
		}
	})
	return r
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

func (r *Remote) InitBackend() {
	// No-op. The worker calls engine.InitBackend() on startup; when a Load
	// request arrives we Spawn() the worker if it's not already running.
}

func (r *Remote) FreeBackend() {
	// Best-effort graceful teardown; ignored if the worker is already gone.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = r.worker.Shutdown(ctx)
}

func (r *Remote) Close() error {
	return r.worker.Shutdown(context.Background())
}

// ── Model management ─────────────────────────────────────────────────────────

func (r *Remote) LoadModel(modelID, path string, opts engine.LoadOptions) error {
	// Lazy spawn: first model load brings up the worker. Later loads reuse
	// the same process (llama.cpp inside worker will unload the old model
	// inside engine.LoadModel).
	if !r.worker.IsReady() {
		if err := r.worker.Spawn(); err != nil {
			return fmt.Errorf("spawn worker: %w", err)
		}
	}
	r.state.Store(engine.StateLoading)
	_, err := r.worker.Call(context.Background(), rpc.MethodLoadModel, rpc.LoadParams{
		ModelID:       modelID,
		Path:          path,
		GPULayers:     opts.GPULayers,
		CtxSize:       opts.CtxSize,
		Threads:       opts.Threads,
		BatchSize:     opts.BatchSize,
		RopeFreqBase:  opts.RopeFreqBase,
		RopeFreqScale: opts.RopeFreqScale,
		FlashAttn:     opts.FlashAttn,
		OffloadKQV:    opts.OffloadKQV,
		UseMmap:       opts.UseMmap,
		UseMlock:      opts.UseMlock,
		TypeK:         opts.TypeK,
		TypeV:         opts.TypeV,
	})
	if err != nil {
		r.state.Store(engine.StateError)
		return err
	}
	r.modelID.Store(modelID)
	r.loaded.Store(true)
	r.state.Store(engine.StateReady)
	return nil
}

func (r *Remote) UnloadModel() {
	if !r.worker.IsReady() {
		return
	}
	_, _ = r.worker.Call(context.Background(), rpc.MethodUnloadModel, nil)
	r.modelID.Store("")
	r.loaded.Store(false)
	r.state.Store(engine.StateIdle)
}

func (r *Remote) IsLoaded() bool { return r.loaded.Load() }

func (r *Remote) LoadedModelID() string {
	if v, ok := r.modelID.Load().(string); ok {
		return v
	}
	return ""
}

func (r *Remote) State() string {
	if v, ok := r.state.Load().(string); ok {
		return v
	}
	return engine.StateIdle
}

// ModelDesc returns a cached description. First call queries the worker; we
// don't cache long enough to worry about staleness because it never changes
// for a given loaded model.
func (r *Remote) ModelDesc() string {
	// The worker doesn't currently expose this via status; keep it empty on
	// the proxy side — not used for anything critical.
	return ""
}

// ── Inference ────────────────────────────────────────────────────────────────

func (r *Remote) Complete(ctx context.Context, messages []engine.ChatMessage, params engine.CompletionParams) (*engine.CompletionResult, error) {
	if !r.worker.IsReady() {
		return nil, fmt.Errorf("worker not ready")
	}
	raw, err := r.worker.Call(ctx, rpc.MethodComplete, rpc.CompleteParams{
		Messages: toRPCMessages(messages),
		Params:   toRPCParams(params),
	})
	if err != nil {
		return nil, err
	}
	var res rpc.CompleteResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, err
	}
	return &engine.CompletionResult{
		Text:             res.Text,
		PromptTokens:     res.PromptTokens,
		CompletionTokens: res.CompletionTokens,
		FinishReason:     res.FinishReason,
		Timings: engine.Timings{
			TotalNs:      res.Timings.TotalNs,
			PromptEvalNs: res.Timings.PromptEvalNs,
			EvalNs:       res.Timings.EvalNs,
		},
	}, nil
}

func (r *Remote) CompleteStream(ctx context.Context, messages []engine.ChatMessage, params engine.CompletionParams) (<-chan engine.CompletionChunk, error) {
	if !r.worker.IsReady() {
		return nil, fmt.Errorf("worker not ready")
	}
	frames, err := r.worker.CallStream(ctx, rpc.MethodCompleteStream, rpc.CompleteParams{
		Messages: toRPCMessages(messages),
		Params:   toRPCParams(params),
	})
	if err != nil {
		return nil, err
	}

	out := make(chan engine.CompletionChunk, 32)
	go func() {
		defer close(out)
		for env := range frames {
			if env.Error != nil {
				out <- engine.CompletionChunk{Err: env.Error}
				return
			}
			var chunk rpc.StreamChunk
			if env.Result != nil {
				if err := json.Unmarshal(env.Result, &chunk); err != nil {
					out <- engine.CompletionChunk{Err: err}
					return
				}
			}
			out <- engine.CompletionChunk{
				Text:             chunk.Text,
				Done:             chunk.Done,
				FinishReason:     chunk.FinishReason,
				PromptTokens:     chunk.PromptTokens,
				CompletionTokens: chunk.CompletionTokens,
				Timings: engine.Timings{
					TotalNs:      chunk.Timings.TotalNs,
					PromptEvalNs: chunk.Timings.PromptEvalNs,
					EvalNs:       chunk.Timings.EvalNs,
				},
			}
			if chunk.Done {
				return
			}
		}
		// Channel closed without done → worker died mid-stream.
		slog.Warn("stream ended abruptly", "stderr", r.worker.LastStderr())
		out <- engine.CompletionChunk{Err: rpc.ErrWorkerCrashed}
	}()
	return out, nil
}

func (r *Remote) Embed(ctx context.Context, text string, normalize bool) (*engine.EmbeddingResult, error) {
	if !r.worker.IsReady() {
		return nil, fmt.Errorf("worker not ready")
	}
	raw, err := r.worker.Call(ctx, rpc.MethodEmbed, rpc.EmbedParams{
		Text:      text,
		Normalize: normalize,
	})
	if err != nil {
		return nil, err
	}
	var res rpc.EmbedResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, err
	}
	return &engine.EmbeddingResult{
		Vector:       res.Vector,
		PromptTokens: res.PromptTokens,
	}, nil
}

// ── Idle/keep-alive ──────────────────────────────────────────────────────────

func (r *Remote) SetIdleTimeout(d time.Duration) {
	r.idleTimeout.Store(int64(d))
	if !r.worker.IsReady() {
		return
	}
	_, _ = r.worker.Call(context.Background(), rpc.MethodSetIdleTimeout, rpc.SetIdleTimeoutParams{
		Seconds: int64(d.Seconds()),
	})
}

func (r *Remote) IdleTimeout() time.Duration {
	return time.Duration(r.idleTimeout.Load())
}

func (r *Remote) ApplyKeepAlive(seconds *int64) {
	if seconds == nil {
		return
	}
	if !r.worker.IsReady() {
		return
	}
	_, _ = r.worker.Call(context.Background(), rpc.MethodApplyKeepAlive, rpc.ApplyKeepAliveParams{
		Seconds: *seconds,
	})
	// Mirror locally.
	if *seconds > 0 {
		r.idleTimeout.Store(int64(time.Duration(*seconds) * time.Second))
	} else if *seconds == 0 {
		// 0 → unload right now
		r.modelID.Store("")
		r.loaded.Store(false)
		r.state.Store(engine.StateIdle)
	}
}

// MarkUsed is a no-op on the remote side: the worker already updates its own
// lastUsedAt on every inference call.
func (r *Remote) MarkUsed() {}

// ── Type conversions ─────────────────────────────────────────────────────────

func toRPCMessages(in []engine.ChatMessage) []rpc.ChatMessage {
	out := make([]rpc.ChatMessage, len(in))
	for i, m := range in {
		out[i] = rpc.ChatMessage{Role: m.Role, Content: m.Content}
	}
	return out
}

func toRPCParams(p engine.CompletionParams) rpc.CompletionParams {
	return rpc.CompletionParams{
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

// Compile-time interface check.
var _ engine.Backend = (*Remote)(nil)

package service

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/operium/orchestra-runtime/internal/engine"
)

// RuntimeScheduler serializes all access to the single llama context.
// Inference, embeddings, load, unload and immediate keep_alive unloads must
// pass through this layer so lifecycle operations cannot race active work.
type RuntimeScheduler struct {
	engine        engine.Backend
	sem           chan struct{}
	maxQueue      int
	mu            sync.Mutex
	queueLen      int
	activeState   string
	activeModelID string
}

type RuntimeSnapshot struct {
	State         string
	ActiveModelID string
	QueueDepth    int
}

func NewRuntimeScheduler(eng engine.Backend, maxQueue int) *RuntimeScheduler {
	if maxQueue <= 0 {
		maxQueue = 1
	}
	return &RuntimeScheduler{
		engine:   eng,
		sem:      make(chan struct{}, 1),
		maxQueue: maxQueue,
	}
}

func (s *RuntimeScheduler) Backend() engine.Backend {
	return s.engine
}

// QueueDepth returns the number of waiting operations. The active operation is
// intentionally excluded so callers can distinguish backlog from utilization.
func (s *RuntimeScheduler) QueueDepth() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queueLen
}

func (s *RuntimeScheduler) Snapshot() RuntimeSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.activeState
	if state == "" {
		state = s.engine.State()
	}
	return RuntimeSnapshot{
		State:         state,
		ActiveModelID: s.activeModelID,
		QueueDepth:    s.queueLen,
	}
}

// State returns the active lifecycle operation when the scheduler is using the
// runtime slot, otherwise the backend's own steady-state lifecycle.
func (s *RuntimeScheduler) State() string {
	return s.Snapshot().State
}

func (s *RuntimeScheduler) ActiveModelID() string {
	return s.Snapshot().ActiveModelID
}

func (s *RuntimeScheduler) acquire(ctx context.Context) (func(), error) {
	return s.acquireFor(ctx, "", "")
}

func (s *RuntimeScheduler) acquireFor(ctx context.Context, state, modelID string) (func(), error) {
	select {
	case s.sem <- struct{}{}:
		s.setActive(state, modelID)
		return func() {
			s.clearActive()
			<-s.sem
		}, nil
	default:
	}

	s.mu.Lock()
	if s.queueLen >= s.maxQueue {
		s.mu.Unlock()
		return nil, fmt.Errorf("runtime queue full")
	}
	s.queueLen++
	s.mu.Unlock()

	dequeued := false
	defer func() {
		if !dequeued {
			s.mu.Lock()
			s.queueLen--
			s.mu.Unlock()
		}
	}()

	select {
	case s.sem <- struct{}{}:
		dequeued = true
		s.mu.Lock()
		s.queueLen--
		s.mu.Unlock()
		s.setActive(state, modelID)
		return func() {
			s.clearActive()
			<-s.sem
		}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *RuntimeScheduler) setActive(state, modelID string) {
	if state == "" && modelID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeState = state
	s.activeModelID = modelID
}

func (s *RuntimeScheduler) clearActive() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeState = ""
	s.activeModelID = ""
}

func (s *RuntimeScheduler) LoadModel(ctx context.Context, modelID, path string, opts engine.LoadOptions) error {
	_, err := s.LoadModelAttempts(ctx, modelID, path, []engine.LoadOptions{opts})
	return err
}

// LoadModelAttempts holds the lifecycle slot for the complete adaptive load.
// No inference or competing model switch can run between OOM retries.
func (s *RuntimeScheduler) LoadModelAttempts(ctx context.Context, modelID, path string, attempts []engine.LoadOptions) (engine.LoadOptions, error) {
	release, err := s.acquireFor(ctx, engine.StateLoading, modelID)
	if err != nil {
		return engine.LoadOptions{}, err
	}
	defer release()
	return loadModelWithAttempts(ctx, s.engine, modelID, path, attempts)
}

func loadModelWithAttempts(ctx context.Context, backend engine.Backend, modelID, path string, attempts []engine.LoadOptions) (engine.LoadOptions, error) {
	if len(attempts) == 0 {
		return engine.LoadOptions{}, fmt.Errorf("no model load attempts configured")
	}
	var lastErr error
	for index, opts := range attempts {
		if err := ctx.Err(); err != nil {
			return engine.LoadOptions{}, err
		}
		if err := backend.LoadModel(modelID, path, opts); err == nil {
			if index > 0 {
				slog.Info("model loaded after automatic memory retry",
					"model", modelID,
					"attempt", index+1,
					"ctx_size", opts.CtxSize,
					"batch_size", opts.BatchSize,
					"type_k", normalizeKVName(opts.TypeK),
					"type_v", normalizeKVName(opts.TypeV),
				)
			}
			return opts, nil
		} else {
			lastErr = err
			if index == len(attempts)-1 || opts.DisableAutoFit || !isRetryableMemoryLoadError(err) {
				break
			}
			next := attempts[index+1]
			slog.Warn("model load hit memory pressure; retrying with a smaller automatic profile",
				"model", modelID,
				"attempt", index+1,
				"error", err,
				"next_ctx_size", next.CtxSize,
				"next_batch_size", next.BatchSize,
				"next_type_k", normalizeKVName(next.TypeK),
				"next_type_v", normalizeKVName(next.TypeV),
			)
		}
	}
	return engine.LoadOptions{}, lastErr
}

func isRetryableMemoryLoadError(err error) bool {
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	for _, marker := range []string{
		"out of memory",
		"cannot allocate memory",
		"failed to allocate",
		"allocation failed",
		"insufficient memory",
		"resource shortage",
		"erroroutofmemory",
		"memory exhausted",
	} {
		if strings.Contains(message, marker) {
			return true
		}
	}
	return false
}

func (s *RuntimeScheduler) UnloadModel(ctx context.Context) error {
	release, err := s.acquireFor(ctx, engine.StateUnloading, s.engine.LoadedModelID())
	if err != nil {
		return err
	}
	defer release()
	s.engine.UnloadModel()
	return nil
}

func (s *RuntimeScheduler) ApplyKeepAlive(seconds *int64) {
	if seconds == nil {
		return
	}
	if *seconds != 0 {
		s.engine.ApplyKeepAlive(seconds)
		return
	}
	go func() {
		_ = s.UnloadModel(context.Background())
	}()
}

package engine

import (
	"testing"
	"time"
)

func TestIdleTimerMarkUsedExtendsDeadline(t *testing.T) {
	e := New()
	e.state = StateReady
	e.model = &llamaModel{}
	e.SetIdleTimeout(40 * time.Millisecond)
	e.MarkUsed()

	time.Sleep(25 * time.Millisecond)
	e.MarkUsed()
	time.Sleep(25 * time.Millisecond)

	if e.State() != StateReady {
		t.Fatal("idle timer unloaded before extended deadline")
	}

	time.Sleep(35 * time.Millisecond)
	if e.State() != StateIdle {
		t.Fatalf("expected idle unload after extended deadline, got %s", e.State())
	}
}

func TestApplyKeepAliveNegativeDisablesIdleTimer(t *testing.T) {
	e := New()
	e.state = StateReady
	e.model = &llamaModel{}
	e.SetIdleTimeout(20 * time.Millisecond)

	forever := int64(-1)
	e.ApplyKeepAlive(&forever)
	time.Sleep(40 * time.Millisecond)

	if e.State() != StateReady {
		t.Fatal("negative keep_alive should keep model loaded")
	}
	if got := e.IdleTimeout(); got != 0 {
		t.Fatalf("expected idle timeout disabled, got %s", got)
	}
}

func TestStatusReadsDoNotWaitForEngineLock(t *testing.T) {
	e := New()
	e.state = StateReady
	e.model = &llamaModel{}
	e.SetIdleTimeout(time.Minute)
	t.Cleanup(func() { e.SetIdleTimeout(0) })

	// Simulate a long generation or native load holding the engine lock.
	e.mu.Lock()
	defer e.mu.Unlock()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = e.State()
		_ = e.IsLoaded()
		_ = e.LoadedModelID()
		_ = e.LoadedContextSize()
		_ = e.LoadedOptions()
		_ = e.LoadedAt()
		_ = e.LastError()
		_ = e.ModelDesc()
		_ = e.IdleTimeout()
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("status readers blocked on the engine lock")
	}
	if got := e.State(); got != StateReady {
		t.Fatalf("state = %s, want %s", got, StateReady)
	}
}

func TestLoadModelWaitsForBackendInit(t *testing.T) {
	e := New()
	if e.BackendReady() {
		t.Fatal("backend must not be ready before InitBackend")
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- e.LoadModel("missing", "/nonexistent/model.gguf", DefaultLoadOptions())
	}()
	select {
	case err := <-errCh:
		t.Fatalf("LoadModel returned before backend init: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	e.InitBackend()
	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("expected load error for a missing file")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("LoadModel did not proceed after backend init")
	}
	if got := e.State(); got != StateError {
		t.Fatalf("state = %s, want %s", got, StateError)
	}
	if e.LoadedModelID() != "" {
		t.Fatalf("failed load must not report a model id, got %q", e.LoadedModelID())
	}
}

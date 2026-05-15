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

package service

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
)

func TestSystemInfoIncludesBuildMetadata(t *testing.T) {
	oldVersion := Version
	oldBuildCommit := BuildCommit
	oldLlamaCppCommit := LlamaCppCommit
	defer func() {
		Version = oldVersion
		BuildCommit = oldBuildCommit
		LlamaCppCommit = oldLlamaCppCommit
	}()

	Version = "1.2.3"
	BuildCommit = "runtime-sha"
	LlamaCppCommit = "llama-sha"

	info := NewSystemInfo(&autoLoadBackend{}).GetInfo(3)
	if info.Version != "1.2.3" {
		t.Fatalf("version = %q", info.Version)
	}
	if info.BuildCommit != "runtime-sha" {
		t.Fatalf("buildCommit = %q", info.BuildCommit)
	}
	if info.LlamaCppCommit != "llama-sha" {
		t.Fatalf("llamaCppCommit = %q", info.LlamaCppCommit)
	}
	if info.Platform == "" || info.Arch == "" {
		t.Fatalf("expected platform and arch, got platform=%q arch=%q", info.Platform, info.Arch)
	}
	if info.QueueDepth != 3 {
		t.Fatalf("queueDepth = %d", info.QueueDepth)
	}
}

func (b *autoLoadBackend) LoadedContextSize() int {
	if b.loadedID == "" {
		return 0
	}
	return b.lastOpts.CtxSize
}

func (b *autoLoadBackend) LoadedOptions() engine.LoadOptions {
	return b.lastOpts
}

func (b *autoLoadBackend) LoadedAt() time.Time {
	if b.loadedID == "" {
		return time.Time{}
	}
	return time.Date(2026, 5, 15, 9, 0, 0, 0, time.UTC)
}

func (b *autoLoadBackend) LastError() string {
	return ""
}

func TestSystemStatusReportsActualLoadedContext(t *testing.T) {
	backend := &autoLoadBackend{}
	opts := engine.DefaultLoadOptions()
	opts.CtxSize = 12032
	opts.GPULayers = -1
	opts.Threads = 8
	if err := backend.LoadModel("model-1", "/tmp/model.gguf", opts); err != nil {
		t.Fatalf("load: %v", err)
	}

	status := NewSystemInfo(backend).GetStatus()
	if status.Model == nil || *status.Model != "model-1" {
		t.Fatalf("model = %#v", status.Model)
	}
	if status.ContextSize == nil || *status.ContextSize != 12032 {
		t.Fatalf("context size = %#v", status.ContextSize)
	}
	if status.GPULayers != -1 || status.Threads != 8 {
		t.Fatalf("load params = gpu %d threads %d", status.GPULayers, status.Threads)
	}
	if status.LoadedAt == nil {
		t.Fatal("expected loadedAt")
	}
}

func TestSystemStatusReportsNullContextWhenUnloaded(t *testing.T) {
	status := NewSystemInfo(&autoLoadBackend{}).GetStatus()
	if status.Model != nil {
		t.Fatalf("model = %#v", status.Model)
	}
	if status.ContextSize != nil {
		t.Fatalf("context size = %#v", status.ContextSize)
	}
}

func TestSystemInfoUsesActiveSchedulerModelDuringLoad(t *testing.T) {
	backend := &fakeBackend{
		loadStart: make(chan struct{}, 1),
		loadBlock: make(chan struct{}),
	}
	scheduler := NewRuntimeScheduler(backend, 1)
	sysInfo := NewSystemInfo(backend)
	sysInfo.SetScheduler(scheduler)

	done := make(chan error, 1)
	go func() {
		done <- scheduler.LoadModel(context.Background(), "loading-model", "/tmp/model.gguf", engine.DefaultLoadOptions())
	}()

	select {
	case <-backend.loadStart:
	case <-time.After(time.Second):
		t.Fatal("load did not start")
	}

	info := sysInfo.GetInfo(0)
	if info.EngineState != engine.StateLoading {
		t.Fatalf("expected loading state, got %s", info.EngineState)
	}
	if info.CurrentModel == nil || *info.CurrentModel != "loading-model" {
		t.Fatalf("expected current_model loading-model, got %#v", info.CurrentModel)
	}

	close(backend.loadBlock)
	if err := <-done; err != nil {
		t.Fatalf("load failed: %v", err)
	}
}

// startingBackend is an idle backend whose native init has not finished.
type startingBackend struct {
	*autoLoadBackend
	ready bool
}

func (b *startingBackend) State() string      { return engine.StateIdle }
func (b *startingBackend) BackendReady() bool { return b.ready }

func TestSystemInfoReportsStartingUntilBackendReady(t *testing.T) {
	backend := &startingBackend{autoLoadBackend: &autoLoadBackend{}}
	sysInfo := NewSystemInfo(backend)

	if got := sysInfo.BackendState(); got != BackendStarting {
		t.Fatalf("backend state = %q, want %q", got, BackendStarting)
	}
	if got := sysInfo.GetInfo(0).EngineState; got != BackendStarting {
		t.Fatalf("engine_state = %q, want %q", got, BackendStarting)
	}
	if got := sysInfo.GetStatus().State; got != BackendStarting {
		t.Fatalf("status state = %q, want %q", got, BackendStarting)
	}

	backend.ready = true
	if got := sysInfo.BackendState(); got != engine.StateReady {
		t.Fatalf("backend state = %q, want %q", got, engine.StateReady)
	}
	if got := sysInfo.GetInfo(0).EngineState; got != engine.StateIdle {
		t.Fatalf("engine_state = %q, want %q", got, engine.StateIdle)
	}
}

func TestSystemInfoServesStaleHardwareWhileRefreshing(t *testing.T) {
	sysInfo := NewSystemInfo(&autoLoadBackend{})
	sysInfo.hwMaxAge = time.Millisecond
	var calls int32
	release := make(chan struct{})
	sysInfo.sampleHW = func() hwSample {
		n := atomic.AddInt32(&calls, 1)
		if n > 1 {
			<-release
		}
		return hwSample{totalRAM: int64(n), availableRAM: int64(n), sampledAt: time.Now()}
	}

	if info := sysInfo.GetInfo(0); info.TotalRAM != 1 {
		t.Fatalf("first request must sample synchronously, got total=%d", info.TotalRAM)
	}
	time.Sleep(5 * time.Millisecond)

	start := time.Now()
	info := sysInfo.GetInfo(0)
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("stale hardware sample blocked the request for %s", elapsed)
	}
	if info.TotalRAM != 1 {
		t.Fatalf("expected the stale sample while refreshing, got total=%d", info.TotalRAM)
	}
	// A second stale read must not start another refresh.
	_ = sysInfo.GetInfo(0)

	close(release)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if sysInfo.GetInfo(0).TotalRAM == 2 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if got := sysInfo.GetInfo(0).TotalRAM; got < 2 {
		t.Fatalf("background refresh did not publish, total=%d", got)
	}
	if got := atomic.LoadInt32(&calls); got > 3 {
		t.Fatalf("refresh started %d times, want at most one per expiry", got)
	}
}

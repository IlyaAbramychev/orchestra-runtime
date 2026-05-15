package service

import (
	"context"
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

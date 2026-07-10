package service

import (
	"strings"
	"testing"

	"github.com/operium/orchestra-runtime/internal/engine"
)

const testGiB = int64(1024 * 1024 * 1024)

func fixedMemoryPlanner(total, available int64) *LoadPlanner {
	planner := NewLoadPlanner()
	planner.totalRAM = func() int64 { return total }
	planner.availableRAM = func() int64 { return available }
	return planner
}

func TestLoadPlannerUsesQ8BeforeReducingAutomaticContext(t *testing.T) {
	planner := fixedMemoryPlanner(24*testGiB, 10*testGiB)
	opts := engine.DefaultLoadOptions()
	opts.CtxSize = 120832
	opts.BatchSize = 1024

	plan, err := planner.Plan(LoadPlanRequest{
		Options:        opts,
		ModelBytes:     11 * testGiB,
		ProjectorBytes: 1 * testGiB,
		Family:         "llama",
		Vision:         true,
	})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	selected := plan.Attempts[0]
	if selected.Options.CtxSize != opts.CtxSize {
		t.Fatalf("context = %d; want preserved %d", selected.Options.CtxSize, opts.CtxSize)
	}
	if selected.Options.TypeK != "q8_0" || selected.Options.TypeV != "q8_0" {
		t.Fatalf("KV types = %q/%q; want q8_0/q8_0", selected.Options.TypeK, selected.Options.TypeV)
	}
	if !strings.Contains(selected.Adjustment, "KV f16/f16 -> q8_0/q8_0") {
		t.Fatalf("unexpected adjustment: %q", selected.Adjustment)
	}
}

func TestLoadPlannerNeverChangesExplicitContext(t *testing.T) {
	planner := fixedMemoryPlanner(24*testGiB, 10*testGiB)
	opts := engine.DefaultLoadOptions()
	opts.CtxSize = 120832
	opts.CtxSizeExplicit = true

	plan, err := planner.Plan(LoadPlanRequest{
		Options:        opts,
		ModelBytes:     11 * testGiB,
		ProjectorBytes: 1 * testGiB,
		Family:         "llama",
		Vision:         true,
	})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	for _, attempt := range plan.Attempts {
		if attempt.Options.CtxSize != opts.CtxSize {
			t.Fatalf("explicit context changed to %d", attempt.Options.CtxSize)
		}
	}
}

func TestLoadPlannerRejectsProfileWhenAllMemoryControlsAreExplicit(t *testing.T) {
	planner := fixedMemoryPlanner(24*testGiB, 10*testGiB)
	opts := engine.DefaultLoadOptions()
	opts.CtxSize = 120832
	opts.CtxSizeExplicit = true
	opts.BatchExplicit = true
	opts.TypeKExplicit = true
	opts.TypeVExplicit = true

	_, err := planner.Plan(LoadPlanRequest{
		Options:        opts,
		ModelBytes:     11 * testGiB,
		ProjectorBytes: 1 * testGiB,
		Family:         "llama",
		Vision:         true,
	})
	if err == nil || !strings.Contains(err.Error(), "load would exceed RAM safety budget") {
		t.Fatalf("expected memory budget error, got %v", err)
	}
}

func TestLoadPlannerDisableAutoFitRequiresExactProfile(t *testing.T) {
	planner := fixedMemoryPlanner(24*testGiB, 10*testGiB)
	opts := engine.DefaultLoadOptions()
	opts.CtxSize = 120832
	opts.DisableAutoFit = true

	_, err := planner.Plan(LoadPlanRequest{
		Options:        opts,
		ModelBytes:     11 * testGiB,
		ProjectorBytes: 1 * testGiB,
		Family:         "llama",
		Vision:         true,
	})
	if err == nil {
		t.Fatal("expected exact profile to exceed memory budget")
	}
	if strings.Contains(err.Error(), "Automatic fitting tried") {
		t.Fatalf("disabled auto-fit should not claim it tried fallbacks: %v", err)
	}
}

func TestLoadPlannerRetryAttemptsStrictlyDecreaseMemory(t *testing.T) {
	planner := fixedMemoryPlanner(0, 0)
	opts := engine.DefaultLoadOptions()
	opts.CtxSize = 65536
	opts.BatchSize = 2048

	plan, err := planner.Plan(LoadPlanRequest{
		Options:    opts,
		ModelBytes: 6 * testGiB,
		Family:     "qwen3.5",
	})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if len(plan.Attempts) < 2 {
		t.Fatalf("expected adaptive retry profiles, got %d", len(plan.Attempts))
	}
	for i := 1; i < len(plan.Attempts); i++ {
		if plan.Attempts[i].Estimate.TotalBytes >= plan.Attempts[i-1].Estimate.TotalBytes {
			t.Fatalf("attempt %d is not smaller: %d >= %d", i, plan.Attempts[i].Estimate.TotalBytes, plan.Attempts[i-1].Estimate.TotalBytes)
		}
	}
}

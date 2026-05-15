package engine

import "testing"

func TestTrimAtStopRemovesStopSequence(t *testing.T) {
	got, stopped := trimAtStop("hello<stop>leak", []string{"<stop>"})
	if !stopped {
		t.Fatal("expected stop")
	}
	if got != "hello" {
		t.Fatalf("expected trimmed text, got %q", got)
	}
}

func TestStopStreamFilterHoldsPotentialStopTail(t *testing.T) {
	f := newStopStreamFilter([]string{"<stop>"})

	if out, stopped := f.PushCheck("hello<st"); stopped || out != "hel" {
		t.Fatalf("first push out=%q stopped=%v", out, stopped)
	}
	if out, stopped := f.PushCheck("op>leak"); !stopped || out != "lo" {
		t.Fatalf("second push out=%q stopped=%v", out, stopped)
	}
	if out := f.Flush(); out != "" {
		t.Fatalf("expected empty flush after stop, got %q", out)
	}
}

func TestStopStreamFilterFlushesWhenNoStop(t *testing.T) {
	f := newStopStreamFilter([]string{"<stop>"})
	out, stopped := f.PushCheck("hello")
	if stopped {
		t.Fatal("did not expect stop")
	}
	if out != "" {
		t.Fatalf("expected held text due stop tail guard, got %q", out)
	}
	if out := f.Flush(); out != "hello" {
		t.Fatalf("expected flush to release held text, got %q", out)
	}
}

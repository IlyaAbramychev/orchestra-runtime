package service

import (
	"os"
	"testing"
)

func TestKVBytesPerTokenForModel(t *testing.T) {
	const gib = int64(1024 * 1024 * 1024)
	tests := []struct {
		name       string
		modelBytes int64
		family     string
		want       int
	}{
		{name: "9B fallback", modelBytes: 6 * gib, family: "llama", want: 64 * 1024},
		{name: "27B fallback", modelBytes: 12 * gib, family: "qwen3", want: 128 * 1024},
		{name: "Qwen3.5 9B hybrid", modelBytes: 6 * gib, family: "qwen3.5", want: 32 * 1024},
		{name: "Qwen3.5 27B hybrid", modelBytes: 12 * gib, family: "qwen35", want: 64 * 1024},
		{name: "Qwen3.5 normalized family", modelBytes: 12 * gib, family: "QWEN3_5", want: 64 * 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := kvBytesPerTokenForModel(tt.modelBytes, tt.family); got != tt.want {
				t.Fatalf("kvBytesPerTokenForModel(%d, %q) = %d; want %d", tt.modelBytes, tt.family, got, tt.want)
			}
		})
	}
}

func TestQwen35MemoryEstimateFits24GiBAt120832Context(t *testing.T) {
	const gib = int64(1024 * 1024 * 1024)
	const contextSize = int64(120832)
	modelBytes := int64(12073948576)
	kvBytes := contextSize * int64(kvBytesPerTokenForModel(modelBytes, "qwen3.5"))
	projectorBytes := int64(888 * 1024 * 1024)
	needed := modelBytes + projectorBytes + kvBytes
	budget := int64(24)*gib - 2*gib

	if needed > budget {
		t.Fatalf("Qwen3.5 27B estimate %.2f GiB exceeds %.2f GiB budget", float64(needed)/float64(gib), float64(budget)/float64(gib))
	}
	if got := kvBytes / (1024 * 1024); got != 7552 {
		t.Fatalf("KV estimate = %d MiB; want 7552 MiB", got)
	}
}

func TestMMProjSize(t *testing.T) {
	dir := t.TempDir()
	projector := dir + "/mmproj.gguf"
	if err := os.WriteFile(projector, make([]byte, 42), 0o600); err != nil {
		t.Fatal(err)
	}

	got, err := mmprojSize(projector)
	if err != nil || got != 42 {
		t.Fatalf("mmprojSize() = %d, %v; want 42, nil", got, err)
	}
	if _, err := mmprojSize(dir); err == nil {
		t.Fatal("expected directory to be rejected")
	}
	if _, err := mmprojSize(dir + "/missing.gguf"); err == nil {
		t.Fatal("expected missing file to be rejected")
	}
}

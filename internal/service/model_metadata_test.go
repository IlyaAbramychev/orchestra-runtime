package service

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/operium/orchestra-runtime/internal/storage"
)

type modelMetadataTestKV struct {
	key   string
	kind  uint32
	value any
}

func TestNormalizeModelMetadataUsesGGUFInsteadOfFilename(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mystery-model.gguf")
	writeServiceGGUFFixture(t, path, []modelMetadataTestKV{
		{key: "general.architecture", kind: 8, value: "bert"},
		{key: "general.parameter_count", kind: 10, value: uint64(110_000_000)},
		{key: "bert.context_length", kind: 4, value: uint32(512)},
		{key: "bert.embedding_length", kind: 4, value: uint32(768)},
		{key: "bert.pooling_type", kind: 4, value: uint32(1)},
	})
	entry := &storage.ModelEntry{
		Name:     "mystery-model",
		Filename: filepath.Base(path),
		FilePath: path,
	}

	normalizeModelMetadata(entry)

	if entry.Family != "bert" || entry.Parameters != "110M" {
		t.Fatalf("GGUF identity not applied: %+v", entry)
	}
	if entry.TrainingContext != 512 || entry.EmbeddingLength != 768 {
		t.Fatalf("GGUF dimensions not applied: %+v", entry)
	}
	if !entry.Capabilities.Embeddings || entry.Capabilities.Chat {
		t.Fatalf("pooling model capabilities = %+v; want embeddings-only", entry.Capabilities)
	}
	if entry.RecommendedSettings.ContextSize != 512 {
		t.Fatalf("recommended context = %d; want clamped training context 512", entry.RecommendedSettings.ContextSize)
	}
}

func TestNormalizeModelMetadataDetectsTemplateCapabilitiesAndVision(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plain.gguf")
	writeServiceGGUFFixture(t, path, []modelMetadataTestKV{
		{key: "general.architecture", kind: 8, value: "qwen35"},
		{key: "qwen35.context_length", kind: 4, value: uint32(131072)},
		{key: "tokenizer.chat_template", kind: 8, value: "{% if tools %}{{ tool_calls }}{% endif %}{% if enable_thinking %}<think>{% endif %}"},
	})
	projectorName := "mmproj-plain-f16.gguf"
	if err := os.WriteFile(filepath.Join(filepath.Dir(path), projectorName), []byte("projector"), 0o600); err != nil {
		t.Fatal(err)
	}
	entry := &storage.ModelEntry{
		Name:     "plain",
		Filename: filepath.Base(path),
		FilePath: path,
	}

	normalizeModelMetadata(entry)

	if !entry.Capabilities.Chat || !entry.Capabilities.Tools || !entry.Capabilities.Thinking || !entry.Capabilities.Vision {
		t.Fatalf("template capabilities = %+v", entry.Capabilities)
	}
	if entry.Template == "" || entry.TrainingContext != 131072 {
		t.Fatalf("template/context metadata not applied: %+v", entry)
	}
	if entry.MMProjFilename != projectorName {
		t.Fatalf("projector = %q; want %q", entry.MMProjFilename, projectorName)
	}
	if err := requireModelCapability(entry, "vision"); err != nil {
		t.Fatalf("vision capability rejected: %v", err)
	}
}

func writeServiceGGUFFixture(t *testing.T, path string, values []modelMetadataTestKV) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, err := file.Write([]byte("GGUF")); err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{uint32(3), uint64(0), uint64(len(values))} {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, item := range values {
		writeServiceGGUFString(t, file, item.key)
		if err := binary.Write(file, binary.LittleEndian, item.kind); err != nil {
			t.Fatal(err)
		}
		switch value := item.value.(type) {
		case string:
			writeServiceGGUFString(t, file, value)
		case uint32:
			if err := binary.Write(file, binary.LittleEndian, value); err != nil {
				t.Fatal(err)
			}
		case uint64:
			if err := binary.Write(file, binary.LittleEndian, value); err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatalf("unsupported fixture type %T", value)
		}
	}
}

func writeServiceGGUFString(t *testing.T, file *os.File, value string) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(value); err != nil {
		t.Fatal(err)
	}
}

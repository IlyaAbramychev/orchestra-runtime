package engine

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

type ggufTestKV struct {
	key   string
	kind  uint32
	value any
}

func TestReadGGUFMetadataWithoutTensorAllocation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "metadata.gguf")
	writeGGUFMetadataFixture(t, path, []ggufTestKV{
		{key: "general.architecture", kind: 8, value: "qwen35"},
		{key: "general.name", kind: 8, value: "Metadata Fixture"},
		{key: "general.parameter_count", kind: 10, value: uint64(9_000_000_000)},
		{key: "qwen35.context_length", kind: 4, value: uint32(131072)},
		{key: "qwen35.embedding_length", kind: 4, value: uint32(4096)},
		{key: "qwen35.pooling_type", kind: 4, value: uint32(1)},
		{key: "tokenizer.chat_template", kind: 8, value: "{% if tools %}{{ tool_calls }}{% endif %}{% if enable_thinking %}<think>{% endif %}"},
	})

	metadata, err := ReadGGUFMetadata(path)
	if err != nil {
		t.Fatalf("ReadGGUFMetadata: %v", err)
	}
	if metadata.Architecture != "qwen35" || metadata.Name != "Metadata Fixture" {
		t.Fatalf("identity metadata = %+v", metadata)
	}
	if metadata.ContextLength != 131072 || metadata.EmbeddingLength != 4096 {
		t.Fatalf("dimensions metadata = %+v", metadata)
	}
	if metadata.ParameterCount != 9_000_000_000 || !metadata.HasPoolingType || metadata.PoolingType != 1 {
		t.Fatalf("numeric metadata = %+v", metadata)
	}
	if metadata.ChatTemplate == "" {
		t.Fatal("chat template not read")
	}
}

func writeGGUFMetadataFixture(t *testing.T, path string, values []ggufTestKV) {
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
		writeGGUFTestString(t, file, item.key)
		if err := binary.Write(file, binary.LittleEndian, item.kind); err != nil {
			t.Fatal(err)
		}
		switch value := item.value.(type) {
		case string:
			writeGGUFTestString(t, file, value)
		case uint32:
			if err := binary.Write(file, binary.LittleEndian, value); err != nil {
				t.Fatal(err)
			}
		case uint64:
			if err := binary.Write(file, binary.LittleEndian, value); err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatalf("unsupported GGUF fixture value %T", value)
		}
	}
}

func writeGGUFTestString(t *testing.T, file *os.File, value string) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(value); err != nil {
		t.Fatal(err)
	}
}

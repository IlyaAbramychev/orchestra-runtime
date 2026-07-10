package service

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/storage"
)

type cachedModelMetadata struct {
	size       int64
	modifiedNS int64
	metadata   engine.GGUFMetadata
}

var modelMetadataCache sync.Map // path -> cachedModelMetadata

func readCachedModelMetadata(path string) (engine.GGUFMetadata, bool) {
	path = strings.TrimSpace(path)
	if path == "" {
		return engine.GGUFMetadata{}, false
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return engine.GGUFMetadata{}, false
	}
	if cached, ok := modelMetadataCache.Load(path); ok {
		value := cached.(cachedModelMetadata)
		if value.size == info.Size() && value.modifiedNS == info.ModTime().UnixNano() {
			return value.metadata, true
		}
	}
	file, err := os.Open(path)
	if err != nil {
		return engine.GGUFMetadata{}, false
	}
	var magic [4]byte
	_, readErr := io.ReadFull(file, magic[:])
	file.Close()
	if readErr != nil || string(magic[:]) != "GGUF" {
		return engine.GGUFMetadata{}, false
	}
	metadata, err := engine.ReadGGUFMetadata(path)
	if err != nil {
		return engine.GGUFMetadata{}, false
	}
	modelMetadataCache.Store(path, cachedModelMetadata{
		size:       info.Size(),
		modifiedNS: info.ModTime().UnixNano(),
		metadata:   metadata,
	})
	return metadata, true
}

func enrichModelEntryFromGGUF(entry *storage.ModelEntry) bool {
	if entry == nil {
		return false
	}
	metadata, ok := readCachedModelMetadata(entry.FilePath)
	if !ok {
		return false
	}
	if metadata.Architecture != "" {
		entry.Family = metadata.Architecture
	}
	if metadata.ParameterCount > 0 && entry.Parameters == "" {
		entry.Parameters = formatParameterCount(metadata.ParameterCount)
	}
	if metadata.ChatTemplate != "" && entry.Template == "" {
		entry.Template = metadata.ChatTemplate
	}
	entry.TrainingContext = metadata.ContextLength
	entry.EmbeddingLength = metadata.EmbeddingLength

	detected := capabilitiesFromGGUF(metadata, entry.MMProjFilename != "")
	baseCapabilities := entry.Capabilities
	if detected.Embeddings && metadata.ChatTemplate == "" {
		// GGUF pooling metadata is stronger evidence than an old filename-based
		// default that marked every unknown model as chat-capable.
		baseCapabilities.Chat = false
	}
	entry.Capabilities = mergeModelCapabilities(baseCapabilities, detected)
	return true
}

func capabilitiesFromGGUF(metadata engine.GGUFMetadata, hasProjector bool) storage.ModelCapabilities {
	template := strings.ToLower(metadata.ChatTemplate)
	embeddingOnly := metadata.HasPoolingType && metadata.EmbeddingLength > 0 && metadata.ChatTemplate == ""
	return storage.ModelCapabilities{
		Chat:       !embeddingOnly,
		Embeddings: embeddingOnly,
		Tools: strings.Contains(template, "tools") &&
			(strings.Contains(template, "tool_call") || strings.Contains(template, "function")),
		Thinking: strings.Contains(template, "enable_thinking") ||
			strings.Contains(template, "reasoning_content") ||
			strings.Contains(template, "<think>"),
		Vision: hasProjector,
	}
}

func mergeModelCapabilities(base, detected storage.ModelCapabilities) storage.ModelCapabilities {
	return storage.ModelCapabilities{
		Chat:       base.Chat || detected.Chat,
		Embeddings: base.Embeddings || detected.Embeddings,
		Rerank:     base.Rerank || detected.Rerank,
		Tools:      base.Tools || detected.Tools,
		Thinking:   base.Thinking || detected.Thinking,
		Vision:     base.Vision || detected.Vision,
	}
}

func formatParameterCount(count uint64) string {
	const billion = uint64(1_000_000_000)
	if count >= billion {
		value := float64(count) / float64(billion)
		if value == float64(uint64(value)) {
			return fmt.Sprintf("%.0fB", value)
		}
		return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.1f", value), "0"), ".") + "B"
	}
	const million = uint64(1_000_000)
	if count >= million {
		return fmt.Sprintf("%.0fM", float64(count)/float64(million))
	}
	return fmt.Sprintf("%d", count)
}

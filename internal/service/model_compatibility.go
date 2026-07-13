package service

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/operium/orchestra-runtime/internal/storage"
)

const (
	modelArtifactIncompatibleCode = "model_artifact_incompatible"
	legacyOllamaGPTOSSArch        = "gptoss"
	llamaCppGPTOSSArch            = "gpt-oss"
)

// IncompatibleModelArtifactError reports a GGUF file that is valid for its
// source runtime but cannot be interpreted safely by the bundled llama.cpp.
// It is intentionally not bypassable: loading an incompatible tensor layout
// can succeed and still produce corrupted output.
type IncompatibleModelArtifactError struct {
	ModelID      string
	Architecture string
	ModelType    string
}

func (e *IncompatibleModelArtifactError) Error() string {
	return fmt.Sprintf(
		"model %s uses Ollama's legacy %q GGUF layout, which is incompatible with llama.cpp; use a llama.cpp-compatible gpt-oss GGUF with architecture %q",
		e.ModelID,
		e.Architecture,
		llamaCppGPTOSSArch,
	)
}

func (e *IncompatibleModelArtifactError) Code() string {
	return modelArtifactIncompatibleCode
}

func (e *IncompatibleModelArtifactError) HTTPStatus() int {
	return http.StatusUnprocessableEntity
}

func (e *IncompatibleModelArtifactError) RuntimeErrorDetails() map[string]any {
	repository, filename := compatibleGPTOSSArtifact(e.ModelType)
	return map[string]any{
		"architecture":           e.Architecture,
		"compatibleArchitecture": llamaCppGPTOSSArch,
		"reason":                 "legacy_ollama_gptoss_layout",
		"remediation": map[string]any{
			"action":     "replace_model_artifact",
			"repository": repository,
			"filename":   filename,
		},
	}
}

func compatibleGPTOSSArtifact(modelType string) (repository, filename string) {
	if strings.Contains(strings.ToLower(modelType), "120") {
		return "ggml-org/gpt-oss-120b-GGUF", "gpt-oss-120b-mxfp4.gguf"
	}
	return "ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-mxfp4.gguf"
}

func validateModelArtifact(entry *storage.ModelEntry) error {
	if entry == nil {
		return nil
	}

	// Prefer the file header over registry metadata. Family can be supplied by
	// callers or left stale after an import, while general.architecture is the
	// compatibility boundary llama.cpp actually uses.
	architecture := strings.TrimSpace(entry.Family)
	if metadata, ok := readCachedModelMetadata(entry.FilePath); ok && strings.TrimSpace(metadata.Architecture) != "" {
		architecture = strings.TrimSpace(metadata.Architecture)
	}

	if strings.EqualFold(architecture, legacyOllamaGPTOSSArch) {
		return &IncompatibleModelArtifactError{
			ModelID:      entry.ID,
			Architecture: architecture,
			ModelType:    entry.Parameters,
		}
	}
	return nil
}

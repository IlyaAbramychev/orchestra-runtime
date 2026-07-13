package service

import (
	"errors"
	"testing"

	"github.com/operium/orchestra-runtime/internal/storage"
)

func TestValidateModelArtifactRejectsLegacyOllamaGPTOSS(t *testing.T) {
	err := validateModelArtifact(&storage.ModelEntry{
		ID:     "gpt-oss-20b",
		Family: legacyOllamaGPTOSSArch,
	})

	var incompatible *IncompatibleModelArtifactError
	if !errors.As(err, &incompatible) {
		t.Fatalf("validateModelArtifact() error = %v; want IncompatibleModelArtifactError", err)
	}
	if incompatible.Code() != modelArtifactIncompatibleCode {
		t.Fatalf("error code = %q; want %q", incompatible.Code(), modelArtifactIncompatibleCode)
	}
}

func TestValidateModelArtifactAllowsLlamaCppGPTOSS(t *testing.T) {
	if err := validateModelArtifact(&storage.ModelEntry{
		ID:     "gpt-oss-20b",
		Family: llamaCppGPTOSSArch,
	}); err != nil {
		t.Fatalf("validateModelArtifact() = %v; want nil", err)
	}
}

func TestCloneModelEntryExposesLegacyOllamaGPTOSSAsError(t *testing.T) {
	entry := cloneModelEntry(&storage.ModelEntry{
		ID:     "gpt-oss-20b",
		Family: legacyOllamaGPTOSSArch,
		Status: "ready",
	})

	if entry.Status != "error" {
		t.Fatalf("status = %q; want error", entry.Status)
	}
	if entry.ErrorMessage == "" {
		t.Fatal("expected compatibility error message")
	}
}

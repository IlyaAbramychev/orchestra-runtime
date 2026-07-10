package config

import "testing"

func TestLoadMarksEnvironmentContextAsExplicit(t *testing.T) {
	t.Setenv("ORCHESTRA_CONFIG_DIR", t.TempDir())
	t.Setenv("ORCHESTRA_MODELS_DIR", t.TempDir())
	t.Setenv("ORCHESTRA_CTX_SIZE", "8192")

	cfg := Load()
	if cfg.ContextSize != 8192 || !cfg.ContextSizeExplicit {
		t.Fatalf("context size=%d explicit=%v; want 8192, true", cfg.ContextSize, cfg.ContextSizeExplicit)
	}
}

func TestLoadCanDisableAutomaticMemoryFitting(t *testing.T) {
	t.Setenv("ORCHESTRA_CONFIG_DIR", t.TempDir())
	t.Setenv("ORCHESTRA_MODELS_DIR", t.TempDir())
	t.Setenv("ORCHESTRA_AUTO_FIT", "0")

	cfg := Load()
	if cfg.AutoFit {
		t.Fatal("ORCHESTRA_AUTO_FIT=0 should disable automatic fitting")
	}
}

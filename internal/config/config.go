package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

type Config struct {
	Port             string
	ModelsDir        string
	ConfigDir        string
	DefaultGPULayers int
	ContextSize      int
	Threads          int
	MaxQueueSize     int
	CORSOrigins      []string
	ShutdownTimeout  time.Duration
	APIKey           string
	LogLevel         string
}

func Load() *Config {
	homeDir, _ := os.UserHomeDir()
	defaultConfigDir := filepath.Join(homeDir, ".orchestra")
	defaultModelsDir := filepath.Join(defaultConfigDir, "models")

	configDir := getEnv("ORCHESTRA_CONFIG_DIR", defaultConfigDir)
	modelsDir := getEnv("ORCHESTRA_MODELS_DIR", defaultModelsDir)

	// Ensure directories exist
	os.MkdirAll(configDir, 0755)
	os.MkdirAll(modelsDir, 0755)

	return &Config{
		Port:             getEnv("ORCHESTRA_PORT", getEnv("PORT", "8100")),
		ModelsDir:        modelsDir,
		ConfigDir:        configDir,
		DefaultGPULayers: getEnvInt("ORCHESTRA_GPU_LAYERS", -1),
		ContextSize:      getEnvInt("ORCHESTRA_CTX_SIZE", 4096),
		Threads:          getEnvInt("ORCHESTRA_THREADS", runtime.NumCPU()),
		MaxQueueSize:     getEnvInt("ORCHESTRA_MAX_QUEUE", 64),
		CORSOrigins:      splitEnv("CORS_ORIGINS", ",", []string{"http://localhost:3000", "http://localhost:3002", "http://localhost:5173"}),
		ShutdownTimeout:  time.Duration(getEnvInt("SHUTDOWN_TIMEOUT_SECONDS", 30)) * time.Second,
		APIKey:           getEnv("ORCHESTRA_API_KEY", ""),
		LogLevel:         getEnv("LOG_LEVEL", "info"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func splitEnv(key, sep string, fallback []string) []string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	var result []string
	for _, s := range splitString(v, sep) {
		s = trimSpace(s)
		if s != "" {
			result = append(result, s)
		}
	}
	return result
}

func splitString(s, sep string) []string {
	var result []string
	for len(s) > 0 {
		idx := indexOf(s, sep)
		if idx < 0 {
			result = append(result, s)
			break
		}
		result = append(result, s[:idx])
		s = s[idx+len(sep):]
	}
	return result
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func trimSpace(s string) string {
	start := 0
	for start < len(s) && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r') {
		start++
	}
	end := len(s)
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	return s[start:end]
}

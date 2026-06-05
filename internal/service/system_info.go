package service

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
)

type loadedContextSizer interface {
	LoadedContextSize() int
}

type loadedOptionsReporter interface {
	LoadedOptions() engine.LoadOptions
}

type loadedAtReporter interface {
	LoadedAt() time.Time
}

type lastErrorReporter interface {
	LastError() string
}

// Version is the runtime's advertised build version, surfaced via /api/system.
// It is set at process start from the `main.version` variable in cmd/server,
// which the Makefile populates via `-ldflags="-X main.version=$(VERSION)"`.
// Falls back to "dev" if the caller forgot to set it — makes it obvious when
// a hand-built binary is running vs a release artefact.
var Version = "dev"

// BuildCommit and LlamaCppCommit identify the exact runtime and llama.cpp
// sources used for this binary. Release builds inject both via ldflags.
var BuildCommit = "unknown"
var LlamaCppCommit = "unknown"

// Cached hardware sample — avoid spawning `vm_stat` / `nvidia-smi` on every
// /api/system call (they're fork-heavy and get spammy under load).
type hwSample struct {
	totalRAM     int64
	availableRAM int64
	gpu          *model.GPUInfo
	sampledAt    time.Time
}

type SystemInfo struct {
	engine    engine.Backend
	scheduler *RuntimeScheduler

	hwMu     sync.Mutex
	hwCache  hwSample
	hwMaxAge time.Duration
}

func NewSystemInfo(eng engine.Backend) *SystemInfo {
	return &SystemInfo{
		engine:   eng,
		hwMaxAge: 5 * time.Second,
	}
}

func (s *SystemInfo) SetScheduler(scheduler *RuntimeScheduler) {
	s.scheduler = scheduler
}

// hardware returns a lightly-cached hardware snapshot.
func (s *SystemInfo) hardware() (int64, int64, *model.GPUInfo) {
	s.hwMu.Lock()
	defer s.hwMu.Unlock()
	if time.Since(s.hwCache.sampledAt) < s.hwMaxAge && s.hwCache.totalRAM > 0 {
		return s.hwCache.totalRAM, s.hwCache.availableRAM, s.hwCache.gpu
	}
	s.hwCache = hwSample{
		totalRAM:     getTotalRAM(),
		availableRAM: getAvailableRAM(),
		gpu:          detectGPU(),
		sampledAt:    time.Now(),
	}
	return s.hwCache.totalRAM, s.hwCache.availableRAM, s.hwCache.gpu
}

func (s *SystemInfo) GetInfo(queueDepth int) *model.SystemInfoResponse {
	totalRAM, availableRAM, gpu := s.hardware()
	engineState := s.engine.State()
	currentModelID := s.engine.LoadedModelID()
	if s.scheduler != nil {
		snapshot := s.scheduler.Snapshot()
		engineState = snapshot.State
		if snapshot.ActiveModelID != "" {
			currentModelID = snapshot.ActiveModelID
		}
	}
	info := &model.SystemInfoResponse{
		Service:            "orchestra-runtime",
		Version:            Version,
		BuildCommit:        BuildCommit,
		LlamaCppCommit:     LlamaCppCommit,
		Platform:           runtime.GOOS,
		OS:                 runtime.GOOS,
		Arch:               runtime.GOARCH,
		CPUCount:           runtime.NumCPU(),
		EngineState:        engineState,
		QueueDepth:         queueDepth,
		IdleTimeoutSeconds: int(s.engine.IdleTimeout().Seconds()),
		TotalRAM:           totalRAM,
		AvailableRAM:       availableRAM,
		GPU:                gpu,
	}

	if currentModelID != "" {
		info.CurrentModel = &currentModelID
	}
	if sized, ok := s.engine.(loadedContextSizer); ok {
		info.ContextSize = sized.LoadedContextSize()
	}

	return info
}

func (s *SystemInfo) GetStatus() *model.RuntimeStatusResponse {
	engineState := s.engine.State()
	currentModelID := s.engine.LoadedModelID()
	if s.scheduler != nil {
		snapshot := s.scheduler.Snapshot()
		engineState = snapshot.State
		if snapshot.ActiveModelID != "" {
			currentModelID = snapshot.ActiveModelID
		}
	}

	var modelID *string
	if currentModelID != "" {
		modelID = &currentModelID
	}

	var contextSize *int
	if sized, ok := s.engine.(loadedContextSizer); ok {
		if size := sized.LoadedContextSize(); size > 0 {
			contextSize = &size
		}
	}

	opts := engine.LoadOptions{}
	if reporter, ok := s.engine.(loadedOptionsReporter); ok {
		opts = reporter.LoadedOptions()
	}

	var loadedAt *string
	if reporter, ok := s.engine.(loadedAtReporter); ok {
		if at := reporter.LoadedAt(); !at.IsZero() {
			formatted := at.UTC().Format(time.RFC3339)
			loadedAt = &formatted
		}
	}

	var errorText *string
	if reporter, ok := s.engine.(lastErrorReporter); ok {
		if msg := reporter.LastError(); msg != "" {
			errorText = &msg
		}
	}

	return &model.RuntimeStatusResponse{
		State:           engineState,
		Model:           modelID,
		ContextSize:     contextSize,
		MaxOutputTokens: engine.DefaultCompletionParams().MaxTokens,
		GPULayers:       opts.GPULayers,
		Threads:         opts.Threads,
		LoadedAt:        loadedAt,
		Error:           errorText,
	}
}

// --- Platform-specific hardware detection ---

func getTotalRAM() int64 {
	switch runtime.GOOS {
	case "darwin":
		return sysctlInt64("hw.memsize")
	case "linux":
		return readMemInfo("MemTotal")
	}
	return 0
}

func getAvailableRAM() int64 {
	switch runtime.GOOS {
	case "darwin":
		// On macOS, approximate via vm_stat
		pageSize := sysctlInt64("hw.pagesize")
		if pageSize == 0 {
			pageSize = 4096
		}
		out, err := exec.Command("vm_stat").Output()
		if err != nil {
			return 0
		}
		free := parseVMStatValue(string(out), "Pages free")
		inactive := parseVMStatValue(string(out), "Pages inactive")
		return (free + inactive) * pageSize
	case "linux":
		return readMemInfo("MemAvailable")
	}
	return 0
}

func detectGPU() *model.GPUInfo {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return &model.GPUInfo{
				Name:    "Apple Silicon (Metal)",
				Backend: "metal",
			}
		}
	case "linux":
		// Try nvidia-smi
		out, err := exec.Command("nvidia-smi",
			"--query-gpu=name,memory.total,memory.free",
			"--format=csv,noheader,nounits").Output()
		if err == nil {
			return parseNvidiaSMI(string(out))
		}
	}
	return nil
}

// --- Helper functions ---

func sysctlInt64(key string) int64 {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	return v
}

func readMemInfo(key string) int64 {
	out, err := exec.Command("grep", key, "/proc/meminfo").Output()
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(out))
	if len(fields) < 2 {
		return 0
	}
	v, _ := strconv.ParseInt(fields[1], 10, 64)
	return v * 1024 // /proc/meminfo is in kB
}

func parseVMStatValue(vmstat, key string) int64 {
	for _, line := range strings.Split(vmstat, "\n") {
		if strings.Contains(line, key) {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				numStr := strings.TrimSuffix(parts[len(parts)-1], ".")
				v, _ := strconv.ParseInt(numStr, 10, 64)
				return v
			}
		}
	}
	return 0
}

func parseNvidiaSMI(output string) *model.GPUInfo {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) == 0 {
		return nil
	}
	fields := strings.Split(lines[0], ",")
	if len(fields) < 3 {
		return nil
	}
	totalMB, _ := strconv.ParseInt(strings.TrimSpace(fields[1]), 10, 64)
	freeMB, _ := strconv.ParseInt(strings.TrimSpace(fields[2]), 10, 64)
	return &model.GPUInfo{
		Name:      strings.TrimSpace(fields[0]),
		TotalVRAM: totalMB * 1024 * 1024,
		FreeVRAM:  freeMB * 1024 * 1024,
		Backend:   "cuda",
	}
}

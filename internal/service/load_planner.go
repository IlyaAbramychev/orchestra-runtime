package service

import (
	"fmt"
	"strings"

	"github.com/operium/orchestra-runtime/internal/engine"
)

const (
	defaultLoadContext    = 4096
	minimumTextContext    = 1024
	minimumVisionContext  = 2048
	defaultMemoryHeadroom = int64(2 * 1024 * 1024 * 1024)
	// MaxAdaptiveLoadAttempts bounds expensive native model-load retries.
	MaxAdaptiveLoadAttempts = 8
)

// LoadPlanner converts a requested model profile into a bounded sequence of
// progressively smaller profiles. It never changes fields marked explicit.
// The first attempt is safe against the pre-flight budget; following attempts
// are only used after the backend reports a real allocation/OOM failure.
type LoadPlanner struct {
	totalRAM      func() int64
	availableRAM  func() int64
	reservedBytes int64
}

type LoadPlanRequest struct {
	Options         engine.LoadOptions
	ModelBytes      int64
	ProjectorBytes  int64
	Family          string
	Vision          bool
	TrainingContext int
	AllowOvercommit bool
}

type MemoryEstimate struct {
	ModelBytes     int64
	ProjectorBytes int64
	KVBytes        int64
	BatchBytes     int64
	TotalBytes     int64
}

type LoadAttempt struct {
	Options    engine.LoadOptions
	Estimate   MemoryEstimate
	Adjustment string
}

type LoadPlan struct {
	Attempts        []LoadAttempt
	TotalMemory     int64
	AvailableMemory int64
	SafetyBudget    int64
}

func NewLoadPlanner() *LoadPlanner {
	return &LoadPlanner{
		totalRAM:      getTotalRAM,
		availableRAM:  getAvailableRAM,
		reservedBytes: defaultMemoryHeadroom,
	}
}

func (p *LoadPlanner) Plan(req LoadPlanRequest) (LoadPlan, error) {
	base := normalizePlannedOptions(req.Options)
	if !base.CtxSizeExplicit && req.TrainingContext > 0 && base.CtxSize > req.TrainingContext {
		base.CtxSize = req.TrainingContext
		if !base.BatchExplicit && base.BatchSize > base.CtxSize {
			base.BatchSize = base.CtxSize
		}
	}
	total := p.totalRAM()
	available := p.availableRAM()
	budget := total - p.reservedBytes
	if budget < 0 {
		budget = 0
	}
	plan := LoadPlan{
		TotalMemory:     total,
		AvailableMemory: available,
		SafetyBudget:    budget,
	}

	candidates := []engine.LoadOptions{base}
	if !base.DisableAutoFit {
		candidates = buildAutomaticLoadCandidates(base, req.Vision)
	}

	type estimatedCandidate struct {
		opts     engine.LoadOptions
		estimate MemoryEstimate
	}
	estimated := make([]estimatedCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		estimated = append(estimated, estimatedCandidate{
			opts:     candidate,
			estimate: estimateLoadMemory(req.ModelBytes, req.ProjectorBytes, req.Family, candidate),
		})
	}

	start := -1
	for i, candidate := range estimated {
		if req.AllowOvercommit || budget <= 0 || candidate.estimate.TotalBytes <= budget {
			start = i
			break
		}
	}
	if start < 0 {
		minimum := estimated[len(estimated)-1].estimate
		return LoadPlan{}, newMemoryBudgetError(req, minimum, available, total, p.reservedBytes)
	}

	lastBytes := int64(0)
	for _, candidate := range estimated[start:] {
		// Retries must be materially smaller than the previous profile. This
		// prevents repeated expensive loads that cannot improve an OOM outcome.
		if len(plan.Attempts) > 0 && candidate.estimate.TotalBytes >= lastBytes {
			continue
		}
		plan.Attempts = append(plan.Attempts, LoadAttempt{
			Options:    candidate.opts,
			Estimate:   candidate.estimate,
			Adjustment: describeLoadAdjustment(base, candidate.opts),
		})
		lastBytes = candidate.estimate.TotalBytes
		if len(plan.Attempts) >= MaxAdaptiveLoadAttempts {
			break
		}
	}
	if len(plan.Attempts) == 0 {
		return LoadPlan{}, fmt.Errorf("load planner produced no attempts")
	}
	return plan, nil
}

func normalizePlannedOptions(opts engine.LoadOptions) engine.LoadOptions {
	if opts.CtxSize <= 0 {
		opts.CtxSize = defaultLoadContext
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = min(2048, opts.CtxSize)
	}
	if opts.BatchSize > opts.CtxSize {
		opts.BatchSize = opts.CtxSize
	}
	return opts
}

func buildAutomaticLoadCandidates(base engine.LoadOptions, vision bool) []engine.LoadOptions {
	seen := make(map[string]struct{})
	result := make([]engine.LoadOptions, 0, 24)
	appendCandidate := func(opts engine.LoadOptions) {
		key := loadOptionsMemoryKey(opts)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		result = append(result, opts)
	}
	appendBatchVariants := func(opts engine.LoadOptions) {
		appendCandidate(opts)
		if opts.BatchExplicit {
			return
		}
		for _, batch := range []int{1024, 512, 256, 128} {
			if batch >= opts.BatchSize || batch > opts.CtxSize {
				continue
			}
			candidate := opts
			candidate.BatchSize = batch
			appendCandidate(candidate)
		}
	}

	appendBatchVariants(base)
	q8Base := withAutomaticKVType(base, "q8_0")
	appendBatchVariants(q8Base)

	contexts := automaticContextCandidates(base, vision)
	for _, ctx := range contexts[1:] {
		candidate := withAutomaticContext(base, ctx)
		appendBatchVariants(candidate)
		appendBatchVariants(withAutomaticKVType(candidate, "q8_0"))
	}
	return result
}

func automaticContextCandidates(base engine.LoadOptions, vision bool) []int {
	if base.CtxSizeExplicit {
		return []int{base.CtxSize}
	}
	floor := minimumTextContext
	if vision {
		floor = minimumVisionContext
	}
	if base.BatchExplicit && base.BatchSize > floor {
		floor = base.BatchSize
	}
	contexts := []int{base.CtxSize}
	for _, value := range []int{65536, 32768, 16384, 8192, 4096, 2048, 1024} {
		if value < base.CtxSize && value >= floor {
			contexts = append(contexts, value)
		}
	}
	return contexts
}

func withAutomaticContext(opts engine.LoadOptions, contextSize int) engine.LoadOptions {
	if opts.CtxSizeExplicit {
		return opts
	}
	opts.CtxSize = contextSize
	if !opts.BatchExplicit && opts.BatchSize > contextSize {
		opts.BatchSize = contextSize
	}
	return opts
}

func withAutomaticKVType(opts engine.LoadOptions, kind string) engine.LoadOptions {
	if !opts.TypeKExplicit {
		opts.TypeK = kind
	}
	if !opts.TypeVExplicit {
		opts.TypeV = kind
	}
	return opts
}

func loadOptionsMemoryKey(opts engine.LoadOptions) string {
	return fmt.Sprintf("%d/%d/%s/%s", opts.CtxSize, opts.BatchSize, strings.ToLower(opts.TypeK), strings.ToLower(opts.TypeV))
}

func estimateLoadMemory(modelBytes, projectorBytes int64, family string, opts engine.LoadOptions) MemoryEstimate {
	kvPerToken := kvBytesPerTokenForModel(modelBytes, family)
	kFactor := kvQuantFactor(opts.TypeK)
	vFactor := kvQuantFactor(opts.TypeV)
	kvBytes := int64(float64(int64(opts.CtxSize)*int64(kvPerToken)) * ((kFactor + vFactor) / 2.0))
	batchBytes := estimatedBatchSurcharge(opts.BatchSize)
	return MemoryEstimate{
		ModelBytes:     modelBytes,
		ProjectorBytes: projectorBytes,
		KVBytes:        kvBytes,
		BatchBytes:     batchBytes,
		TotalBytes:     modelBytes + projectorBytes + kvBytes + batchBytes,
	}
}

func estimatedBatchSurcharge(batch int) int64 {
	const (
		mib = int64(1024 * 1024)
		gib = int64(1024 * 1024 * 1024)
	)
	switch {
	case batch >= 2048:
		return 2 * gib
	case batch >= 1024:
		return 768 * mib
	default:
		return 0
	}
}

func describeLoadAdjustment(requested, selected engine.LoadOptions) string {
	var changes []string
	if requested.CtxSize != selected.CtxSize {
		changes = append(changes, fmt.Sprintf("n_ctx %d -> %d", requested.CtxSize, selected.CtxSize))
	}
	if requested.BatchSize != selected.BatchSize {
		changes = append(changes, fmt.Sprintf("n_batch %d -> %d", requested.BatchSize, selected.BatchSize))
	}
	if normalizeKVName(requested.TypeK) != normalizeKVName(selected.TypeK) || normalizeKVName(requested.TypeV) != normalizeKVName(selected.TypeV) {
		changes = append(changes, fmt.Sprintf("KV %s/%s -> %s/%s",
			normalizeKVName(requested.TypeK), normalizeKVName(requested.TypeV),
			normalizeKVName(selected.TypeK), normalizeKVName(selected.TypeV)))
	}
	if len(changes) == 0 {
		return "requested profile"
	}
	return "automatic memory fit: " + strings.Join(changes, ", ")
}

func normalizeKVName(value string) string {
	if strings.TrimSpace(value) == "" {
		return "f16"
	}
	return strings.ToLower(strings.TrimSpace(value))
}

func newMemoryBudgetError(req LoadPlanRequest, estimate MemoryEstimate, available, total, reserved int64) error {
	projectorEstimate := ""
	if req.ProjectorBytes > 0 {
		projectorEstimate = fmt.Sprintf(" + mmproj %.1f GB", bytesInGiB(req.ProjectorBytes))
	}
	autoHint := ""
	if !req.Options.DisableAutoFit {
		autoHint = " Automatic fitting tried smaller context, batch, and KV-cache profiles without finding a safe fit."
	}
	return fmt.Errorf(
		"load would exceed RAM safety budget: model %.1f GB%s + KV ~%.1f GB + batch ~%.1f GB = %.1f GB, "+
			"available %.1f GB, total %.1f GB (reserved %.0f GB for OS).%s "+
			"Close other apps, choose a smaller quantization, or set ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT=1 to bypass",
		bytesInGiB(req.ModelBytes), projectorEstimate, bytesInGiB(estimate.KVBytes),
		bytesInGiB(estimate.BatchBytes), bytesInGiB(estimate.TotalBytes),
		bytesInGiB(available), bytesInGiB(total), bytesInGiB(reserved), autoHint,
	)
}

func bytesInGiB(value int64) float64 {
	return float64(value) / 1024 / 1024 / 1024
}

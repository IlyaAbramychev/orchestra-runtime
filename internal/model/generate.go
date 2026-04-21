package model

// Ollama-compatible /api/generate shapes.
// Spec: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	// If true, concatenate the prompt to the model's own completion on next
	// call — not yet supported here (requires KV-cache persistence across
	// requests). Present in schema for future compatibility.
	Raw    bool   `json:"raw,omitempty"`
	Stream *bool  `json:"stream,omitempty"` // default true
	System string `json:"system,omitempty"`

	// Mirrors ChatCompletionRequest sampling.
	Options   *GenerateOptions `json:"options,omitempty"`
	KeepAlive *int64           `json:"keep_alive,omitempty"`
}

type GenerateOptions struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	MinP             *float64 `json:"min_p,omitempty"`
	TypicalP         *float64 `json:"typical_p,omitempty"`
	RepeatPenalty    *float64 `json:"repeat_penalty,omitempty"`
	RepeatLastN      *int     `json:"repeat_last_n,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	Seed             *int64   `json:"seed,omitempty"`
	Mirostat         *int     `json:"mirostat,omitempty"`
	MirostatTau      *float64 `json:"mirostat_tau,omitempty"`
	MirostatEta      *float64 `json:"mirostat_eta,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// GenerateResponse is emitted once per-chunk when streaming, or a single
// time for non-streaming calls.
type GenerateResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	// `done == true` marks the final chunk and carries the aggregate timings.
	Done      bool   `json:"done"`
	CreatedAt string `json:"created_at"`

	// Final-chunk fields (zero on intermediate chunks).
	TotalDurationNs      int64 `json:"total_duration,omitempty"`
	PromptEvalDurationNs int64 `json:"prompt_eval_duration,omitempty"`
	PromptEvalCount      int   `json:"prompt_eval_count,omitempty"`
	EvalDurationNs       int64 `json:"eval_duration,omitempty"`
	EvalCount            int   `json:"eval_count,omitempty"`
	DoneReason           string `json:"done_reason,omitempty"`
}

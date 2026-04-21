package model

// OpenAI-compatible request/response types.

type ChatCompletionRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`

	// Sampling — matches Ollama `options` + OpenAI chat API + LM Studio panel.
	Temperature      *float64 `json:"temperature,omitempty"`
	MaxTokens        *int     `json:"max_tokens,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"` // Ollama alias for max_tokens
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

	// keep_alive overrides the server-wide idle timeout for this session
	// *after* the request completes. Accepts a number of seconds (0 = unload
	// immediately, negative = keep forever). Matches Ollama's spelling.
	KeepAlive *int64 `json:"keep_alive,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
	Timings *Timings `json:"timings,omitempty"`
}

type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
	Timings *Timings      `json:"timings,omitempty"`
}

type Choice struct {
	Index        int          `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type ChunkChoice struct {
	Index        int          `json:"index"`
	Delta        *ChatMessage `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Timings mirrors Ollama's response fields so Ollama-native clients see
// tok/s without extra work. All durations are in nanoseconds.
type Timings struct {
	TotalDurationNs      int64 `json:"total_duration"`
	PromptEvalDurationNs int64 `json:"prompt_eval_duration"`
	PromptEvalCount      int   `json:"prompt_eval_count"`
	EvalDurationNs       int64 `json:"eval_duration"`
	EvalCount            int   `json:"eval_count"`
}

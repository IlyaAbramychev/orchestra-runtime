package model

import "encoding/json"

// Ollama-compatible /api/embed.
// Spec: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
type EmbedRequest struct {
	Model string `json:"model"`
	// Input can be a string OR []string. We accept both via json.RawMessage
	// and normalise in the handler.
	Input     json.RawMessage `json:"input"`
	Truncate  *bool           `json:"truncate,omitempty"`
	KeepAlive *int64          `json:"keep_alive,omitempty"`
	Options   json.RawMessage `json:"options,omitempty"`
}

type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
	// Token count for the *full batch* — sum of tokens across inputs.
	PromptEvalCount int   `json:"prompt_eval_count,omitempty"`
	TotalDurationNs int64 `json:"total_duration,omitempty"`
}

// OpenAI /v1/embeddings shapes.
// Spec: https://platform.openai.com/docs/api-reference/embeddings
type OpenAIEmbeddingsRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"` // "float" (default) or "base64"
	Dimensions     *int            `json:"dimensions,omitempty"`      // not supported; present for compat
}

type OpenAIEmbeddingsResponse struct {
	Object string                   `json:"object"` // "list"
	Data   []OpenAIEmbeddingRecord  `json:"data"`
	Model  string                   `json:"model"`
	Usage  OpenAIEmbeddingsUsage    `json:"usage"`
}

type OpenAIEmbeddingRecord struct {
	Object    string    `json:"object"` // "embedding"
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type OpenAIEmbeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

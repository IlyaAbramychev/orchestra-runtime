package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// OpenAI-compatible request/response types.

type ChatCompletionRequest struct {
	Model             string           `json:"model"`
	Messages          []ChatMessage    `json:"messages"`
	Stream            bool             `json:"stream,omitempty"`
	Tools             []ToolDefinition `json:"tools,omitempty"`
	ToolChoice        json.RawMessage  `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	ReasoningEffort   string           `json:"reasoning_effort,omitempty"`
	Think             json.RawMessage  `json:"think,omitempty"`

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

	// Grammar is an internal transport field for Ollama structured output.
	// It is not part of the OpenAI-compatible request JSON.
	Grammar string `json:"-"`
}

type ChatMessage struct {
	Role             string               `json:"role"`
	Content          string               `json:"content"`
	Parts            []MessageContentPart `json:"-"`
	Images           []string             `json:"images,omitempty"`
	Thinking         string               `json:"thinking,omitempty"`
	ReasoningContent string               `json:"reasoning_content,omitempty"`
	ToolName         string               `json:"tool_name,omitempty"`
	ToolCallID       string               `json:"tool_call_id,omitempty"`
	ToolCalls        []ToolCall           `json:"tool_calls,omitempty"`
}

type MessageContentPart struct {
	Type        string
	Text        string
	ImageURL    string
	ImageDetail string
}

// UnmarshalJSON accepts both the classic string-valued chat content used by
// Ollama and OpenAI's multimodal content-part array. Parts remain internal so
// response serialization keeps the existing string-valued content contract.
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		Images           []string        `json:"images,omitempty"`
		Thinking         string          `json:"thinking,omitempty"`
		ReasoningContent string          `json:"reasoning_content,omitempty"`
		ToolName         string          `json:"tool_name,omitempty"`
		ToolCallID       string          `json:"tool_call_id,omitempty"`
		ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*m = ChatMessage{
		Role:             raw.Role,
		Images:           raw.Images,
		Thinking:         raw.Thinking,
		ReasoningContent: raw.ReasoningContent,
		ToolName:         raw.ToolName,
		ToolCallID:       raw.ToolCallID,
		ToolCalls:        raw.ToolCalls,
	}
	trimmed := bytes.TrimSpace(raw.Content)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	if trimmed[0] == '"' {
		return json.Unmarshal(trimmed, &m.Content)
	}
	if trimmed[0] != '[' {
		return fmt.Errorf("message content must be a string or an array of content parts")
	}

	var parts []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text,omitempty"`
		ImageURL json.RawMessage `json:"image_url,omitempty"`
	}
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return fmt.Errorf("decode message content parts: %w", err)
	}
	var text strings.Builder
	for i, part := range parts {
		switch part.Type {
		case "text":
			m.Parts = append(m.Parts, MessageContentPart{Type: "text", Text: part.Text})
			text.WriteString(part.Text)
		case "image_url":
			url, detail, err := decodeOpenAIImageURL(part.ImageURL)
			if err != nil {
				return fmt.Errorf("content[%d].image_url: %w", i, err)
			}
			m.Parts = append(m.Parts, MessageContentPart{
				Type:        "image_url",
				ImageURL:    url,
				ImageDetail: detail,
			})
		default:
			return fmt.Errorf("content[%d].type %q is not supported", i, part.Type)
		}
	}
	m.Content = text.String()
	return nil
}

func decodeOpenAIImageURL(raw json.RawMessage) (url string, detail string, err error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", "", fmt.Errorf("url is required")
	}
	if trimmed[0] == '"' {
		if err := json.Unmarshal(trimmed, &url); err != nil {
			return "", "", err
		}
	} else {
		var value struct {
			URL    string `json:"url"`
			Detail string `json:"detail,omitempty"`
		}
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return "", "", err
		}
		url = value.URL
		detail = value.Detail
	}
	if strings.TrimSpace(url) == "" {
		return "", "", fmt.Errorf("url is required")
	}
	return url, detail, nil
}

// OllamaChatRequest is the /api/chat request shape. Ollama defaults to
// streaming unless stream is explicitly false and puts generation parameters
// under the nested options object.
type OllamaChatRequest struct {
	Model     string           `json:"model"`
	Messages  []ChatMessage    `json:"messages"`
	Stream    *bool            `json:"stream,omitempty"`
	Think     json.RawMessage  `json:"think,omitempty"`
	Format    json.RawMessage  `json:"format,omitempty"`
	Tools     []ToolDefinition `json:"tools,omitempty"`
	Options   *GenerateOptions `json:"options,omitempty"`
	KeepAlive *int64           `json:"keep_alive,omitempty"`
}

type ToolDefinition struct {
	Type     string       `json:"type,omitempty"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Index     int            `json:"index,omitempty"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

// UnmarshalJSON accepts both Ollama's object-valued arguments and OpenAI's
// JSON-string-valued arguments. Responses keep the native Ollama object shape;
// OpenAI response types below serialize arguments as strings explicitly.
func (f *ToolCallFunction) UnmarshalJSON(data []byte) error {
	var raw struct {
		Index     int             `json:"index,omitempty"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	f.Index = raw.Index
	f.Name = raw.Name
	if len(raw.Arguments) == 0 || string(raw.Arguments) == "null" {
		f.Arguments = map[string]any{}
		return nil
	}
	argumentJSON := raw.Arguments
	if len(argumentJSON) > 0 && argumentJSON[0] == '"' {
		var encoded string
		if err := json.Unmarshal(argumentJSON, &encoded); err != nil {
			return err
		}
		argumentJSON = []byte(encoded)
	}
	if len(bytes.TrimSpace(argumentJSON)) == 0 {
		f.Arguments = map[string]any{}
		return nil
	}
	if err := json.Unmarshal(argumentJSON, &f.Arguments); err != nil {
		return err
	}
	if f.Arguments == nil {
		f.Arguments = map[string]any{}
	}
	return nil
}

type OpenAIToolCall struct {
	Index    *int                   `json:"index,omitempty"`
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function OpenAIToolCallFunction `json:"function"`
}

type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIResponseMessage struct {
	Role             string           `json:"role,omitempty"`
	Content          *string          `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIResponseDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          *string          `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
	Timings *Timings       `json:"timings,omitempty"`
}

type OpenAIChoice struct {
	Index        int                    `json:"index"`
	Message      *OpenAIResponseMessage `json:"message,omitempty"`
	FinishReason *string                `json:"finish_reason"`
}

type OpenAIChatCompletionChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []OpenAIChunkChoice `json:"choices"`
	Usage   *Usage              `json:"usage,omitempty"`
	Timings *Timings            `json:"timings,omitempty"`
}

type OpenAIChunkChoice struct {
	Index        int                  `json:"index"`
	Delta        *OpenAIResponseDelta `json:"delta,omitempty"`
	FinishReason *string              `json:"finish_reason"`
}

type OpenAIToolChoice struct {
	Type     string `json:"type"`
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

type OllamaChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt string      `json:"created_at"`
	Message   ChatMessage `json:"message,omitempty"`
	Done      bool        `json:"done"`

	// Final-chunk fields.
	TotalDurationNs      int64  `json:"total_duration,omitempty"`
	PromptEvalDurationNs int64  `json:"prompt_eval_duration,omitempty"`
	PromptEvalCount      int    `json:"prompt_eval_count,omitempty"`
	EvalDurationNs       int64  `json:"eval_duration,omitempty"`
	EvalCount            int    `json:"eval_count,omitempty"`
	DoneReason           string `json:"done_reason,omitempty"`
	Error                string `json:"error,omitempty"`
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
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type PromptTokensDetails struct {
	TextTokens   int `json:"text_tokens"`
	VisionTokens int `json:"vision_tokens"`
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

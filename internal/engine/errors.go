package engine

import "fmt"

const ContextLengthExceededCode = "context_length_exceeded"

type ContextLengthExceededError struct {
	PromptTokens    int
	ContextSize     int
	MaxOutputTokens int
	ReloadHint      bool
}

func NewContextLengthExceededError(promptTokens, contextSize int, reloadHint bool, maxOutputTokens ...int) *ContextLengthExceededError {
	maxTokens := 0
	if len(maxOutputTokens) > 0 {
		maxTokens = maxOutputTokens[0]
	}
	return &ContextLengthExceededError{
		PromptTokens:    promptTokens,
		ContextSize:     contextSize,
		MaxOutputTokens: maxTokens,
		ReloadHint:      reloadHint,
	}
}

func (e *ContextLengthExceededError) Error() string {
	action := "load the model with a bigger n_ctx"
	if e.ReloadHint {
		action = "reload the model with a bigger n_ctx"
	}
	return fmt.Sprintf(
		"prompt too long: %d tokens \u2265 context window %d - reduce attachments/text or %s",
		e.PromptTokens,
		e.ContextSize,
		action,
	)
}

func (e *ContextLengthExceededError) Code() string {
	return ContextLengthExceededCode
}

func (e *ContextLengthExceededError) OverflowTokens() int {
	overflow := e.PromptTokens - e.ContextSize
	if e.MaxOutputTokens > 0 {
		overflow += e.MaxOutputTokens
	}
	if overflow < 0 {
		return 0
	}
	return overflow
}

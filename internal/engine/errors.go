package engine

import "fmt"

const ContextLengthExceededCode = "context_length_exceeded"

type ContextLengthExceededError struct {
	PromptTokens int
	ContextSize  int
	ReloadHint   bool
}

func NewContextLengthExceededError(promptTokens, contextSize int, reloadHint bool) *ContextLengthExceededError {
	return &ContextLengthExceededError{
		PromptTokens: promptTokens,
		ContextSize:  contextSize,
		ReloadHint:   reloadHint,
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

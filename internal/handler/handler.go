package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeRuntimeError(w http.ResponseWriter, err error) {
	writeJSON(w, runtimeHTTPStatus(err), map[string]any{
		"error": runtimeErrorPayload(err),
	})
}

func runtimeErrorPayload(err error) map[string]any {
	payload := map[string]any{
		"code":    runtimeErrorCode(err),
		"message": err.Error(),
	}
	if payload["code"] == "" {
		payload["code"] = "runtime_error"
	}
	var contextLength *engine.ContextLengthExceededError
	if errors.As(err, &contextLength) {
		payload["message"] = "Prompt exceeds model context window."
		payload["promptTokens"] = contextLength.PromptTokens
		payload["contextSize"] = contextLength.ContextSize
		payload["maxOutputTokens"] = contextLength.MaxOutputTokens
		payload["overflowTokens"] = contextLength.OverflowTokens()
	}
	return payload
}

func runtimeHTTPStatus(err error) int {
	if err == nil {
		return http.StatusInternalServerError
	}
	var badReq *badRequestErr
	if errors.As(err, &badReq) {
		return http.StatusBadRequest
	}
	var contextLength *engine.ContextLengthExceededError
	if errors.As(err, &contextLength) {
		return http.StatusBadRequest
	}
	if errors.Is(err, context.Canceled) {
		return 499
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return http.StatusGatewayTimeout
	}
	if errors.Is(err, rpc.ErrWorkerCrashed) {
		return http.StatusServiceUnavailable
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "queue full"):
		return http.StatusTooManyRequests
	case strings.Contains(msg, "no model loaded"),
		strings.Contains(msg, "model not found"):
		return http.StatusNotFound
	case strings.Contains(msg, "prompt too long"),
		strings.Contains(msg, "input too long"),
		strings.Contains(msg, "context window"),
		strings.Contains(msg, "context_overflow"),
		strings.Contains(msg, "does not support"),
		strings.Contains(msg, "custom chat template failed"):
		return http.StatusBadRequest
	case strings.Contains(msg, "engine not ready"),
		strings.Contains(msg, "worker not ready"):
		return http.StatusServiceUnavailable
	default:
		return http.StatusInternalServerError
	}
}

func runtimeErrorCode(err error) string {
	if err == nil {
		return ""
	}
	var contextLength *engine.ContextLengthExceededError
	if errors.As(err, &contextLength) {
		return contextLength.Code()
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "prompt too long"),
		strings.Contains(msg, "input too long"),
		strings.Contains(msg, "context window"),
		strings.Contains(msg, "context_overflow"):
		return engine.ContextLengthExceededCode
	default:
		return ""
	}
}

func readJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

func hasMeaningfulRawJSON(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return false
	}
	if bytes.Equal(trimmed, []byte("{}")) || bytes.Equal(trimmed, []byte("[]")) {
		return false
	}
	return true
}

func structuredFormatInstruction(raw json.RawMessage) (string, bool, error) {
	if !hasMeaningfulRawJSON(raw) {
		return "", false, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false, fmt.Errorf("format must be \"json\" or a JSON schema object")
	}
	switch v := value.(type) {
	case string:
		if v != "json" {
			return "", false, fmt.Errorf("format string must be \"json\"")
		}
		return "Respond with exactly one valid JSON object. Do not include markdown fences, commentary, or text outside the JSON object.", true, nil
	case map[string]any:
		schema, err := json.Marshal(v)
		if err != nil {
			return "", false, fmt.Errorf("format schema is invalid")
		}
		return "Respond with exactly one valid JSON value that conforms to this JSON Schema: " + string(schema) + ". Do not include markdown fences, commentary, or text outside the JSON.", true, nil
	default:
		return "", false, fmt.Errorf("format must be \"json\" or a JSON schema object")
	}
}

func validateStructuredOutput(raw json.RawMessage, text string) error {
	if !hasMeaningfulRawJSON(raw) {
		return nil
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return fmt.Errorf("model returned empty structured output")
	}

	var format any
	if err := json.Unmarshal(raw, &format); err != nil {
		return fmt.Errorf("format must be \"json\" or a JSON schema object")
	}
	var output any
	if err := json.Unmarshal([]byte(trimmed), &output); err != nil {
		return fmt.Errorf("model returned invalid JSON for requested format")
	}
	if formatString, ok := format.(string); ok && formatString == "json" {
		if _, ok := output.(map[string]any); !ok {
			return fmt.Errorf("model returned JSON, but format \"json\" requires an object")
		}
		return nil
	}
	if _, ok := format.(map[string]any); ok {
		if err := validateJSONSchema(raw, trimmed); err != nil {
			return err
		}
	}
	return nil
}

func validateJSONSchema(schemaRaw json.RawMessage, output string) error {
	schemaDoc, err := jsonschema.UnmarshalJSON(bytes.NewReader(schemaRaw))
	if err != nil {
		return fmt.Errorf("format schema is invalid")
	}
	outputDoc, err := jsonschema.UnmarshalJSON(strings.NewReader(output))
	if err != nil {
		return fmt.Errorf("model returned invalid JSON for requested format")
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("schema.json", schemaDoc); err != nil {
		return fmt.Errorf("format schema is invalid")
	}
	schema, err := compiler.Compile("schema.json")
	if err != nil {
		return fmt.Errorf("format schema is invalid")
	}
	if err := schema.Validate(outputDoc); err != nil {
		return fmt.Errorf("model returned JSON that does not conform to requested schema: %w", err)
	}
	return nil
}

func collectCompletionStream(ch <-chan engine.CompletionChunk) (string, engine.CompletionChunk, error) {
	var text strings.Builder
	var final engine.CompletionChunk
	for chunk := range ch {
		if chunk.Err != nil {
			return "", engine.CompletionChunk{}, chunk.Err
		}
		if chunk.Done {
			final = chunk
			return text.String(), final, nil
		}
		text.WriteString(chunk.Text)
	}
	return text.String(), final, nil
}

func validateThinkOption(raw json.RawMessage) error {
	if !hasMeaningfulRawJSON(raw) {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return fmt.Errorf("think must be a boolean or one of \"low\", \"medium\", \"high\"")
	}
	switch v := value.(type) {
	case bool:
		return nil
	case string:
		switch v {
		case "low", "medium", "high":
			return nil
		default:
			return fmt.Errorf("think must be a boolean or one of \"low\", \"medium\", \"high\"")
		}
	default:
		return fmt.Errorf("think must be a boolean or one of \"low\", \"medium\", \"high\"")
	}
}

func applyThinkingOutput(raw json.RawMessage, text string) (content string, thinking string) {
	thinking, content, ok := splitThinkingTags(text)
	if !ok {
		return text, ""
	}
	if thinkDisabled(raw) {
		return content, ""
	}
	return content, thinking
}

func splitThinkingTags(text string) (thinking string, content string, ok bool) {
	lower := strings.ToLower(text)
	start := strings.Index(lower, "<think>")
	if start < 0 {
		return "", text, false
	}
	bodyStart := start + len("<think>")
	relEnd := strings.Index(lower[bodyStart:], "</think>")
	if relEnd < 0 {
		return "", text, false
	}
	end := bodyStart + relEnd
	afterStart := end + len("</think>")

	thinking = strings.TrimSpace(text[bodyStart:end])
	content = strings.TrimSpace(text[:start] + text[afterStart:])
	return thinking, content, true
}

func thinkDisabled(raw json.RawMessage) bool {
	if !hasMeaningfulRawJSON(raw) {
		return false
	}
	var value bool
	if err := json.Unmarshal(raw, &value); err != nil {
		return false
	}
	return !value
}

func withStructuredInstruction(messages []model.ChatMessage, instruction string) []model.ChatMessage {
	out := append([]model.ChatMessage(nil), messages...)
	for i := range out {
		if out[i].Role == "system" {
			out[i].Content = appendInstruction(out[i].Content, instruction)
			return out
		}
	}
	return append([]model.ChatMessage{{Role: "system", Content: instruction}}, out...)
}

func appendInstruction(existing, instruction string) string {
	existing = strings.TrimSpace(existing)
	if existing == "" {
		return instruction
	}
	return existing + "\n\n" + instruction
}

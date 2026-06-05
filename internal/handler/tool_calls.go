package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/operium/orchestra-runtime/internal/model"
)

func toolCallInstruction(tools []model.ToolDefinition) (string, bool, error) {
	if len(tools) == 0 {
		return "", false, nil
	}
	for i, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			return "", false, fmt.Errorf("tools[%d].type must be \"function\"", i)
		}
		if strings.TrimSpace(tool.Function.Name) == "" {
			return "", false, fmt.Errorf("tools[%d].function.name is required", i)
		}
	}
	data, err := json.Marshal(tools)
	if err != nil {
		return "", false, fmt.Errorf("tools are invalid")
	}
	return "You may call these tools: " + string(data) + ". If a tool is needed, respond with exactly one JSON object in this shape: {\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"tool_name\",\"arguments\":{}}}]}. Do not execute tools yourself. If no tool is needed, answer normally.", true, nil
}

func parseToolCallsFromText(text string, tools []model.ToolDefinition) ([]model.ToolCall, bool, error) {
	if len(tools) == 0 {
		return nil, false, nil
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || !strings.HasPrefix(trimmed, "{") {
		return nil, false, nil
	}

	var envelope map[string]json.RawMessage
	if err := json.Unmarshal([]byte(trimmed), &envelope); err != nil {
		return nil, false, nil
	}
	rawCalls, ok := envelope["tool_calls"]
	if !ok {
		return nil, false, nil
	}

	var calls []rawToolCall
	if err := json.Unmarshal(rawCalls, &calls); err != nil {
		return nil, true, fmt.Errorf("model returned invalid tool_calls")
	}
	if len(calls) == 0 {
		return nil, true, fmt.Errorf("model returned empty tool_calls")
	}

	allowed := toolNameSet(tools)
	out := make([]model.ToolCall, 0, len(calls))
	for i, call := range calls {
		callType := strings.TrimSpace(call.Type)
		if callType == "" {
			callType = "function"
		}
		if callType != "function" {
			return nil, true, fmt.Errorf("model returned unsupported tool call type %q", call.Type)
		}

		name := call.Name
		rawArgs := call.Arguments
		if call.Function != nil {
			name = call.Function.Name
			rawArgs = call.Function.Arguments
		}
		name = strings.TrimSpace(name)
		if name == "" {
			return nil, true, fmt.Errorf("model returned tool call without function name")
		}
		if _, ok := allowed[name]; !ok {
			return nil, true, fmt.Errorf("model requested unknown tool %q", name)
		}

		args, err := decodeToolArguments(rawArgs)
		if err != nil {
			return nil, true, fmt.Errorf("model returned invalid arguments for tool %q: %w", name, err)
		}
		out = append(out, model.ToolCall{
			Type: "function",
			Function: model.ToolCallFunction{
				Index:     i,
				Name:      name,
				Arguments: args,
			},
		})
	}
	return out, true, nil
}

type rawToolCall struct {
	Type      string           `json:"type"`
	Function  *rawToolFunction `json:"function"`
	Name      string           `json:"name"`
	Arguments json.RawMessage  `json:"arguments"`
}

type rawToolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func toolNameSet(tools []model.ToolDefinition) map[string]struct{} {
	out := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		name := strings.TrimSpace(tool.Function.Name)
		if name != "" {
			out[name] = struct{}{}
		}
	}
	return out
}

func decodeToolArguments(raw json.RawMessage) (map[string]any, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return map[string]any{}, nil
	}

	var data []byte
	if bytes.HasPrefix(trimmed, []byte(`"`)) {
		var encoded string
		if err := json.Unmarshal(trimmed, &encoded); err != nil {
			return nil, err
		}
		encoded = strings.TrimSpace(encoded)
		if encoded == "" {
			return map[string]any{}, nil
		}
		data = []byte(encoded)
	} else {
		data = trimmed
	}

	var args map[string]any
	if err := json.Unmarshal(data, &args); err != nil {
		return nil, err
	}
	if args == nil {
		return map[string]any{}, nil
	}
	return args, nil
}

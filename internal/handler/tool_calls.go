package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/model"
)

type openAIToolMode struct {
	active        bool
	required      bool
	forcedName    string
	allowParallel bool
}

func engineToolCallsToModel(calls []engine.ToolCall) ([]model.ToolCall, error) {
	out := make([]model.ToolCall, 0, len(calls))
	for i, call := range calls {
		args, err := decodeToolArguments(call.Arguments)
		if err != nil {
			return nil, fmt.Errorf("model returned invalid arguments for tool %q: %w", call.Name, err)
		}
		out = append(out, model.ToolCall{
			ID:   call.ID,
			Type: "function",
			Function: model.ToolCallFunction{
				Index:     i,
				Name:      call.Name,
				Arguments: args,
			},
		})
	}
	return out, nil
}

func resolvedToolCalls(resultCalls []engine.ToolCall, text string, tools []model.ToolDefinition) ([]model.ToolCall, bool, error) {
	var calls []model.ToolCall
	var hasCalls bool
	var err error
	if len(resultCalls) > 0 {
		calls, err = engineToolCallsToModel(resultCalls)
		hasCalls = true
	} else {
		// Compatibility path for old workers during a rolling update. Protocol v2
		// rejects mixed binaries, so this can be removed after the migration window.
		calls, hasCalls, err = parseToolCallsFromText(text, tools)
	}
	if err != nil || !hasCalls {
		return calls, hasCalls, err
	}
	definitions := make(map[string]model.ToolDefinition, len(tools))
	for _, tool := range tools {
		definitions[tool.Function.Name] = tool
	}
	for _, call := range calls {
		definition, ok := definitions[call.Function.Name]
		if !ok {
			return nil, true, fmt.Errorf("model requested unknown tool %q", call.Function.Name)
		}
		if len(bytes.TrimSpace(definition.Function.Parameters)) == 0 {
			continue
		}
		arguments, marshalErr := json.Marshal(call.Function.Arguments)
		if marshalErr != nil {
			return nil, true, fmt.Errorf("encode arguments for tool %q: %w", call.Function.Name, marshalErr)
		}
		if schemaErr := validateJSONSchema(definition.Function.Parameters, string(arguments)); schemaErr != nil {
			return nil, true, fmt.Errorf("tool %q arguments do not match schema: %w", call.Function.Name, schemaErr)
		}
	}
	return calls, true, nil
}

func applyOpenAIToolInstructions(req *model.ChatCompletionRequest) (openAIToolMode, error) {
	if len(req.Tools) == 0 {
		if hasOpenAIToolChoice(req.ToolChoice) {
			return openAIToolMode{}, fmt.Errorf("tool_choice requires tools")
		}
		return openAIToolMode{}, nil
	}

	if _, _, err := toolCallInstruction(req.Tools); err != nil {
		return openAIToolMode{}, err
	}
	mode := openAIToolMode{active: true, allowParallel: true}
	choice := "auto"
	forcedName := ""
	if hasOpenAIToolChoice(req.ToolChoice) {
		if err := json.Unmarshal(req.ToolChoice, &choice); err != nil {
			var forced model.OpenAIToolChoice
			if objectErr := json.Unmarshal(req.ToolChoice, &forced); objectErr != nil {
				return openAIToolMode{}, fmt.Errorf("tool_choice must be \"auto\", \"none\", \"required\", or a named function")
			}
			if forced.Type != "function" || strings.TrimSpace(forced.Function.Name) == "" {
				return openAIToolMode{}, fmt.Errorf("tool_choice named function is invalid")
			}
			forcedName = strings.TrimSpace(forced.Function.Name)
			if _, ok := toolNameSet(req.Tools)[forcedName]; !ok {
				return openAIToolMode{}, fmt.Errorf("tool_choice references unknown tool %q", forcedName)
			}
			choice = "required"
		}
	}

	switch choice {
	case "none":
		return openAIToolMode{}, nil
	case "auto":
	case "required":
		mode.required = true
	default:
		return openAIToolMode{}, fmt.Errorf("tool_choice must be \"auto\", \"none\", or \"required\"")
	}
	if forcedName != "" {
		mode.forcedName = forcedName
		for _, tool := range req.Tools {
			if tool.Function.Name == forcedName {
				req.Tools = []model.ToolDefinition{tool}
				break
			}
		}
	}
	if req.ParallelToolCalls != nil && !*req.ParallelToolCalls {
		mode.allowParallel = false
	}
	return mode, nil
}

func hasOpenAIToolChoice(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

func validateOpenAIToolResult(mode openAIToolMode, calls []model.ToolCall, hasToolCalls bool) error {
	if mode.required && !hasToolCalls {
		return fmt.Errorf("model did not return a required tool call")
	}
	if !hasToolCalls {
		return nil
	}
	if !mode.allowParallel && len(calls) > 1 {
		return fmt.Errorf("model returned parallel tool calls when parallel_tool_calls is false")
	}
	if mode.forcedName != "" {
		for _, call := range calls {
			if call.Function.Name != mode.forcedName {
				return fmt.Errorf("model returned tool %q, but tool_choice requires %q", call.Function.Name, mode.forcedName)
			}
		}
	}
	return nil
}

func toOpenAIToolCalls(calls []model.ToolCall, streaming bool) []model.OpenAIToolCall {
	out := make([]model.OpenAIToolCall, 0, len(calls))
	for i, call := range calls {
		arguments, err := json.Marshal(call.Function.Arguments)
		if err != nil {
			arguments = []byte("{}")
		}
		id := call.ID
		if id == "" {
			id = "call_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:24]
		}
		converted := model.OpenAIToolCall{
			ID:   id,
			Type: "function",
			Function: model.OpenAIToolCallFunction{
				Name:      call.Function.Name,
				Arguments: string(arguments),
			},
		}
		if streaming {
			index := i
			converted.Index = &index
		}
		out = append(out, converted)
	}
	return out
}

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

#!/usr/bin/env bash
set -euo pipefail

# Runs the same agent/tool request against Ollama and Orchestra using the exact
# same GGUF bytes, context size, sampling options, and tool schema.
: "${ORCHESTRA_PARITY_GGUF:?set ORCHESTRA_PARITY_GGUF to an absolute GGUF path}"

RUNTIME_URL="${ORCHESTRA_PARITY_RUNTIME_URL:-http://127.0.0.1:8100}"
OLLAMA_URL="${ORCHESTRA_PARITY_OLLAMA_URL:-http://127.0.0.1:11434}"
N_CTX="${ORCHESTRA_PARITY_N_CTX:-4096}"
OLLAMA_MODEL="${ORCHESTRA_PARITY_OLLAMA_MODEL:-orchestra-parity-$$}"
CREATED_OLLAMA_MODEL=0
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/orchestra-parity.XXXXXX")"

cleanup() {
  if [[ "$CREATED_OLLAMA_MODEL" == "1" ]]; then
    ollama rm "$OLLAMA_MODEL" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

GGUF_PATH="$(cd "$(dirname "$ORCHESTRA_PARITY_GGUF")" && pwd)/$(basename "$ORCHESTRA_PARITY_GGUF")"
ln "$GGUF_PATH" "$WORK_DIR/parity.gguf" 2>/dev/null || cp "$GGUF_PATH" "$WORK_DIR/parity.gguf"

if [[ -z "${ORCHESTRA_PARITY_OLLAMA_MODEL:-}" ]]; then
  printf 'FROM %s\nPARAMETER num_ctx %s\n' "$GGUF_PATH" "$N_CTX" > "$WORK_DIR/Modelfile"
  ollama create "$OLLAMA_MODEL" -f "$WORK_DIR/Modelfile" >/dev/null
  CREATED_OLLAMA_MODEL=1
else
  OLLAMA_FROM="$(ollama show "$OLLAMA_MODEL" --modelfile | awk '/^FROM / {print $2; exit}')"
  if [[ "$OLLAMA_FROM" != "$GGUF_PATH" ]]; then
    printf 'Ollama model %s uses %s, expected exact GGUF %s\n' "$OLLAMA_MODEL" "$OLLAMA_FROM" "$GGUF_PATH" >&2
    exit 1
  fi
fi

IMPORT_RESPONSE="$(curl -fsS -X POST "$RUNTIME_URL/api/models/import" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg path "$WORK_DIR" '{path:$path}')")"
RUNTIME_MODEL="$(jq -r '.models[] | select(.filename == "parity.gguf") | .id' <<<"$IMPORT_RESPONSE" | head -n1)"
if [[ -z "$RUNTIME_MODEL" || "$RUNTIME_MODEL" == "null" ]]; then
  printf 'Could not resolve imported runtime model: %s\n' "$IMPORT_RESPONSE" >&2
  exit 1
fi

curl -fsS -X POST "$RUNTIME_URL/api/models/$RUNTIME_MODEL/load" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --argjson n "$N_CTX" '{context_size:$n}')" >/dev/null

TOOLS='[{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 file from the workspace","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}]'
MESSAGES='[{"role":"system","content":"Use tools when workspace data is required. Do not invent file contents."},{"role":"user","content":"Call read_file now with path exactly README.md."}]'
OPTIONS="$(jq -n --argjson n "$N_CTX" '{num_ctx:$n,temperature:0,seed:42,num_predict:256}')"

request() {
  local model="$1"
  jq -n \
    --arg model "$model" \
    --argjson messages "$MESSAGES" \
    --argjson tools "$TOOLS" \
    --argjson options "$OPTIONS" \
    '{model:$model,messages:$messages,tools:$tools,options:$options,stream:false,think:false}'
}

request "$OLLAMA_MODEL" | curl -fsS -X POST "$OLLAMA_URL/api/chat" -H 'Content-Type: application/json' -d @- > "$WORK_DIR/ollama.json"
request "$RUNTIME_MODEL" | curl -fsS -X POST "$RUNTIME_URL/api/chat" -H 'Content-Type: application/json' -d @- > "$WORK_DIR/runtime.json"

jq -n \
  --slurpfile ollama "$WORK_DIR/ollama.json" \
  --slurpfile runtime "$WORK_DIR/runtime.json" \
  --arg gguf "$GGUF_PATH" \
  --argjson n_ctx "$N_CTX" \
  '{
    invariant:{same_gguf:$gguf,same_context:$n_ctx,same_tools:true,same_sampling:true},
    ollama:{done_reason:$ollama[0].done_reason,tool_calls:$ollama[0].message.tool_calls,content:$ollama[0].message.content},
    runtime:{done_reason:$runtime[0].done_reason,tool_calls:$runtime[0].message.tool_calls,content:$runtime[0].message.content}
  }'

jq -e '.message.tool_calls | length > 0' "$WORK_DIR/ollama.json" >/dev/null
jq -e '.message.tool_calls | length > 0' "$WORK_DIR/runtime.json" >/dev/null
jq -e '.message.tool_calls[0].function.name == "read_file"' "$WORK_DIR/ollama.json" >/dev/null
jq -e '.message.tool_calls[0].function.name == "read_file"' "$WORK_DIR/runtime.json" >/dev/null

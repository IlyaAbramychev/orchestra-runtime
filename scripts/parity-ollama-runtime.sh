#!/usr/bin/env bash
set -euo pipefail

# Runs the same requests against Ollama and Orchestra using the exact same
# GGUF bytes, context size, sampling options, tool schema, and (optionally)
# vision projector/image. Both successful responses and capability errors are
# compared; raw responses and a machine-readable report are retained.

die() {
  printf 'parity: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

absolute_file() {
  local path="$1"
  [[ -f "$path" ]] || die "file not found: $path"
  (cd "$(dirname "$path")" && printf '%s/%s\n' "$PWD" "$(basename "$path")")
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

http_post() {
  local url="$1"
  local request_file="$2"
  local response_file="$3"
  curl -sS -o "$response_file" -w '%{http_code}' \
    -X POST "$url" \
    -H 'Content-Type: application/json' \
    --data-binary "@$request_file"
}

json_error() {
  jq -r '.error // .message // "request failed"' "$1" 2>/dev/null || printf 'invalid JSON response'
}

record_case() {
  local name="$1"
  local status="$2"
  local reason="$3"
  local ollama_status="$4"
  local runtime_status="$5"

  jq -n \
    --arg name "$name" \
    --arg status "$status" \
    --arg reason "$reason" \
    --argjson ollama_http "$ollama_status" \
    --argjson runtime_http "$runtime_status" \
    '{name:$name,status:$status,reason:$reason,ollama_http:$ollama_http,runtime_http:$runtime_http}' \
    > "$WORK_DIR/case-$name.json"

  if [[ "$status" == "fail" ]]; then
    FAILURES=$((FAILURES + 1))
  fi
}

record_skip() {
  local name="$1"
  local reason="$2"
  jq -n \
    --arg name "$name" \
    --arg reason "$reason" \
    '{name:$name,status:"skip",reason:$reason,ollama_http:0,runtime_http:0}' \
    > "$WORK_DIR/case-$name.json"
}

validate_success_case() {
  local name="$1"
  local ollama_file="$2"
  local runtime_file="$3"

  case "$name" in
    chat)
      jq -e '.message.content == "PARITY_OK"' "$ollama_file" >/dev/null &&
        jq -e '.message.content == "PARITY_OK"' "$runtime_file" >/dev/null
      ;;
    vision)
      jq -e '.message.content | type == "string" and length > 0' "$ollama_file" >/dev/null &&
        jq -e '.message.content | type == "string" and length > 0' "$runtime_file" >/dev/null
      ;;
    tools)
      jq -e '.message.tool_calls[0].function.name == "read_file" and .message.tool_calls[0].function.arguments.path == "README.md"' "$ollama_file" >/dev/null &&
        jq -e '.message.tool_calls[0].function.name == "read_file" and .message.tool_calls[0].function.arguments.path == "README.md"' "$runtime_file" >/dev/null &&
        [[ "$(jq -r '.done_reason // ""' "$ollama_file")" == "$(jq -r '.done_reason // ""' "$runtime_file")" ]]
      ;;
    thinking)
      jq -e '(.message.thinking // "") | type == "string"' "$ollama_file" >/dev/null &&
        jq -e '(.message.thinking // "") | type == "string"' "$runtime_file" >/dev/null &&
        jq -e '((.message.thinking // "") + (.message.content // "")) | length > 0' "$ollama_file" >/dev/null &&
        jq -e '((.message.thinking // "") + (.message.content // "")) | length > 0' "$runtime_file" >/dev/null &&
        jq -e '(.message.content // "") | contains("<think>") | not' "$ollama_file" >/dev/null &&
        jq -e '(.message.content // "") | contains("<think>") | not' "$runtime_file" >/dev/null &&
        [[ "$(jq -r '((.message.thinking // "") | length) > 0' "$ollama_file")" == "$(jq -r '((.message.thinking // "") | length) > 0' "$runtime_file")" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

run_ollama_case() {
  local name="$1"
  local request_file="$2"
  local response_file="$WORK_DIR/$name-ollama.json"
  http_post "$OLLAMA_URL/api/chat" "$request_file" "$response_file" > "$WORK_DIR/$name-ollama.http"
}

run_runtime_case() {
  local name="$1"
  local request_file="$2"
  local runtime_request="$WORK_DIR/$name-runtime-request.json"
  local ollama_response="$WORK_DIR/$name-ollama.json"
  local runtime_response="$WORK_DIR/$name-runtime.json"
  local ollama_status runtime_status

  jq --arg model "$RUNTIME_MODEL" '.model = $model' "$request_file" > "$runtime_request"
  ollama_status="$(< "$WORK_DIR/$name-ollama.http")"
  runtime_status="$(http_post "$RUNTIME_URL/api/chat" "$runtime_request" "$runtime_response")"

  if [[ "$ollama_status" =~ ^2 && "$runtime_status" =~ ^2 ]]; then
    if validate_success_case "$name" "$ollama_response" "$runtime_response"; then
      record_case "$name" pass "both implementations returned a valid outcome" "$ollama_status" "$runtime_status"
    else
      record_case "$name" fail "successful responses violate the $name invariant" "$ollama_status" "$runtime_status"
    fi
    return
  fi

  if [[ "$ollama_status" =~ ^4 && "$runtime_status" =~ ^4 ]]; then
    record_case "$name" pass "both implementations rejected the unsupported request" "$ollama_status" "$runtime_status"
    return
  fi

  record_case "$name" fail \
    "outcomes differ: Ollama $(json_error "$ollama_response"); runtime $(json_error "$runtime_response")" \
    "$ollama_status" "$runtime_status"
}

require_command curl
require_command jq
require_command ollama
if ! command -v shasum >/dev/null 2>&1 && ! command -v sha256sum >/dev/null 2>&1; then
  die "missing required command: shasum or sha256sum"
fi

: "${ORCHESTRA_PARITY_GGUF:?set ORCHESTRA_PARITY_GGUF to an absolute GGUF path}"

RUNTIME_URL="${ORCHESTRA_PARITY_RUNTIME_URL:-http://127.0.0.1:8100}"
OLLAMA_URL="${ORCHESTRA_PARITY_OLLAMA_URL:-http://127.0.0.1:11434}"
N_CTX="${ORCHESTRA_PARITY_N_CTX:-4096}"
OLLAMA_MODEL="${ORCHESTRA_PARITY_OLLAMA_MODEL:-orchestra-parity-$$}"
MMProj_INPUT="${ORCHESTRA_PARITY_MMPROJ:-}"
IMAGE_INPUT="${ORCHESTRA_PARITY_IMAGE:-}"
CREATED_OLLAMA_MODEL=0
RUNTIME_MODEL=""
FAILURES=0

if [[ -n "${ORCHESTRA_PARITY_OUTPUT_DIR:-}" ]]; then
  WORK_DIR="$ORCHESTRA_PARITY_OUTPUT_DIR"
  mkdir -p "$WORK_DIR"
  [[ -z "$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "output directory must be empty: $WORK_DIR"
else
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/orchestra-parity.XXXXXX")"
fi

cleanup() {
  if [[ -n "$RUNTIME_MODEL" ]]; then
    jq -n --arg model "$RUNTIME_MODEL" '{model:$model}' > "$WORK_DIR/runtime-delete-request.json"
    curl -sS -X DELETE "$RUNTIME_URL/api/delete" \
      -H 'Content-Type: application/json' \
      --data-binary "@$WORK_DIR/runtime-delete-request.json" >/dev/null 2>&1 || true
  fi
  if [[ "$CREATED_OLLAMA_MODEL" == "1" ]]; then
    ollama rm "$OLLAMA_MODEL" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

GGUF_PATH="$(absolute_file "$ORCHESTRA_PARITY_GGUF")"
GGUF_SHA256="$(sha256_file "$GGUF_PATH")"
MMPROJ_PATH=""
MMPROJ_SHA256=""
if [[ -n "$MMProj_INPUT" ]]; then
  MMPROJ_PATH="$(absolute_file "$MMProj_INPUT")"
  MMPROJ_SHA256="$(sha256_file "$MMPROJ_PATH")"
fi
if [[ -n "$IMAGE_INPUT" ]]; then
  IMAGE_PATH="$(absolute_file "$IMAGE_INPUT")"
else
  IMAGE_PATH=""
fi

curl -fsS "$OLLAMA_URL/api/version" > "$WORK_DIR/ollama-version.json" || die "Ollama is not reachable at $OLLAMA_URL"
curl -fsS "$RUNTIME_URL/api/version" > "$WORK_DIR/runtime-version.json" || die "Orchestra Runtime is not reachable at $RUNTIME_URL"

ln "$GGUF_PATH" "$WORK_DIR/parity.gguf" 2>/dev/null || cp "$GGUF_PATH" "$WORK_DIR/parity.gguf"
[[ "$(sha256_file "$WORK_DIR/parity.gguf")" == "$GGUF_SHA256" ]] || die "runtime GGUF copy does not match source SHA-256"
if [[ -n "$MMPROJ_PATH" ]]; then
  ln "$MMPROJ_PATH" "$WORK_DIR/mmproj-parity.gguf" 2>/dev/null || cp "$MMPROJ_PATH" "$WORK_DIR/mmproj-parity.gguf"
  [[ "$(sha256_file "$WORK_DIR/mmproj-parity.gguf")" == "$MMPROJ_SHA256" ]] || die "runtime mmproj copy does not match source SHA-256"
fi

if [[ -z "${ORCHESTRA_PARITY_OLLAMA_MODEL:-}" ]]; then
  {
    printf 'FROM %s\n' "$GGUF_PATH"
    if [[ -n "$MMPROJ_PATH" ]]; then
      printf 'ADAPTER %s\n' "$MMPROJ_PATH"
    fi
    printf 'PARAMETER num_ctx %s\n' "$N_CTX"
  } > "$WORK_DIR/Modelfile"
  ollama create "$OLLAMA_MODEL" -f "$WORK_DIR/Modelfile" >/dev/null 2>&1
  CREATED_OLLAMA_MODEL=1
fi

ollama show "$OLLAMA_MODEL" --modelfile > "$WORK_DIR/ollama-modelfile.txt"
OLLAMA_FROM="$(awk '/^FROM / {print $2; exit}' "$WORK_DIR/ollama-modelfile.txt")"
[[ -f "$OLLAMA_FROM" ]] || die "could not resolve Ollama model artifact from $OLLAMA_FROM"
OLLAMA_GGUF_SHA256="$(sha256_file "$OLLAMA_FROM")"
OLLAMA_GGUF_BYTE_IDENTICAL=false
if [[ "$OLLAMA_GGUF_SHA256" == "$GGUF_SHA256" ]]; then
  OLLAMA_GGUF_BYTE_IDENTICAL=true
elif [[ "$CREATED_OLLAMA_MODEL" != "1" ]]; then
  die "pre-existing Ollama model does not use the requested GGUF bytes"
fi
if [[ -n "$MMPROJ_PATH" ]]; then
  # Ollama versions differ here: some preserve ADAPTER, while 0.31 emits the
  # projector as a second FROM layer in the generated Modelfile.
  OLLAMA_ADAPTER="$(awk '/^ADAPTER / {print $2; exit} /^FROM / {from++; if (from == 2) {print $2; exit}}' "$WORK_DIR/ollama-modelfile.txt")"
  [[ -f "$OLLAMA_ADAPTER" ]] || die "could not resolve Ollama vision adapter from $OLLAMA_ADAPTER"
  OLLAMA_MMPROJ_SHA256="$(sha256_file "$OLLAMA_ADAPTER")"
  OLLAMA_MMPROJ_BYTE_IDENTICAL=false
  if [[ "$OLLAMA_MMPROJ_SHA256" == "$MMPROJ_SHA256" ]]; then
    OLLAMA_MMPROJ_BYTE_IDENTICAL=true
  elif [[ "$CREATED_OLLAMA_MODEL" != "1" ]]; then
    die "pre-existing Ollama model does not use the requested mmproj bytes"
  fi
else
  OLLAMA_MMPROJ_SHA256=""
  OLLAMA_MMPROJ_BYTE_IDENTICAL=false
fi

jq -n --arg path "$WORK_DIR" '{path:$path}' > "$WORK_DIR/runtime-import-request.json"
IMPORT_STATUS="$(http_post "$RUNTIME_URL/api/models/import" "$WORK_DIR/runtime-import-request.json" "$WORK_DIR/runtime-import.json")"
[[ "$IMPORT_STATUS" =~ ^2 ]] || die "runtime import failed ($(json_error "$WORK_DIR/runtime-import.json"))"
RUNTIME_MODEL="$(jq -r '.models[] | select(.filename == "parity.gguf") | .id' "$WORK_DIR/runtime-import.json" | head -n1)"
[[ -n "$RUNTIME_MODEL" && "$RUNTIME_MODEL" != "null" ]] || die "could not resolve imported runtime model"

jq -n --arg model "$RUNTIME_MODEL" '{model:$model}' > "$WORK_DIR/runtime-show-request.json"
SHOW_STATUS="$(http_post "$RUNTIME_URL/api/show" "$WORK_DIR/runtime-show-request.json" "$WORK_DIR/runtime-show.json")"
[[ "$SHOW_STATUS" =~ ^2 ]] || die "runtime show failed ($(json_error "$WORK_DIR/runtime-show.json"))"

TOOLS='[{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 file from the workspace","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}]'
OPTIONS="$(jq -n --argjson n "$N_CTX" '{num_ctx:$n,temperature:0,seed:42,num_predict:256}')"

jq -n \
  --arg model "$OLLAMA_MODEL" \
  --argjson options "$OPTIONS" \
  '{model:$model,messages:[{role:"user",content:"Reply with exactly PARITY_OK and nothing else."}],options:$options,stream:false,think:false}' \
  > "$WORK_DIR/chat-request.json"

jq -n \
  --arg model "$OLLAMA_MODEL" \
  --argjson options "$OPTIONS" \
  --argjson tools "$TOOLS" \
  '{model:$model,messages:[{role:"system",content:"Use tools when workspace data is required. Do not invent file contents."},{role:"user",content:"Call read_file now with path exactly README.md."}],tools:$tools,options:$options,stream:false,think:false}' \
  > "$WORK_DIR/tools-request.json"

jq -n \
  --arg model "$OLLAMA_MODEL" \
  --argjson options "$OPTIONS" \
  '{model:$model,messages:[{role:"user",content:"What is 17 + 25? Answer briefly."}],options:$options,stream:false,think:true}' \
  > "$WORK_DIR/thinking-request.json"

if [[ -n "$MMPROJ_PATH" && -n "$IMAGE_PATH" ]]; then
  IMAGE_BASE64="$(base64 < "$IMAGE_PATH" | tr -d '\r\n')"
  jq -n \
    --arg model "$OLLAMA_MODEL" \
    --arg image "$IMAGE_BASE64" \
    --argjson options "$OPTIONS" \
    '{model:$model,messages:[{role:"user",content:"Describe the image in one short sentence.",images:[$image]}],options:$options,stream:false,think:false}' \
    > "$WORK_DIR/vision-request.json"
fi

# Run providers in separate phases. Keeping both backends loaded at the same
# time makes VLM parity impossible on unified-memory machines and changes the
# very memory behaviour being compared.
run_ollama_case chat "$WORK_DIR/chat-request.json"
run_ollama_case tools "$WORK_DIR/tools-request.json"
run_ollama_case thinking "$WORK_DIR/thinking-request.json"
if [[ -f "$WORK_DIR/vision-request.json" ]]; then
  run_ollama_case vision "$WORK_DIR/vision-request.json"
fi

jq -n --arg model "$OLLAMA_MODEL" '{model:$model,keep_alive:0}' > "$WORK_DIR/ollama-unload-request.json"
http_post "$OLLAMA_URL/api/generate" "$WORK_DIR/ollama-unload-request.json" "$WORK_DIR/ollama-unload.json" > "$WORK_DIR/ollama-unload.http" || true

jq -n --argjson n "$N_CTX" '{context_size:$n,auto_fit:false}' > "$WORK_DIR/runtime-load-request.json"
LOAD_STATUS="$(http_post "$RUNTIME_URL/api/models/$RUNTIME_MODEL/load" "$WORK_DIR/runtime-load-request.json" "$WORK_DIR/runtime-load.json")"
[[ "$LOAD_STATUS" =~ ^2 ]] || die "runtime load failed ($(json_error "$WORK_DIR/runtime-load.json"))"

run_runtime_case chat "$WORK_DIR/chat-request.json"
run_runtime_case tools "$WORK_DIR/tools-request.json"
run_runtime_case thinking "$WORK_DIR/thinking-request.json"
if [[ -f "$WORK_DIR/vision-request.json" ]]; then
  run_runtime_case vision "$WORK_DIR/vision-request.json"
else
  record_skip vision "set both ORCHESTRA_PARITY_MMPROJ and ORCHESTRA_PARITY_IMAGE"
fi

jq -s '.' "$WORK_DIR"/case-*.json > "$WORK_DIR/cases.json"
jq -n \
  --arg gguf "$GGUF_PATH" \
  --arg gguf_sha256 "$GGUF_SHA256" \
  --arg ollama_gguf_sha256 "$OLLAMA_GGUF_SHA256" \
  --argjson ollama_gguf_byte_identical "$OLLAMA_GGUF_BYTE_IDENTICAL" \
  --arg mmproj "$MMPROJ_PATH" \
  --arg mmproj_sha256 "$MMPROJ_SHA256" \
  --arg ollama_mmproj_sha256 "$OLLAMA_MMPROJ_SHA256" \
  --argjson ollama_mmproj_byte_identical "$OLLAMA_MMPROJ_BYTE_IDENTICAL" \
  --arg ollama_model "$OLLAMA_MODEL" \
  --arg runtime_model "$RUNTIME_MODEL" \
  --argjson n_ctx "$N_CTX" \
  --slurpfile capabilities "$WORK_DIR/runtime-show.json" \
  --slurpfile cases "$WORK_DIR/cases.json" \
  '{
    invariant:{
      source_gguf:$gguf,
      source_gguf_sha256:$gguf_sha256,
      ollama_materialized_gguf_sha256:$ollama_gguf_sha256,
      ollama_gguf_byte_identical:$ollama_gguf_byte_identical,
      source_mmproj:$mmproj,
      source_mmproj_sha256:$mmproj_sha256,
      ollama_materialized_mmproj_sha256:$ollama_mmproj_sha256,
      ollama_mmproj_byte_identical:$ollama_mmproj_byte_identical,
      same_source_artifacts:true,
      context:$n_ctx,
      sampling:{temperature:0,seed:42,num_predict:256},
      same_tools:true
    },
    models:{ollama:$ollama_model,runtime:$runtime_model,runtime_capabilities:$capabilities[0].capabilities},
    cases:$cases[0],
    passed:([$cases[0][] | select(.status == "fail")] | length == 0)
  }' > "$WORK_DIR/report.json"

jq '.' "$WORK_DIR/report.json"
printf 'Parity artifacts: %s\n' "$WORK_DIR" >&2

if (( FAILURES > 0 )); then
  exit 1
fi

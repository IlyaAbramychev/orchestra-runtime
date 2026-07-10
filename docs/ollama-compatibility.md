# Ollama Compatibility Roadmap

Orchestra Runtime is developed as an Ollama-oriented local runtime for Operium.
The target is practical compatibility for local clients while keeping
Orchestra-specific model registry and direct GGUF workflows available.

## Current API Surface

Implemented:

- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`
- `GET /api/tags`
- `GET /api/ps`
- `POST /api/show`
- `POST /api/copy`
- `POST /api/create`
- `DELETE /api/delete`
- `POST /api/pull`
- `GET /api/version`

Extended Orchestra endpoints remain available under `/api/models`, `/api/system`,
`/api/status`, `/api/capabilities`, `/api/logs`, and `/api/shutdown`.

## Compatibility Notes

- `/api/pull` supports Ollama's `model`, `stream`, and `insecure` request
  fields.
- `/api/pull` can resolve Ollama registry names through the manifest/blob
  registry path.
- `/api/pull` also supports an Orchestra extension: `source_url`, for direct
  GGUF downloads outside an Ollama registry.
- Streaming pull responses use newline-delimited JSON and end with
  `{"status":"success"}`.
- Concurrent `/api/pull` calls for the same Ollama registry model share the
  active download and progress stream.
- Ollama registry pulls persist manifest-derived template, parameters, stop
  tokens, system prompt, license, and a Modelfile-style representation for
  `/api/show`.
- Ollama registry manifest/blob pull flow is covered by recorded fixture tests
  with digest verification and manifest metadata layers.
- `/api/create` supports metadata-only derived models from an existing `from`
  model. It updates registry metadata and shares the base GGUF artifact.
- `/api/chat` and `/api/generate` accept Ollama's `format` for structured
  output. The runtime steers the prompt, converts JSON Schema objects to GBNF
  with llama.cpp's converter, applies the grammar during decoding, and validates
  that the final response is valid JSON and conforms to the requested schema.
- `/api/chat` routes Ollama `tools` through llama.cpp's native Jinja
  chat-template, grammar, and parser APIs and returns `message.tool_calls`.
  Tool execution remains the client's responsibility; streaming tool calls are
  supported as buffered NDJSON responses, not token-level deltas. Availability
  is model-scoped because the embedded GGUF template must support tools.
- `/v1/chat/completions` accepts OpenAI-compatible `tools`, `tool_choice`, and
  `parallel_tool_calls=false`. It returns string-valued function arguments and
  preserves structured assistant tool-call/tool-result history without prompt
  injection. Function arguments are schema-validated; malformed calls finish
  with `tool_protocol_error`. Streaming tool calls use buffered SSE and are
  emitted after generation completes.
- `/api/chat` message `images` and `/api/generate` `images` support
  multimodal prompts when the runtime has either a global
  `ORCHESTRA_MMPROJ_PATH=/path/to/mmproj.gguf` or a model-scoped
  `mmproj_filename` in registry metadata; imported models also auto-detect a
  sibling `*mmproj*.gguf` in the same directory. The loaded model must still
  be compatible with that projector. Without a resolved projector, requests
  fail fast with a clear configuration error instead of being silently ignored.
  If multiple sibling `mmproj` files are present and none is configured
  explicitly, load is rejected with an ambiguity error instead of guessing.
- `/v1/chat/completions` accepts OpenAI multimodal content arrays with
  interleaved `text` and `image_url` parts. `image_url.url` currently requires
  a base64 `data:` URI using PNG, JPEG, or WebP; remote URLs are rejected.
  Requests are limited to 16 images, 20 MiB decoded per image, and 50 MiB
  decoded across the request.
- `/api/chat` validates Ollama's `think` option and passes thinking control to
  the native model chat template. Native parsing separates reasoning from
  visible content (`message.thinking` for Ollama and `reasoning_content` for
  OpenAI-compatible responses). `<think>` splitting remains a compatibility
  fallback for templates that do not expose structured reasoning.
- `/api/chat` and `/api/generate` support streaming structured output as a
  buffered NDJSON response: the runtime validates the final JSON/schema output
  before emitting stream chunks.
- `/api/capabilities` exposes machine-readable feature detection for supported,
  partial, extension, and unsupported compatibility behavior.
- `/api/embeddings` supports `encoding_format` values `float` and `base64`,
  `dimensions` truncation, and stable `error.code` values for common request
  and model capability failures.
- Core Ollama response shapes are covered by golden JSON tests for chat,
  generate, tags, show, ps, pull, delete, and version.
- A router-level Ollama compatibility smoke suite covers version, capabilities,
  tags, chat, generate streaming, structured output, tool calls, thinking,
  multimodal configuration gating, and embeddings.
- Real multimodal compatibility fixtures are available as env-gated integration
  tests. Set `ORCHESTRA_TEST_VISION_MODEL_PATH` and
  `ORCHESTRA_TEST_VISION_MMPROJ_PATH` to run a real `/api/chat` and
  `/api/generate` image request against a compatible vision pair, including
  mixed raw-base64 and `data:` URI image payloads plus multi-image requests;
  optionally set `ORCHESTRA_TEST_BAD_MMPROJ_PATH` to verify
  projector/model mismatch failures.
- `scripts/parity-ollama-runtime.sh` runs Ollama and Orchestra against one
  source GGUF with identical context, sampling, tools, and optional
  mmproj/image input. It records source and materialized SHA-256 values, raw
  responses, capability errors, and a machine-readable `report.json`. Provider
  phases are sequential so unified-memory pressure does not bias VLM results.
- Requests for tools or thinking are rejected before model load when GGUF
  metadata says the model does not support that capability, matching Ollama's
  model-scoped capability gate. Ollama `/api/chat` tool calls use
  `done_reason: "stop"`; OpenAI-compatible responses continue to use
  `finish_reason: "tool_calls"`.
- `/api/delete` resolves model id, display name, filename, or filename stem.
- Durations in chat and generate responses are returned in nanoseconds.

## Next Milestones

1. Additional lifecycle endpoints
   - `POST /api/push` only if Operium needs publishing.
   - Blob endpoints only if full Ollama manifest support requires them.

2. Advanced request parity
   - Token-level streaming tool-call/thinking deltas and tool-result lifecycle
     hardening.
   - Multimodal operational maturity: broader client coverage and release
     fixtures beyond the env-gated vision integration tests.
   - Broader JSON Schema compatibility fixtures against Ollama clients.

3. Operational maturity
   - Keep Orchestra extensions additive and avoid changing Ollama field names.
   - Stable error codes and HTTP mappings.
   - Expand release smoke tests against real Ollama SDK/client flows.

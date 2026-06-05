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
`/api/status`, `/api/logs`, and `/api/shutdown`.

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
- `/api/chat` and `/api/generate` accept Ollama's `format` for non-streaming
  structured output. The runtime steers the prompt and validates that the final
  response is valid JSON and conforms to the requested JSON Schema when a schema
  object is provided; grammar-constrained decoding is not implemented yet.
- `/api/chat` accepts Ollama `tools` in non-streaming mode and returns parsed
  `message.tool_calls`. Tool execution remains the client's responsibility;
  streaming tool-call deltas are not implemented yet.
- `/api/chat` message `images` and `/api/generate` `images` are recognized and
  rejected with a clear unsupported-feature error instead of being silently
  ignored.
- `/api/chat` and `/api/generate` validate Ollama's `think` option and separate
  `<think>...</think>` model output into `message.thinking` or `thinking` for
  non-streaming responses. Streaming thinking output is not implemented yet.
- Core Ollama response shapes are covered by golden JSON tests for chat,
  generate, tags, show, ps, pull, delete, and version.
- `/api/delete` resolves model id, display name, filename, or filename stem.
- Durations in chat and generate responses are returned in nanoseconds.

## Next Milestones

1. Additional lifecycle endpoints
   - `POST /api/push` only if Operium needs publishing.
   - Blob endpoints only if full Ollama manifest support requires them.

2. Advanced request parity
   - Streaming tool-call deltas and tool-result lifecycle hardening.
   - Multimodal inference with image embeddings and mmproj loading.
   - Streaming structured output and grammar-constrained decoding.
   - Streaming thinking accumulation and backend-level thinking controls.

3. Operational maturity
   - Keep Orchestra extensions additive and avoid changing Ollama field names.
   - Runtime capability endpoint for extension feature detection.
   - Stable error codes and HTTP mappings.
   - Release smoke tests against an Ollama-client compatibility suite.

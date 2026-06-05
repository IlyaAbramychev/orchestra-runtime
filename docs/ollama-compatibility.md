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
- `/api/delete` resolves model id, display name, filename, or filename stem.
- Durations in chat and generate responses are returned in nanoseconds.

## Next Milestones

1. Full Ollama pull registry support
   - Persist richer manifest metadata for `/api/show`.
   - Add integration coverage against the public Ollama registry with a small
     model fixture or recorded registry responses.

2. Response parity hardening
   - Add golden response tests for `/api/chat`, `/api/generate`, `/api/tags`,
     `/api/show`, `/api/ps`, `/api/pull`, `/api/delete`, and `/api/version`.
   - Keep Orchestra extensions additive and avoid changing Ollama field names.

3. Modelfile-compatible metadata
   - Represent template, parameters, stop tokens, system prompt, and license.
   - Return richer `/api/show` `modelfile`, `parameters`, and `model_info`.

4. Additional lifecycle endpoints
   - `POST /api/copy`
   - `POST /api/create`
   - `POST /api/push` only if Operium needs publishing.
   - Blob endpoints only if full Ollama manifest support requires them.

5. Advanced request parity
   - Chat tool calls.
   - Multimodal `images`.
   - Structured output via `format`.
   - Thinking model controls.

6. Operational maturity
   - Runtime capability endpoint for extension feature detection.
   - Stable error codes and HTTP mappings.
   - Release smoke tests against an Ollama-client compatibility suite.

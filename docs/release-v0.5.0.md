# Orchestra Runtime 0.5.0

This release focuses on local VLM correctness and reliable agent inference. The downloadable host and worker are built from the same revision; update both together.

## Changes since 0.4.0

- Fixed multimodal prefill against the pinned llama.cpp API: `mtmd_input_text.text_len` now receives the complete prompt byte length, preserving image markers. Added opt-in real-model buffered/streamed vision regressions.
- Added adaptive model load planning and GGUF-derived model capabilities.
- Preserved per-request context/GPU load profiles inside scheduler ownership, including `/api/generate`.
- Improved native incremental chat parsing and separation of visible text from reasoning, including final buffered deltas and partial `<think>` boundaries.
- Native template failures now return an explicit error instead of silently dropping tool/reasoning semantics. Malformed tool envelopes preserve `tool_protocol_error`.
- Added Ollama parity validation tooling and rejected the incompatible legacy Ollama GPT-OSS representation rather than pretending it is supported.

## Verification

- `go test ./...` and race tests for engine, handler and supervisor passed locally.
- Real Qwen3.5-9B-Q4_K_M + BF16 projector: buffered and streamed vision checks passed.
- Desktop integration: three runs on Runtime and three on Ollama, using the same GGUF/projector and an 8192 context. Both image → tool → answer and local description → text-only request → tool → answer passed in every run.

These are narrow smoke tests on one synthetic image, not a universal VLM quality or performance benchmark. The text-only request used the same model without pixels. See [the detailed evaluation](vlm-eval-2026-09-22.md) for hashes, setup and limitations.

## Distribution

Release assets target macOS Apple Silicon: `orchestra-runtime-darwin-arm64`, `orchestra-worker-darwin-arm64`, and `SHA256SUMS`. This is not a Desktop or VS Code extension binary release. Desktop image routing and analyzer changes require a separately distributed Desktop build.

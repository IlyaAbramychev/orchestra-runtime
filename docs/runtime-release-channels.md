# Runtime Release Channels

This repository uses channel manifests for controlled runtime rollout:

- `canary`
- `beta`
- `stable`

The VS Code extension resolves updates from:

`{runtimeManifestBaseUrl}/{channel}.json`

## Why channel manifests

GitHub `releases/latest` is not enough for production safety:

- no channel control,
- no extension compatibility gate,
- no deterministic promotion flow.

Channel manifests add:

- compatibility fields (`minExtensionVersion`, `maxExtensionVersion`),
- checksums (`sha256`),
- explicit promotion path.

## Manifest schema

```json
{
  "schemaVersion": 1,
  "channel": "stable",
  "releasedAt": "2026-04-29T12:00:00Z",
  "runtimeVersion": "1.4.2",
  "minExtensionVersion": "0.2.7",
  "maxExtensionVersion": "0.x",
  "critical": false,
  "notes": "Fix model unload deadlock on Apple Silicon",
  "artifacts": [
    {
      "platform": "darwin",
      "arch": "arm64",
      "url": "https://updates.operium.ru/orchestra-runtime/1.4.2/orchestra-runtime-darwin-arm64",
      "sha256": "hex",
      "size": 12345678,
      "workerUrl": "https://updates.operium.ru/orchestra-runtime/1.4.2/orchestra-worker-darwin-arm64",
      "workerSha256": "hex",
      "workerSize": 12345678
    }
  ]
}
```

## Artifact naming

Generator recognizes runtime binaries with exact names:

- `orchestra-runtime-darwin-arm64`
- `orchestra-runtime-darwin-x64`
- `orchestra-runtime-linux-arm64`
- `orchestra-runtime-linux-x64`
- `orchestra-runtime-win32-arm64.exe`
- `orchestra-runtime-win32-x64.exe`

Every Runtime artifact must have a matching Worker artifact for the same target:

- `orchestra-worker-darwin-arm64`
- `orchestra-worker-darwin-x64`
- `orchestra-worker-linux-arm64`
- `orchestra-worker-linux-x64`
- `orchestra-worker-win32-arm64.exe`
- `orchestra-worker-win32-x64.exe`

For GitHub Releases, use `https://github.com/<owner>/<repo>/releases/download` as
the base URL and pass the release tag (for example `v1.4.2`) to `--version`.

## Manual generation

```bash
node scripts/generate-runtime-manifest.mjs \
  --channel stable \
  --version 1.4.2 \
  --artifacts-dir ./release-assets \
  --base-url https://updates.operium.ru/orchestra-runtime \
  --min-extension-version 0.2.7 \
  --max-extension-version 0.x \
  --notes "Release note text" \
  --output ./release-manifests/stable.json
```

## GitHub Actions flow

Use workflow `Runtime Channel Manifest`:

1. pick channel (`canary|beta|stable`),
2. enter an existing release tag (for example `v1.4.2`),
3. set base URL for published binaries,
4. run workflow.

The workflow downloads release assets, generates manifest, and commits:

- `release-manifests/canary.json`
- `release-manifests/beta.json`
- `release-manifests/stable.json`

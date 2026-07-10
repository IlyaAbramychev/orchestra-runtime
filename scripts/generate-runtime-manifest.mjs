#!/usr/bin/env node

import { createHash } from 'crypto';
import { promises as fs } from 'fs';
import path from 'path';

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const [rawKey, inlineValue] = token.slice(2).split('=');
    const key = rawKey.trim();
    const next = argv[i + 1];
    const value = inlineValue ?? (next && !next.startsWith('--') ? next : 'true');
    if (inlineValue === undefined && next && !next.startsWith('--')) i++;
    out[key] = value;
  }
  return out;
}

function assertRequired(value, name) {
  if (!value || String(value).trim() === '') {
    throw new Error(`Missing required argument --${name}`);
  }
}

function parseArtifactName(fileName) {
  // orchestra-runtime-darwin-arm64
  // orchestra-runtime-win32-x64.exe
  const noExt = fileName.replace(/\.exe$/i, '');
  const m = noExt.match(/^orchestra-(runtime|worker)-(darwin|linux|win32)-(arm64|x64)$/i);
  if (!m) return null;
  return {
    kind: m[1].toLowerCase(),
    platform: m[2].toLowerCase(),
    arch: m[3].toLowerCase(),
  };
}

async function sha256File(filePath) {
  const hash = createHash('sha256');
  const content = await fs.readFile(filePath);
  hash.update(content);
  return hash.digest('hex');
}

async function main() {
  const args = parseArgs(process.argv);

  assertRequired(args.channel, 'channel');
  assertRequired(args.version, 'version');
  assertRequired(args['artifacts-dir'], 'artifacts-dir');
  assertRequired(args['base-url'], 'base-url');
  assertRequired(args.output, 'output');

  const channel = String(args.channel).trim().toLowerCase();
  if (!['stable', 'beta', 'canary'].includes(channel)) {
    throw new Error(`Unsupported channel: ${channel}`);
  }

  const artifactsDir = path.resolve(String(args['artifacts-dir']));
  const baseUrl = String(args['base-url']).replace(/\/+$/, '');
  const releaseTag = String(args.version).trim();
  const runtimeVersion = releaseTag.replace(/^v/i, '');
  const outputPath = path.resolve(String(args.output));

  const entries = await fs.readdir(artifactsDir, { withFileTypes: true });
  const files = entries.filter((entry) => entry.isFile()).map((entry) => entry.name);
  const artifactsByTarget = new Map();

  for (const fileName of files) {
    const parsed = parseArtifactName(fileName);
    if (!parsed) continue;
    const abs = path.join(artifactsDir, fileName);
    const stat = await fs.stat(abs);
    const target = `${parsed.platform}/${parsed.arch}`;
    const artifact = artifactsByTarget.get(target) ?? {
      platform: parsed.platform,
      arch: parsed.arch,
    };
    if (parsed.kind === 'runtime') {
      artifact.url = `${baseUrl}/${releaseTag}/${fileName}`;
      artifact.sha256 = await sha256File(abs);
      artifact.size = stat.size;
    } else {
      artifact.workerUrl = `${baseUrl}/${releaseTag}/${fileName}`;
      artifact.workerSha256 = await sha256File(abs);
      artifact.workerSize = stat.size;
    }
    artifactsByTarget.set(target, artifact);
  }

  const artifacts = [...artifactsByTarget.values()].filter((artifact) => artifact.url);

  if (artifacts.length === 0) {
    throw new Error(
      `No runtime artifacts found in ${artifactsDir}. Expected names like orchestra-runtime-darwin-arm64`
    );
  }
  const incompleteTargets = artifacts
    .filter((artifact) => !artifact.workerUrl || !artifact.workerSha256)
    .map((artifact) => `${artifact.platform}/${artifact.arch}`);
  if (incompleteTargets.length > 0) {
    throw new Error(
      `Missing worker artifact for: ${incompleteTargets.join(', ')}. ` +
      'Each Runtime artifact must be published with its matching orchestra-worker binary.',
    );
  }

  const manifest = {
    schemaVersion: 1,
    channel,
    releasedAt: new Date().toISOString(),
    runtimeVersion,
    minExtensionVersion: args['min-extension-version'] || '0.2.7',
    maxExtensionVersion: args['max-extension-version'] || '0.x',
    critical: String(args.critical || 'false').toLowerCase() === 'true',
    notes: String(args.notes || ''),
    artifacts,
  };

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(`Manifest written: ${outputPath}`);
  console.log(`Artifacts: ${artifacts.length}`);
}

main().catch((err) => {
  console.error(err.message);
  process.exitCode = 1;
});

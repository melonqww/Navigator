import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';
import { readFile, access } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { readSample } from './audio.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HELP = `Local voice lab (no paid API)
  npm run voice:doctor
  npm run voice:lab -- phrases
  npm run voice:lab -- inspect --sample data/samples/my-voice.wav
Local model commands: see tools/voice-lab/README.md`;

export async function main(args) {
  const { values, positionals } = parseArgs({ args, allowPositionals: true, strict: true,
    options: { sample: { type: 'string' } } });
  if (positionals.length > 1) throw new Error('EXPECTED_ONE_COMMAND');
  const command = positionals[0] ?? 'help';
  if (values.sample && command !== 'inspect') throw new Error('SAMPLE_ONLY_FOR_INSPECT');
  if (command === 'help') return HELP;
  if (command === 'phrases') return JSON.parse(await readFile(join(ROOT, 'fixtures/phrases/evaluation.ru.json'), 'utf8'));
  if (command === 'inspect') {
    if (!values.sample) throw new Error('SAMPLE_PATH_REQUIRED');
    const { metadata } = await readSample(resolve(values.sample));
    return { ...metadata, localReferenceReady: metadata.durationSeconds >= 3 && metadata.durationSeconds <= 30,
      qualityVerified: false, note: 'Statistics only. Speech clarity and similarity require listening.' };
  }
  if (command === 'doctor') {
    const python = process.env.VOICE_LAB_PYTHON ?? join(ROOT, 'data/voice-runtime',
      process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    let runtimeExists = true;
    try { await access(python); } catch { runtimeExists = false; }
    const gpu = spawnSync('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader'], { encoding: 'utf8', timeout: 10000, windowsHide: true });
    return { node: process.version, platform: process.platform, paidApiRequired: false,
      pythonRuntimeExists: runtimeExists, gpu: gpu.status === 0 ? gpu.stdout.trim() : 'unavailable',
      note: 'Runtime folder is not proof of installed dependencies. Run local.py doctor.' };
  }
  throw new Error('UNKNOWN_COMMAND');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then((result) => {
    console.log(typeof result === 'string' ? result : JSON.stringify(result, null, 2));
  }).catch((error) => {
    // Node parse/fs errors may echo arbitrary command arguments and paths.
    const message = /^[A-Z][A-Z0-9_]+$/.test(error.message) ? error.message : 'INVALID_ARGUMENT_OR_FILE';
    console.error(message);
    process.exitCode = 1;
  });
}

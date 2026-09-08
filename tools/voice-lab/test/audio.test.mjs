import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inspectWav, readSample, MAX_SAMPLE_BYTES } from '../audio.mjs';
import { main } from '../cli.mjs';

function wav({ seconds = 1, rate = 16000, channels = 1, amplitude = 8000 } = {}) {
  const length = seconds * rate * channels * 2;
  const buffer = Buffer.alloc(44 + length);
  buffer.write('RIFF', 0); buffer.writeUInt32LE(buffer.length - 8, 4); buffer.write('WAVE', 8);
  buffer.write('fmt ', 12); buffer.writeUInt32LE(16, 16); buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(channels, 22); buffer.writeUInt32LE(rate, 24);
  buffer.writeUInt32LE(rate * channels * 2, 28); buffer.writeUInt16LE(channels * 2, 32);
  buffer.writeUInt16LE(16, 34); buffer.write('data', 36); buffer.writeUInt32LE(length, 40);
  for (let offset = 44; offset < buffer.length; offset += 2) buffer.writeInt16LE(amplitude, offset);
  return buffer;
}

test('PCM mono and stereo duration uses frames, not scalar samples', () => {
  for (const channels of [1, 2]) {
    const result = inspectWav(wav({ channels, seconds: 2, rate: 44100 }));
    assert.equal(result.durationSeconds, 2);
    assert.equal(result.channels, channels);
    assert.deepEqual(result.warnings, []);
  }
});

test('silence is rejected', () => assert.throws(() => inspectWav(wav({ amplitude: 0 })), /SAMPLE_SILENT/));
test('quiet samples produce an actionable warning', () => assert.deepEqual(inspectWav(wav({ amplitude: 10 })).warnings, ['LOW_LEVEL']));
test('clipping is detected', () => assert.deepEqual(inspectWav(wav({ amplitude: 32767 })).warnings, ['CLIPPING']));
test('arbitrary text and renamed compressed files are rejected', () => assert.throws(() => inspectWav(Buffer.from('ID3 this is not a wave file')), /WAV_REQUIRED/));
test('truncation is rejected', () => assert.throws(() => inspectWav(wav().subarray(0, 100)), /WAV_SIZE_MISMATCH/));
test('oversized input is rejected before parsing', () => assert.throws(() => inspectWav(Buffer.alloc(MAX_SAMPLE_BYTES + 1)), /SAMPLE_TOO_LARGE/));
test('unsupported float WAV is rejected', () => {
  const buffer = wav(); buffer.writeUInt16LE(3, 20);
  assert.throws(() => inspectWav(buffer), /PCM16_REQUIRED/);
});
test('malicious chunk size cannot read beyond the buffer', () => {
  const buffer = wav(); buffer.writeUInt32LE(0xffffffff, 40);
  assert.throws(() => inspectWav(buffer), /WAV_TRUNCATED/);
});
test('invalid byte rate is rejected', () => {
  const buffer = wav(); buffer.writeUInt32LE(1, 28);
  assert.throws(() => inspectWav(buffer), /WAV_BAD_FORMAT/);
});
test('unrelated padded RIFF chunks are accepted', () => {
  const original = wav();
  const chunk = Buffer.alloc(10); chunk.write('JUNK'); chunk.writeUInt32LE(1, 4);
  const buffer = Buffer.concat([original.subarray(0, 12), chunk, original.subarray(12)]);
  buffer.writeUInt32LE(buffer.length - 8, 4);
  assert.equal(inspectWav(buffer).durationSeconds, 1);
});
test('files in directories with spaces are inspected without network access', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navigator audio '));
  try {
    const path = join(directory, 'reference voice.wav');
    await writeFile(path, wav({ seconds: 3 }));
    assert.equal((await readSample(path)).metadata.durationSeconds, 3);
    const report = await main(['inspect', '--sample', path]);
    assert.equal(report.localReferenceReady, true);
    assert.equal(report.qualityVerified, false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
test('missing local file reports failure', async () => {
  await assert.rejects(() => readSample(join(tmpdir(), 'navigator-no-such-reference-file.wav')), { code: 'ENOENT' });
});
test('short sample never claims to be ready for model evaluation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navigator short '));
  try {
    const path = join(directory, 'short.wav'); await writeFile(path, wav());
    assert.equal((await main(['inspect', '--sample', path])).localReferenceReady, false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
test('evaluation corpus has unique identifiers and covers numbers and streets', async () => {
  const phrases = await main(['phrases']);
  assert.equal(new Set(phrases.map((item) => item.id)).size, phrases.length);
  assert.ok(phrases.some((item) => item.category === 'numbers'));
  assert.ok(phrases.some((item) => item.category === 'street'));
  assert.ok(phrases.every((item) => item.text.trim().length > 0));
});
test('unknown options and missing inputs fail instead of starting work', async () => {
  await assert.rejects(() => main(['inspect']), /SAMPLE_PATH_REQUIRED/);
  await assert.rejects(() => main(['clone']), /UNKNOWN_COMMAND/);
  await assert.rejects(() => main(['phrases', '--sample', 'test']), /SAMPLE_ONLY_FOR_INSPECT/);
  await assert.rejects(() => main(['doctor', '--live']), /Unknown option/);
});

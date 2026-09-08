import { open } from 'node:fs/promises';

export const MAX_SAMPLE_BYTES = 25 * 1024 * 1024;

// The first experiment deliberately accepts only decoded PCM16 WAV.
// Container metadata and simple signal statistics do not measure voice similarity.
export function inspectWav(buffer) {
  if (buffer.length > MAX_SAMPLE_BYTES) throw new Error('SAMPLE_TOO_LARGE');
  if (buffer.length < 44 || buffer.toString('ascii', 0, 4) !== 'RIFF'
      || buffer.toString('ascii', 8, 12) !== 'WAVE') throw new Error('WAV_REQUIRED');
  const end = buffer.readUInt32LE(4) + 8;
  if (end !== buffer.length) throw new Error('WAV_SIZE_MISMATCH');
  let format;
  let pcm;
  for (let offset = 12; offset < end;) {
    if (offset + 8 > end) throw new Error('WAV_TRUNCATED');
    const kind = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    const next = start + size + (size % 2);
    if (next > end) throw new Error('WAV_TRUNCATED');
    if (kind === 'fmt ') {
      if (format || size < 16) throw new Error('WAV_BAD_FORMAT');
      format = {
        codec: buffer.readUInt16LE(start), channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4), byteRate: buffer.readUInt32LE(start + 8),
        blockAlign: buffer.readUInt16LE(start + 12), bits: buffer.readUInt16LE(start + 14),
      };
    }
    if (kind === 'data') {
      if (pcm) throw new Error('WAV_MULTIPLE_DATA_CHUNKS');
      pcm = buffer.subarray(start, start + size);
    }
    offset = next;
  }
  if (!format || !pcm?.length) throw new Error('WAV_NO_AUDIO');
  const { codec, channels, sampleRate, byteRate, blockAlign, bits } = format;
  if (codec !== 1 || bits !== 16 || ![1, 2].includes(channels)) throw new Error('PCM16_REQUIRED');
  if (sampleRate < 16000 || sampleRate > 96000 || blockAlign !== channels * 2
      || byteRate !== sampleRate * blockAlign || pcm.length % blockAlign !== 0) {
    throw new Error('WAV_BAD_FORMAT');
  }
  let squareSum = 0;
  let clipped = 0;
  let peak = 0;
  for (let offset = 0; offset < pcm.length; offset += 2) {
    const amplitude = Math.abs(pcm.readInt16LE(offset)) / 32768;
    squareSum += amplitude * amplitude;
    peak = Math.max(peak, amplitude);
    if (amplitude >= 0.999) clipped++;
  }
  if (peak === 0) throw new Error('SAMPLE_SILENT');
  const count = pcm.length / 2;
  const rmsDbfs = 20 * Math.log10(Math.sqrt(squareSum / count));
  const clippingRatio = clipped / count;
  const durationSeconds = pcm.length / byteRate;
  const warnings = [];
  if (rmsDbfs < -35) warnings.push('LOW_LEVEL');
  if (clippingRatio > 0.01) warnings.push('CLIPPING');
  return { bytes: buffer.length, durationSeconds, sampleRate, channels, rmsDbfs, clippingRatio, warnings };
}

export async function readSample(path) {
  const handle = await open(path, 'r');
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > MAX_SAMPLE_BYTES) throw new Error('SAMPLE_TOO_LARGE_OR_NOT_FILE');
    const bytes = Buffer.alloc(stat.size);
    let offset = 0;
    while (offset < bytes.length) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.length - offset, offset);
      if (!bytesRead) throw new Error('SAMPLE_CHANGED_DURING_READ');
      offset += bytesRead;
    }
    return { bytes, metadata: inspectWav(bytes) };
  } finally {
    await handle.close();
  }
}

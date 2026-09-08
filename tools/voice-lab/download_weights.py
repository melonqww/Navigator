"""Resumable fallback for networks that stall on multi-gigabyte HTTP responses."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request

from local import DATA, MODEL_DIR, MODEL_ID, REVISION

# LFS sizes and SHA-256 from the official repository at REVISION.
WEIGHTS = [
    ("model.safetensors", 1829344272, "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6"),
    ("speech_tokenizer/model.safetensors", 682293092, "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"),
]
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_chunk(url, start, end, total, target, opener=urllib.request.urlopen):
    length = end - start + 1
    if target.is_file() and target.stat().st_size == length:
        return
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with opener(request, timeout=30) as response:
                expected = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("UNEXPECTED_HTTP_RANGE")
                content = response.read(length + 1)
                if len(content) != length:
                    raise ValueError("INCOMPLETE_HTTP_RANGE")
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
            return
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Chunk {start}-{end} failed: {type(last_error).__name__}")


def download_weight(name, size, digest):
    destination = MODEL_DIR / name
    if destination.is_file() and destination.stat().st_size == size and sha256_file(destination) == digest:
        print(f"Already verified: {name}", flush=True)
        return
    parts = DATA / "cache" / "weight-parts" / digest
    parts.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}?download=true"
    ranges = [(start, min(start + CHUNK_SIZE, size) - 1) for start in range(0, size, CHUNK_SIZE)]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_chunk, url, start, end, size, parts / f"{start}.part") for start, end in ranges]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == len(ranges):
                print(f"{name}: {completed}/{len(ranges)} chunks; {time.monotonic() - started:.0f}s", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for start, end in ranges:
            with (parts / f"{start}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    if sha256_file(temporary) != digest:
        raise RuntimeError("WEIGHT_SHA256_MISMATCH")
    temporary.replace(destination)
    print(f"SHA-256 verified: {name}", flush=True)


if __name__ == "__main__":
    for item in WEIGHTS:
        download_weight(*item)

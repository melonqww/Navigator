"""Local Qwen experiment. Model download is a separate, explicit command."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
MODEL_DIR = DATA / "models" / "qwen3-tts-0.6b-base"


def configure_environment(offline=True):
    # Keep downloads and runtime caches out of global user directories.
    os.environ["HF_HOME"] = str(DATA / "cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    os.environ["NUMBA_CACHE_DIR"] = str(DATA / "cache" / "numba")
    os.environ["TORCH_HOME"] = str(DATA / "cache" / "torch")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def validate_reference(sample, transcript):
    sample = Path(sample)
    if not sample.is_file() or sample.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("SAMPLE_MISSING_OR_TOO_LARGE")
    with wave.open(str(sample), "rb") as source:
        channels, width, rate, frames = source.getparams()[:4]
        if width != 2 or channels not in (1, 2) or not 16000 <= rate <= 96000:
            raise ValueError("PCM16_WAV_REQUIRED")
        duration = frames / rate
        if not 3 <= duration <= 30:
            raise ValueError("REFERENCE_MUST_BE_3_TO_30_SECONDS")
        pcm = source.readframes(frames)
        if len(pcm) != frames * channels * width:
            raise ValueError("REFERENCE_TRUNCATED")
        if not any(value[0] for value in struct.iter_unpack("<h", pcm)):
            raise ValueError("REFERENCE_SILENT")
    text_file = Path(transcript)
    if not text_file.is_file() or text_file.stat().st_size > 10000:
        raise ValueError("TRANSCRIPT_MISSING_OR_TOO_LARGE")
    text = text_file.read_text(encoding="utf-8-sig").strip()
    if not text or len(text) > 2000:
        raise ValueError("TRANSCRIPT_EMPTY_OR_TOO_LONG")
    return {"sample": sample.resolve(), "transcript": text, "duration_seconds": duration,
            "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest()}


def select_phrase(phrase_id):
    entries = json.loads((ROOT / "fixtures/phrases/evaluation.ru.json").read_text(encoding="utf-8"))
    for entry in entries:
        if entry["id"] == phrase_id:
            return entry
    raise ValueError("UNKNOWN_PHRASE")


def load_model():
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise ValueError("MODEL_NOT_DOWNLOADED")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA_UNAVAILABLE")
    # On this Windows runtime, optional audio/vision dependencies can interfere
    # with CUDA DLL initialization. Establish the context before importing Qwen.
    torch.cuda.init()
    from qwen_tts import Qwen3TTSModel
    # SDPA avoids compiling FlashAttention on Windows; measure memory in practice.
    return Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR), device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )


def write_report(directory, report):
    temporary = directory / "report.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "report.json")


def synthesize(sample, transcript, phrase_id, consent, model_loader=load_model):
    if not consent:
        raise ValueError("VOICE_CONSENT_REQUIRED")
    reference = validate_reference(sample, transcript)
    phrase = select_phrase(phrase_id)
    if phrase["text"].casefold() in reference["transcript"].casefold():
        raise ValueError("EVALUATION_TEXT_MUST_BE_NEW")
    runs = DATA / "voice-lab"
    runs.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="local-", dir=runs))
    report = {"status": "running", "provider": "local-qwen", "model": MODEL_ID,
              "revision": REVISION, "phrase": phrase, "consent": True,
              "reference_seconds": reference["duration_seconds"],
              "reference_sha256": reference["sample_sha256"],
              "api_cost": 0, "quality_verified": False}
    write_report(directory, report)
    try:
        import numpy as np
        import soundfile as sf
        import torch
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model = model_loader()
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - start
        audio, rate = sf.read(str(reference["sample"]), dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1)
        if not np.isfinite(mono).all() or not np.any(mono):
            raise ValueError("REFERENCE_SILENT_AFTER_DOWNMIX")
        start = time.perf_counter()
        with torch.inference_mode():
            results, output_rate = model.generate_voice_clone(
                text=phrase["text"], language="Russian", ref_audio=(mono, rate),
                ref_text=reference["transcript"], max_new_tokens=512,
                do_sample=False, subtalker_dosample=False,
            )
        torch.cuda.synchronize()
        report["synthesis_seconds"] = time.perf_counter() - start
        result = np.asarray(results[0])
        if result.ndim != 1 or not result.size or not np.isfinite(result).all() or not np.any(result):
            raise ValueError("MODEL_INVALID_AUDIO")
        sf.write(str(directory / "result.wav"), result, output_rate, subtype="PCM_16")
        report.update(status="succeeded", audio_seconds=len(result) / output_rate,
                      sample_rate=output_rate,
                      peak_gpu_allocated_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 1))
        # Token cap can truncate speech: completion and pronunciation require listening.
        report["real_time_factor"] = report["synthesis_seconds"] / report["audio_seconds"]
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        write_report(directory, report)
        print(f"Failed run report: {directory / 'report.json'}", file=sys.stderr)
        raise
    write_report(directory, report)
    return {"report": str(directory / "report.json"), "audio": str(directory / "result.wav")}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Free local voice experiment; no voice upload or paid API.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("download")
    commands.add_parser("load-check")
    commands.add_parser("runtime-smoke")
    command = commands.add_parser("synthesize")
    command.add_argument("--sample", required=True)
    command.add_argument("--transcript", required=True)
    command.add_argument("--phrase", default="turn-right")
    command.add_argument("--consent", action="store_true")
    args = parser.parse_args(argv)
    configure_environment(offline=args.command != "download")
    if args.command == "download":
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR, token=False,
                          max_workers=2, allow_patterns=["*.json", "*.txt", "README.md"])
        from download_weights import WEIGHTS, download_weight
        for item in WEIGHTS:
            download_weight(*item)
        print(json.dumps({"model_downloaded": str(MODEL_DIR), "revision": REVISION}))
    elif args.command == "doctor":
        import importlib.metadata
        import torch
        print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                          "qwen_tts": importlib.metadata.version("qwen-tts"),
                          "cuda_available": torch.cuda.is_available(),
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "model_present": (MODEL_DIR / "model.safetensors").is_file(),
                          "api_key_required": False}, indent=2))
    elif args.command in ("load-check", "runtime-smoke"):
        start = time.perf_counter()
        model = load_model()
        import torch
        torch.cuda.synchronize()
        result = {"loaded": True, "load_seconds": time.perf_counter() - start,
                  "languages": model.get_supported_languages(),
                  "allocated_gpu_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                  "quality_verified": False}
        if args.command == "runtime-smoke":
            import numpy as np
            import soundfile as sf
            # Non-human sine wave exercises the full GPU pipeline without anyone's voice.
            # It cannot establish voice quality or pronunciation.
            rate = 24000
            tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(rate * 3) / rate)).astype(np.float32)
            start = time.perf_counter()
            with torch.inference_mode():
                audio, output_rate = model.generate_voice_clone(
                    text="Проверка.", language="Russian", ref_audio=(tone, rate),
                    # Deliberately artificial conditioning: exercise the same
                    # audio+transcript path as real cloning, without assessing speech.
                    ref_text="Технический тест.", x_vector_only_mode=False, max_new_tokens=32,
                    do_sample=False, subtalker_dosample=False,
                )
            torch.cuda.synchronize()
            output = np.asarray(audio[0])
            if output.ndim != 1 or not output.size or not np.isfinite(output).all() or not np.any(output):
                raise ValueError("SMOKE_INVALID_AUDIO")
            result.update(reference_kind="synthetic-tone-not-a-voice", inference_seconds=time.perf_counter() - start,
                          conditioning="audio-and-transcript",
                          audio_seconds=len(output) / output_rate,
                          peak_gpu_allocated_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 1))
            runs = DATA / "voice-lab"
            runs.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="smoke-", dir=runs))
            sf.write(str(directory / "technical-output.wav"), output, output_rate, subtype="PCM_16")
            write_report(directory, result)
            result["report"] = str(directory / "report.json")
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(synthesize(args.sample, args.transcript, args.phrase, args.consent), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        safe = message if message.isascii() and message.replace("_", "").isalnum() else type(error).__name__
        print(f"Local voice error: {safe}", file=sys.stderr)
        sys.exit(1)

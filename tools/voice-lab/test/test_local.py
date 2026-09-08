import importlib.util
import builtins
import sys
from types import SimpleNamespace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

SPEC = importlib.util.spec_from_file_location("voice_local", Path(__file__).parents[1] / "local.py")
local = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="navigator-voice-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sample = self.root / "voice sample.wav"
        self.transcript = self.root / "transcript.txt"
        self.transcript.write_text("Сегодня я рассказываю о своём обычном дне.", encoding="utf-8")
        self.make_wav()

    def make_wav(self, seconds=3, amplitude=5000):
        with wave.open(str(self.sample), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<h", amplitude) * (16000 * seconds))

    def test_valid_reference(self):
        result = local.validate_reference(self.sample, self.transcript)
        self.assertEqual(result["duration_seconds"], 3)
        self.assertEqual(len(result["sample_sha256"]), 64)

    def test_bom_transcript(self):
        self.transcript.write_text("Речь с меткой кодировки.", encoding="utf-8-sig")
        self.assertFalse(local.validate_reference(self.sample, self.transcript)["transcript"].startswith("\ufeff"))

    def test_no_consent_never_loads_model(self):
        loader = Mock()
        with self.assertRaisesRegex(ValueError, "VOICE_CONSENT_REQUIRED"):
            local.synthesize(self.sample, self.transcript, "turn-right", False, loader)
        loader.assert_not_called()

    def test_silent_sample_rejected_before_model_load(self):
        self.make_wav(amplitude=0)
        loader = Mock()
        with self.assertRaisesRegex(ValueError, "REFERENCE_SILENT"):
            local.synthesize(self.sample, self.transcript, "turn-right", True, loader)
        loader.assert_not_called()

    def test_short_and_long_samples_rejected(self):
        for seconds in [1, 31]:
            self.make_wav(seconds=seconds)
            with self.assertRaisesRegex(ValueError, "REFERENCE_MUST_BE_3_TO_30_SECONDS"):
                local.validate_reference(self.sample, self.transcript)

    def test_missing_transcript(self):
        with self.assertRaisesRegex(ValueError, "TRANSCRIPT_MISSING"):
            local.validate_reference(self.sample, self.root / "missing.txt")

    def test_empty_transcript(self):
        self.transcript.write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "TRANSCRIPT_EMPTY"):
            local.validate_reference(self.sample, self.transcript)

    def test_truncated_data(self):
        self.sample.write_bytes(self.sample.read_bytes()[:-100])
        with self.assertRaisesRegex(ValueError, "REFERENCE_TRUNCATED"):
            local.validate_reference(self.sample, self.transcript)

    def test_unknown_phrase(self):
        loader = Mock()
        with self.assertRaisesRegex(ValueError, "UNKNOWN_PHRASE"):
            local.synthesize(self.sample, self.transcript, "missing", True, loader)
        loader.assert_not_called()

    def test_evaluation_phrase_must_not_be_in_transcript(self):
        self.transcript.write_text(local.select_phrase("turn-right")["text"], encoding="utf-8")
        loader = Mock()
        with self.assertRaisesRegex(ValueError, "EVALUATION_TEXT_MUST_BE_NEW"):
            local.synthesize(self.sample, self.transcript, "turn-right", True, loader)
        loader.assert_not_called()

    def test_no_automatic_model_download(self):
        with patch.object(local, "MODEL_DIR", self.root / "absent"):
            with self.assertRaisesRegex(ValueError, "MODEL_NOT_DOWNLOADED"):
                local.load_model()

    def test_offline_by_default(self):
        with patch.dict(local.os.environ, clear=False):
            local.configure_environment()
            self.assertEqual(local.os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(local.os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(local.os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")

    def test_cuda_context_is_created_before_qwen_import(self):
        model_dir = self.root / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"test fixture")
        cuda = SimpleNamespace(is_available=Mock(return_value=True), init=Mock())
        torch = SimpleNamespace(cuda=cuda, bfloat16="test-dtype")
        model_class = SimpleNamespace(from_pretrained=Mock(return_value="loaded"))
        original_import = builtins.__import__

        def check_import(name, *args, **kwargs):
            if name == "qwen_tts":
                cuda.init.assert_called_once()
            return original_import(name, *args, **kwargs)

        with patch.object(local, "MODEL_DIR", model_dir), patch.dict(sys.modules, {
            "torch": torch, "qwen_tts": SimpleNamespace(Qwen3TTSModel=model_class)
        }), patch.object(builtins, "__import__", side_effect=check_import):
            self.assertEqual(local.load_model(), "loaded")
        self.assertTrue(model_class.from_pretrained.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()

"""Exercise WAV output and run reports with real torch/numpy, but a fake generator."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import local


@unittest.skipUnless(torch.cuda.is_available(), "Requires the local CUDA runtime")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        test_root = local.ROOT / "data" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="navigator-inference-", dir=test_root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sample = self.root / "sample.wav"
        self.transcript = self.root / "text.txt"
        self.transcript.write_text("Сегодня я рассказываю о своём обычном дне.", encoding="utf-8")
        self.signal = np.sin(2 * np.pi * 220 * np.arange(48000) / 16000).astype(np.float32) * 0.1
        sf.write(self.sample, self.signal, 16000, subtype="PCM_16")
        self.model = Mock()
        self.model.generate_voice_clone.return_value = ([self.signal], 16000)
        self.patch = patch.object(local, "DATA", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        local.configure_environment()

    def run_synthesis(self):
        return local.synthesize(self.sample, self.transcript, "turn-right", True, lambda: self.model)

    def report(self):
        path = next((self.root / "voice-lab").glob("*/report.json"))
        return json.loads(path.read_text(encoding="utf-8"))

    def test_writes_playable_wav_and_honest_report(self):
        result = self.run_synthesis()
        info = sf.info(result["audio"])
        self.assertEqual(info.subtype, "PCM_16")
        self.assertEqual(info.duration, 3)
        report = self.report()
        self.assertEqual(report["status"], "succeeded")
        self.assertFalse(report["quality_verified"])
        self.assertEqual(report["api_cost"], 0)
        args = self.model.generate_voice_clone.call_args.kwargs
        self.assertEqual(args["language"], "Russian")
        self.assertEqual(args["text"], local.select_phrase("turn-right")["text"])
        self.assertIsInstance(args["ref_audio"], tuple)

    def test_failure_is_persisted_without_raw_exception(self):
        self.model.generate_voice_clone.side_effect = RuntimeError("private details")
        with self.assertRaises(RuntimeError):
            self.run_synthesis()
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertNotIn("private details", json.dumps(report))

    def test_invalid_model_audio_is_not_written(self):
        for output in [np.array([]), np.array([np.nan]), np.array([0.0])]:
            self.model.generate_voice_clone.return_value = ([output], 16000)
            with self.assertRaisesRegex(ValueError, "MODEL_INVALID_AUDIO"):
                self.run_synthesis()
        self.assertEqual(list((self.root / "voice-lab").glob("*/result.wav")), [])

    def test_antiphase_stereo_does_not_reach_generator(self):
        sf.write(self.sample, np.column_stack([self.signal, -self.signal]), 16000, subtype="PCM_16")
        # Use exact signed PCM values so quantization does not leave a tiny DC offset.
        pcm = (self.signal * 30000).astype(np.int16)
        sf.write(self.sample, np.column_stack([pcm, -pcm]), 16000, subtype="PCM_16")
        with self.assertRaisesRegex(ValueError, "REFERENCE_SILENT_AFTER_DOWNMIX"):
            self.run_synthesis()
        self.model.generate_voice_clone.assert_not_called()

    def test_repeated_experiments_never_overwrite_previous_results(self):
        first = self.run_synthesis()
        second = self.run_synthesis()
        self.assertNotEqual(first["audio"], second["audio"])
        self.assertTrue(Path(first["audio"]).is_file())
        self.assertTrue(Path(second["audio"]).is_file())


if __name__ == "__main__":
    unittest.main()

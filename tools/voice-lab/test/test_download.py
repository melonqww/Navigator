import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))
import download_weights as downloader


class Response(io.BytesIO):
    def __init__(self, content, status=206, content_range="bytes 0-3/4"):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": content_range}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="navigator-download-")
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "0.part"

    def test_valid_chunk_is_saved_atomically(self):
        opener = Mock(return_value=Response(b"abcd"))
        downloader.fetch_chunk("https://example.invalid/model", 0, 3, 4, self.target, opener)
        self.assertEqual(self.target.read_bytes(), b"abcd")
        self.assertFalse(self.target.with_suffix(".tmp").exists())
        self.assertEqual(opener.call_args.args[0].headers["Range"], "bytes=0-3")

    def test_completed_chunk_is_reused_without_network(self):
        self.target.write_bytes(b"abcd")
        opener = Mock()
        downloader.fetch_chunk("https://example.invalid/model", 0, 3, 4, self.target, opener)
        opener.assert_not_called()

    def test_server_ignoring_range_is_rejected(self):
        opener = Mock(side_effect=lambda *a, **kw: Response(b"abcd", status=200))
        with patch.object(downloader.time, "sleep"):
            with self.assertRaises(RuntimeError):
                downloader.fetch_chunk("https://example.invalid/model", 0, 3, 4, self.target, opener)
        self.assertEqual(opener.call_count, 3)
        self.assertFalse(self.target.exists())

    def test_cached_wrong_range_is_rejected(self):
        opener = Mock(side_effect=lambda *a, **kw: Response(b"abcd", content_range="bytes 4-7/8"))
        with patch.object(downloader.time, "sleep"):
            with self.assertRaises(RuntimeError):
                downloader.fetch_chunk("https://example.invalid/model", 0, 3, 4, self.target, opener)
        self.assertFalse(self.target.exists())

    def test_short_response_does_not_become_completed_part(self):
        opener = Mock(side_effect=lambda *a, **kw: Response(b"ab"))
        with patch.object(downloader.time, "sleep"):
            with self.assertRaises(RuntimeError):
                downloader.fetch_chunk("https://example.invalid/model", 0, 3, 4, self.target, opener)
        self.assertFalse(self.target.exists())

    def test_digest_detects_corrupt_weights(self):
        self.target.write_bytes(b"abc")
        self.assertEqual(downloader.sha256_file(self.target), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.target.write_bytes(b"abd")
        self.assertNotEqual(downloader.sha256_file(self.target), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()

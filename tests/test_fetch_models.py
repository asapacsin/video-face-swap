import importlib.util
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
spec = importlib.util.spec_from_file_location("fetch_models", SCRIPT)
fetch_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_models)


class Response(BytesIO):
    def __init__(
        self, data: bytes, status: int, fail_after: bool = False, length: int | None = None
    ) -> None:
        super().__init__(data)
        self.status = status
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.fail_after = fail_after

    def read(self, *args) -> bytes:
        data = super().read(*args)
        if not data and self.fail_after:
            raise TimeoutError
        return data


class FetchTest(unittest.TestCase):
    def fetch(self, responses, existing=None):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.onnx"
            if existing is not None:
                path.with_name("model.onnx.part").write_bytes(existing)
            ranges = []

            def urlopen(request, timeout):
                ranges.append(request.headers["Range"])
                item = responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            with (
                mock.patch.object(fetch_models.urllib.request, "urlopen", urlopen),
                mock.patch.object(fetch_models.time, "sleep"),
            ):
                try:
                    fetch_models.fetch("https://example.test/model", path)
                except OSError:
                    return None, ranges
            return path.read_bytes(), ranges

    def test_timeout_resumes_from_the_bytes_already_saved(self):
        data, ranges = self.fetch([Response(b"abc", 200, fail_after=True), Response(b"def", 206)])
        self.assertEqual(data, b"abcdef")
        self.assertEqual(ranges, ["bytes=0-", "bytes=3-"])

    def test_a_cleanly_closed_short_download_resumes(self):
        data, ranges = self.fetch(
            [Response(b"abc", 200, length=6), Response(b"def", 206, length=3)]
        )
        self.assertEqual(data, b"abcdef")
        self.assertEqual(ranges, ["bytes=0-", "bytes=3-"])

    def test_server_ignoring_the_range_restarts_the_file(self):
        data, _ = self.fetch([Response(b"whole", 200)], existing=b"old")
        self.assertEqual(data, b"whole")

    def test_gives_up_after_five_attempts_without_progress(self):
        data, ranges = self.fetch([TimeoutError() for _ in range(5)])
        self.assertIsNone(data)
        self.assertEqual(len(ranges), 5)


if __name__ == "__main__":
    unittest.main()

"""The reviewer bridge must stay read-only and fail closed. No test calls a real model."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ai_native"))

import claude_bridge  # noqa: E402

MODEL = "claude-opus-5-5"


def envelope(result: str, is_error: bool = False, model: str = MODEL) -> str:
    return json.dumps(
        {"type": "result", "is_error": is_error, "result": result, "modelUsage": {model: {}}}
    )


class ClaudeBridgeTest(unittest.TestCase):
    def command(self) -> list[str]:
        return claude_bridge.claude_command("claude", MODEL, "high")

    def option(self, name: str) -> str:
        command = self.command()
        return command[command.index(name) + 1]

    def test_selects_model_and_effort(self):
        self.assertEqual(self.option("--model"), MODEL)
        self.assertEqual(self.option("--effort"), "high")

    def test_cannot_edit_or_run_arbitrary_commands(self):
        self.assertEqual(self.option("--permission-mode"), "dontAsk")
        allowed = self.option("--allowedTools").split(",")
        for tool in ("Edit", "Write", "NotebookEdit", "Bash"):
            self.assertNotIn(tool, allowed)
        self.assertTrue(all(t.startswith("Bash(git ") for t in allowed if t.startswith("Bash")))
        self.assertNotIn("Edit", self.option("--tools"))
        self.assertNotIn("Write", self.option("--tools"))

    def test_personal_media_and_keys_are_denied(self):
        denied = self.option("--disallowedTools")
        for path in ("Read(./assets/**)", "Read(./outputs/**)", "Read(~/.ssh/**)"):
            self.assertIn(path, denied)

    def test_parses_review_with_surrounding_text(self):
        stdout = envelope('Done.\n```json\n{"verdict": "REWORK"}\n```')
        self.assertEqual(claude_bridge.parse_review(stdout, MODEL), {"verdict": "REWORK"})

    def test_invalid_output_fails_closed(self):
        for stdout in (
            "not json",
            envelope("no object here"),
            envelope("{broken"),
            envelope('{"verdict": "PASS"}', is_error=True),
        ):
            with self.subTest(stdout=stdout):
                with self.assertRaises(ValueError):
                    claude_bridge.parse_review(stdout, MODEL)

    def test_review_on_another_model_is_rejected(self):
        stdout = envelope('{"verdict": "PASS"}', model="claude-haiku-5-5")
        with self.assertRaisesRegex(ValueError, "did not run on"):
            claude_bridge.parse_review(stdout, MODEL)


if __name__ == "__main__":
    unittest.main()

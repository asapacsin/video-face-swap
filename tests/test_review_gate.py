"""The reviewer gate must reject false PASS results. No test calls a real model."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "ai_native"
sys.path.insert(0, str(SCRIPTS))

import governance  # noqa: E402
from control_plane import verify_decision  # noqa: E402

FAKE_REVIEWER = """
import json, os, sys
mode = sys.argv[1]
packet = json.load(open(os.environ["AI_NATIVE_REVIEW_PACKET"], encoding="utf-8"))
result = {
    "verdict": "REWORK" if mode == "rework" else "PASS",
    "reviewed_fingerprint": packet["snapshot"]["fingerprint"],
    "reviewed_contract_hash": packet["contract_hash"],
    "findings": [],
    "next_action": "none",
}
if mode == "wrong-fingerprint":
    result["reviewed_fingerprint"] = "0" * 64
if mode == "wrong-contract":
    result["reviewed_contract_hash"] = "0" * 64
if mode == "edits-source":
    open("src.txt", "a").write("changed by reviewer")
if mode == "bad-json":
    open(os.environ["AI_NATIVE_REVIEW_RESULT"], "w").write("PASS")
else:
    json.dump(result, open(os.environ["AI_NATIVE_REVIEW_RESULT"], "w"))
"""


def git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    return subprocess.run(
        ["git", "-C", str(repo), *identity, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


class ReviewGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.repo = root / "repo"
        self.repo.mkdir()
        self.reviewer = root / "fake_reviewer.py"
        self.reviewer.write_text(FAKE_REVIEWER, encoding="utf-8")
        self.contract_path = root / "contract.json"
        git(self.repo, "init", "-q")
        (self.repo / ".gitignore").write_text(".ai-native/\n", encoding="utf-8")
        (self.repo / "src.txt").write_text("original", encoding="utf-8")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-q", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        self.write_contract("raise SystemExit(0)")

    def write_contract(self, gate_code: str) -> dict:
        contract = {
            "schema_version": 1,
            "task_id": "review-gate",
            "goal": "Exercise the gate",
            "acceptance": "Only genuine evidence passes",
            "base_commit": self.base,
            "approved_paths": ["src.txt"],
            "budgets": {"max_attempts": 1, "max_total_seconds": 120},
            "gates": [
                {
                    "id": "unit-tests",
                    "level": "unit",
                    "command": [sys.executable, "-c", gate_code],
                    "timeout_seconds": 60,
                }
            ],
        }
        self.contract_path.write_text(json.dumps(contract), encoding="utf-8")
        return contract

    def cycle(self, mode: str | None) -> tuple[int, dict]:
        command = [sys.executable, str(SCRIPTS / "ai_native.py"), "cycle"]
        command += ["--repo", str(self.repo), "--contract", str(self.contract_path)]
        if mode:
            command += ["--reviewer", f"{sys.executable} {self.reviewer} {mode}"]
        code = subprocess.run(command, capture_output=True, timeout=120, check=False).returncode
        latest = self.repo / ".ai-native" / "runs" / "review-gate" / "latest.json"
        return code, json.loads(latest.read_text(encoding="utf-8"))

    def assert_rejected(self, mode: str, expected_code: int) -> None:
        code, latest = self.cycle(mode)
        self.assertEqual(code, expected_code)
        self.assertNotEqual(latest["control_plane_decision"]["action"], "TASK_VERIFIED")

    def test_matching_review_is_verified(self):
        code, latest = self.cycle("pass")
        self.assertEqual(code, 0)
        self.assertEqual(latest["control_plane_decision"]["action"], "TASK_VERIFIED")
        self.assertTrue(verify_decision(latest["control_plane_decision"]))

    def test_missing_reviewer_is_not_verified(self):
        code, latest = self.cycle(None)
        self.assertEqual(code, 4)
        self.assertEqual(latest["verdict"], "NOT_VERIFIED")

    def test_wrong_fingerprint_is_rejected(self):
        self.assert_rejected("wrong-fingerprint", 4)

    def test_wrong_contract_hash_is_rejected(self):
        self.assert_rejected("wrong-contract", 4)

    def test_malformed_result_is_rejected(self):
        self.assert_rejected("bad-json", 4)

    def test_reviewer_changing_source_is_rejected(self):
        self.assert_rejected("edits-source", 4)

    def test_reviewer_rework_is_not_verified(self):
        self.assert_rejected("rework", 2)

    def test_reviewer_pass_cannot_override_failed_gate(self):
        self.write_contract("raise SystemExit(1)")
        code, latest = self.cycle("pass")
        self.assertEqual(code, 2)
        self.assertEqual(latest["verdict"], "REWORK")
        self.assertEqual(latest["failed_required_gates"], ["unit-tests"])

    def test_gate_changing_source_is_rejected(self):
        self.write_contract("open('src.txt', 'a').write('changed by gate')")
        code, latest = self.cycle("pass")
        self.assertNotEqual(code, 0)
        self.assertEqual(latest["verdict"], "NOT_VERIFIED")

    def merge_ready(self) -> tuple[dict, dict]:
        (self.repo / "src.txt").write_text("feature", encoding="utf-8")
        git(self.repo, "commit", "-q", "-am", "feature")
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        governance.approve(self.repo, contract, "owner", "owner")
        self.assertEqual(self.cycle("pass")[0], 0)
        requirements = {
            "schema_version": 1,
            "requirements": [
                {
                    "id": "tests",
                    "min_level": "unit",
                    "gate_ids": ["unit-tests"],
                    "terminal_required": False,
                }
            ],
        }
        return contract, requirements

    def test_verified_committed_tip_is_merge_eligible(self):
        contract, requirements = self.merge_ready()
        self.assertEqual(governance.eligibility(self.repo, contract, requirements), [])

    def test_edit_after_review_is_not_merge_eligible(self):
        contract, requirements = self.merge_ready()
        (self.repo / "src.txt").write_text("edited after review", encoding="utf-8")
        issues = governance.eligibility(self.repo, contract, requirements)
        self.assertTrue(any("stale" in issue for issue in issues), issues)

    def test_missing_requirements_manifest_is_not_merge_eligible(self):
        contract, _ = self.merge_ready()
        issues = governance.eligibility(self.repo, contract, None)
        self.assertTrue(any("requirements" in issue for issue in issues), issues)


if __name__ == "__main__":
    unittest.main()

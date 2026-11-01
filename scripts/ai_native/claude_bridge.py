#!/usr/bin/env python3
"""Run the independent reviewer as a fresh, read-only Claude Code session.

Written for this project in the shape of the upstream codex_bridge.py. Invoked only by
scripts/review.py, and each run uses the signed-in Claude account's allowance.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ALLOWED_TOOLS = (
    "Read",
    "Grep",
    "Glob",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git show:*)",
)
# The reviewer never needs personal media, model weights or credentials.
DENIED_TOOLS = (
    "Edit",
    "Write",
    "NotebookEdit",
    "Read(./assets/**)",
    "Read(./outputs/**)",
    "Read(./.venv-gpu/**)",
    "Read(./upstream-facefusion/.assets/**)",
    "Read(~/.ssh/**)",
)
PROMPT = """You are an INDEPENDENT READ-ONLY software reviewer in a fresh session. You did not write
this code and have no knowledge of why it was written. Inspect git status, git diff HEAD and the
changed and untracked files yourself. Read docs/REVIEWER.md and apply its checklist. Check real
behavior against the contract and the required evidence levels. Investigate correctness,
security, failure cases, tests that prove less than they claim, and GPU evidence gaps. Do not
trust documentation claims without checking them. NEVER edit files, commit or run inference.

PASS only if every required gate passed and the task is satisfied at its asserted evidence level.
Use REWORK for a fixable defect, BLOCKED for a named missing dependency, and NOT_VERIFIED for
insufficient proof. Your final message must be ONLY one JSON object with exactly these keys:
verdict, reviewed_fingerprint, reviewed_contract_hash, findings, next_action.
verdict is PASS, REWORK, BLOCKED or NOT_VERIFIED. reviewed_fingerprint must exactly equal
packet.snapshot.fingerprint and reviewed_contract_hash must exactly equal packet.contract_hash.
findings is an array of strings, most severe first, each naming the file and the concrete
failure. next_action is a string.

Review packet:
"""


def claude_command(executable: str, model: str, effort: str) -> list[str]:
    return [
        executable,
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Read,Grep,Glob,Bash",
        "--allowedTools",
        ",".join(ALLOWED_TOOLS),
        "--disallowedTools",
        ",".join(DENIED_TOOLS),
        "--setting-sources",
        "project",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]


def parse_review(stdout: str, model: str) -> dict:
    """Return the reviewer's JSON object; anything else raises, which fails the review closed."""
    envelope = json.loads(stdout)
    if not isinstance(envelope, dict) or envelope.get("is_error"):
        message = envelope.get("result") if isinstance(envelope, dict) else None
        raise ValueError(f"Claude reported an error instead of a review: {message}")
    # A silent fallback to another model must not count as the review that was asked for.
    models = envelope.get("modelUsage") or {}
    if not any(model in name for name in models):
        raise ValueError(f"review did not run on {model}: {sorted(models)}")
    text = str(envelope.get("result", "")).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("review contained no JSON object")
    review = json.loads(text[start : end + 1])
    if not isinstance(review, dict):
        raise ValueError("review must be a JSON object")
    return review


def main() -> int:
    packet_path = os.environ.get("AI_NATIVE_REVIEW_PACKET")
    result_path = os.environ.get("AI_NATIVE_REVIEW_RESULT")
    if not packet_path or not result_path:
        print("Missing review packet/result env paths", file=sys.stderr)
        return 12
    executable = shutil.which("claude")
    if not executable:
        print("Claude Code CLI not found on PATH", file=sys.stderr)
        return 10
    packet = Path(packet_path).read_text(encoding="utf-8")
    model = os.environ.get("AI_NATIVE_REVIEWER_MODEL", "claude-opus-5-5")
    command = claude_command(executable, model, os.environ.get("AI_NATIVE_REVIEWER_EFFORT", "high"))
    process = subprocess.run(
        command,
        input=PROMPT + packet,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(os.environ.get("AI_NATIVE_REVIEWER_TIMEOUT", "240")),
        check=False,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout)[-500:]
        print(f"Claude exited with {process.returncode}: {detail}", file=sys.stderr)
        return process.returncode
    review = parse_review(process.stdout, model)
    Path(result_path).write_text(json.dumps(review), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print("Claude reviewer failed: " + str(error), file=sys.stderr)
        raise SystemExit(13)

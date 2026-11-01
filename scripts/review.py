#!/usr/bin/env python3
"""Run a task's required gates, then a fresh read-only Claude review of the exact tree."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTROLLER = REPO / "scripts" / "ai_native" / "ai_native.py"
BRIDGE = REPO / "scripts" / "ai_native" / "claude_bridge.py"
# Names only. Without these the reviewer's stripped environment cannot reach the model service.
# CLAUDE_CONFIG_DIR stays out: an app session's profile may not be signed in to the CLI.
REVIEWER_ENV = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY")


def reviewer_command() -> str:
    command = f"{sys.executable} {BRIDGE}"
    if any(" " in str(part) for part in (sys.executable, BRIDGE)):
        raise SystemExit("error: the controller splits this command on spaces; use a path without")
    return command


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("AI_NATIVE_REVIEWER_MODEL", "claude-opus-5-5")
    env.setdefault("AI_NATIVE_REVIEWER_EFFORT", "high")
    env["AI_NATIVE_REVIEWER_ENV_ALLOWLIST"] = ",".join(
        name for name in REVIEWER_ENV if name in env or name.lower() in env
    )
    env["AI_NATIVE_GATE_ENV_ALLOWLIST"] = "PYTHONUTF8"
    env["PYTHONUTF8"] = "1"
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="Run gates without a model call; the verdict stays NOT_VERIFIED",
    )
    args = parser.parse_args(argv)
    contract = args.contract.resolve()
    command = [sys.executable, str(CONTROLLER), "cycle", "--repo", str(REPO)]
    command += ["--contract", str(contract)]
    if not args.gates_only:
        command += ["--reviewer", reviewer_command()]
    # Contract gates name the venv by relative path, which resolves against this directory.
    os.chdir(REPO)
    code = subprocess.run(command, env=environment(), check=False).returncode
    task_id = json.loads(contract.read_text(encoding="utf-8")).get("task_id", "")
    latest = REPO / ".ai-native" / "runs" / task_id / "latest.json"
    if latest.is_file():
        outcome = json.loads(latest.read_text(encoding="utf-8"))
        print(f"Verdict: {outcome.get('verdict')}")
        for finding in outcome.get("findings", []):
            print(f"- {finding}")
        if outcome.get("next_action"):
            print(f"Next: {outcome['next_action']}")
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

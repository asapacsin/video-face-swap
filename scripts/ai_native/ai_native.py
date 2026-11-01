#!/usr/bin/env python3
"""AI-native, executor-first end-state controller (Python stdlib only).

This tool drives configured *external* executor/reviewer processes. It does NOT
impersonate an LLM, select a provider subscription, or treat tests as review.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from control_plane import canonical_hash, decide_cycle
from process_security import child_environment, redact_argv, redact_text

VERDICTS = {"PASS", "REWORK", "BLOCKED", "NOT_VERIFIED"}
TASK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
EXCLUDE_UNTRACKED = (".ai-native/", ".git/", "__pycache__/", ".pytest_cache/")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if p.returncode:
        raise RuntimeError("git " + " ".join(args) + ": " + p.stderr.decode(errors="replace")[:300])
    return p.stdout


def work_snapshot(repo: Path) -> dict:
    """Bind review to HEAD + dirty diff and untracked contents, not merely a commit."""
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    diff = git(repo, "diff", "--binary", "HEAD")
    others = git(repo, "ls-files", "-o", "--exclude-standard", "-z").split(b"\0")
    untracked = {}
    for item in others:
        if not item:
            continue
        rel = item.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        if any(rel.startswith(p) for p in EXCLUDE_UNTRACKED) or "/__pycache__/" in rel:
            continue
        f = repo / rel
        if not f.is_file() or f.is_symlink():
            continue
        digest = hashlib.sha256()
        with f.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        untracked[rel] = digest.hexdigest()
    fingerprint = sha(json.dumps({"head": head, "diff": sha(diff), "untracked": untracked}, sort_keys=True).encode())
    return {"commit": head, "diff_sha256": sha(diff), "untracked_hashes": untracked,
            "fingerprint": fingerprint, "dirty": bool(diff or untracked)}


def validate_contract(contract: dict) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("contract schema_version must be 1")
    if not TASK_RE.fullmatch(str(contract.get("task_id", ""))):
        raise ValueError("task_id must be a safe nonempty slug")
    if not str(contract.get("goal", "")).strip():
        raise ValueError("goal is required")
    gates = contract.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("nonempty gates array required")
    seen = set()
    for gate in gates:
        if not isinstance(gate, dict) or not TASK_RE.fullmatch(str(gate.get("id", ""))):
            raise ValueError("each gate needs a safe id")
        if gate["id"] in seen:
            raise ValueError("duplicate gate id: " + gate["id"])
        seen.add(gate["id"])
        if gate.get("level") not in {"static", "unit", "integration", "e2e", "device", "visual", "performance"}:
            raise ValueError("gate level must be explicit")
        if not isinstance(gate.get("command"), list) or not gate["command"] or not all(
            isinstance(s, str) and s for s in gate["command"]):
            raise ValueError("gate command requires argv list; shell operators are not interpreted")
        if gate.get("timeout_seconds", 180) <= 0:
            raise ValueError("gate timeout must be positive")
    budgets = contract.get("budgets", {})
    if not isinstance(budgets, dict) or not 1 <= budgets.get("max_attempts", 3) <= 10:
        raise ValueError("max_attempts must be 1..10")
    if budgets.get("max_total_seconds", 900) <= 0:
        raise ValueError("max_total_seconds must be positive")
    if not isinstance(contract.get("acceptance", ""), str) or not contract["acceptance"].strip():
        raise ValueError("acceptance must be explicit")
    if not any(gate.get("required", True) for gate in gates):
        raise ValueError("at least one required gate is mandatory")


def call_argv(argv: list[str], repo: Path, env: dict, timeout: float,
              sensitive_environment: dict | None = None) -> dict:
    began = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        p = subprocess.run(argv, cwd=repo, capture_output=True, text=True, env=env,
                           timeout=max(1, timeout), errors="replace")
        # Store bounded excerpts only. Never pass secrets/credentials in CLI output.
        return {"command": redact_argv(argv, sensitive_environment), "started": began, "exit_code": p.returncode,
                "stdout_tail": redact_text(p.stdout[-5000:], sensitive_environment),
                "stderr_tail": redact_text(p.stderr[-5000:], sensitive_environment),
                "status": "PASS" if p.returncode == 0 else "FAIL"}
    except subprocess.TimeoutExpired:
        return {"command": redact_argv(argv, sensitive_environment), "started": began,
                "status": "FAIL", "error": "timeout"}
    except OSError as e:
        return {"command": argv, "started": began, "status": "BLOCKED",
                "error": redact_text(e.__class__.__name__ + ": " + str(e), sensitive_environment)}


def parse_command(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = shlex.split(value, posix=(os.name != "nt"))
    return parsed or None


def review_packet(contract: dict, snapshot: dict, gates: list[dict], attempt: int) -> dict:
    return {"schema_version": 1, "task_id": contract["task_id"], "goal": contract["goal"],
            "contract_hash": canonical_hash(contract),
            "acceptance": contract["acceptance"], "boundaries": contract.get("boundaries", []),
            "attempt": attempt, "snapshot": snapshot, "gates": gates,
            "review_instructions": (
                "INDEPENDENT, READ-ONLY REVIEW. Inspect actual git diff and changed files, not executor prose. "
                "Validate acceptance, architecture, invariant safety, proof levels, missing E2E, and secret risks. "
                "Write review result JSON with verdict PASS / REWORK / BLOCKED / NOT_VERIFIED, "
                "reviewed_fingerprint exactly equal to snapshot.fingerprint, reviewed_contract_hash exactly equal "
                "to contract_hash, findings array, next_action string. PASS only if all required acceptance levels "
                "have genuine evidence. Do not modify repo."
            )}


def attempt_cycle(repo: Path, contract: dict, attempt: int, executor: list[str] | None,
                  reviewer: list[str] | None, deadline: float) -> dict:
    task_id = contract["task_id"]
    folder = repo / ".ai-native" / "runs" / task_id / f"attempt-{attempt:02}"
    folder.mkdir(parents=True, exist_ok=True)
    parent_env = os.environ.copy()
    protocol = {"AI_NATIVE_TASK_ID": task_id, "AI_NATIVE_ATTEMPT": str(attempt),
                "AI_NATIVE_CONTRACT": str((folder / "contract.json").resolve()),
                "AI_NATIVE_FEEDBACK": str((folder.parent / "latest.json").resolve())}
    save_json(folder / "contract.json", contract)
    execution = None
    if executor:
        execution = call_argv(executor, repo, child_environment("executor", parent_env, protocol),
                              min(300, max(1, deadline - time.monotonic())), parent_env)
        save_json(folder / "executor.json", execution)
        if execution["status"] != "PASS":
            outcome = "BLOCKED" if execution["status"] == "BLOCKED" else "REWORK"
            return record(folder, {"verdict": outcome, "next_action": "Resolve executor command failure", "executor": execution, "attempt": attempt})
    # Capture the exact tree before verification. A test/build command that mutates
    # source invalidates the evidence rather than silently redefining what was tested.
    pre_gate_snapshot = work_snapshot(repo)
    gates = []
    for gate in contract["gates"]:
        if time.monotonic() >= deadline:
            gates.append({"id": gate["id"], "level": gate["level"], "status": "BLOCKED", "error": "time budget reached"})
            continue
        outcome = call_argv(gate["command"], repo, child_environment("gate", parent_env, protocol),
                            min(gate.get("timeout_seconds", 180), max(1, deadline - time.monotonic())), parent_env)
        outcome.update({"id": gate["id"], "level": gate["level"], "required": gate.get("required", True)})
        gates.append(outcome)
    for gate in gates:
        gate["source_fingerprint"] = pre_gate_snapshot["fingerprint"]
    save_json(folder / "gates.json", {"gates": gates,
                                      "source_fingerprint": pre_gate_snapshot["fingerprint"]})
    snapshot = work_snapshot(repo)
    if snapshot["fingerprint"] != pre_gate_snapshot["fingerprint"]:
        return record(folder, {"verdict": "NOT_VERIFIED",
                               "next_action": "Restore source or explicitly ignore generated test outputs; gates mutated the source tree",
                               "reason": "source fingerprint changed while verification gates were running",
                               "attempt": attempt, "snapshot": snapshot,
                               "gate_source_fingerprint": pre_gate_snapshot["fingerprint"],
                               "failed_required_gates": [g["id"] for g in gates if g.get("required") and g["status"] != "PASS"]})
    packet = review_packet(contract, snapshot, gates, attempt)
    save_json(folder / "review_packet.json", packet)
    req_failures = [g for g in gates if g.get("required") and g["status"] != "PASS"]
    if not reviewer:
        return record(folder, {"verdict": "NOT_VERIFIED", "next_action": "Configure an independent reviewer command",
                               "reason": "No external reviewer was invoked", "attempt": attempt, "snapshot": snapshot,
                               "failed_required_gates": [g["id"] for g in req_failures]})
    review_protocol = {**protocol,
                       "AI_NATIVE_REVIEW_PACKET": str((folder / "review_packet.json").resolve()),
                       "AI_NATIVE_REVIEW_RESULT": str((folder / "review_result.json").resolve())}
    review_limit = contract.get("budgets", {}).get("reviewer_timeout_seconds", 240)
    review_protocol["AI_NATIVE_REVIEWER_TIMEOUT"] = str(review_limit)
    rv = call_argv(reviewer, repo, child_environment("reviewer", parent_env, review_protocol),
                   min(review_limit + 30, max(1, deadline - time.monotonic())), parent_env)
    save_json(folder / "reviewer_process.json", rv)
    if rv["status"] != "PASS":
        return record(folder, {"verdict": "BLOCKED" if rv["status"] == "BLOCKED" else "NOT_VERIFIED",
                               "next_action": "Restore reviewer command and rerun review", "attempt": attempt,
                               "reviewer_process": rv})
    try:
        response = load_json(folder / "review_result.json")
        if response.get("verdict") not in VERDICTS:
            raise ValueError("invalid reviewer verdict")
        if response.get("reviewed_fingerprint") != snapshot["fingerprint"]:
            raise ValueError("reviewer reviewed wrong snapshot fingerprint")
        if response.get("reviewed_contract_hash") != packet["contract_hash"]:
            raise ValueError("reviewer reviewed wrong contract hash")
        if (not isinstance(response.get("findings"), list)
                or not all(isinstance(item, str) for item in response["findings"])
                or not isinstance(response.get("next_action"), str)):
            raise ValueError("reviewer must supply findings array and next_action")
        if work_snapshot(repo)["fingerprint"] != snapshot["fingerprint"]:
            raise ValueError("repository changed while reviewer was running")
        response["findings"] = [redact_text(item, parent_env) for item in response["findings"]]
        response["next_action"] = redact_text(response["next_action"], parent_env)
        response = {key: response[key] for key in ("verdict", "reviewed_fingerprint",
                    "reviewed_contract_hash", "findings", "next_action")}
        save_json(folder / "review_result.json", response)
        reviewer_verdict = response["verdict"]
        outcome = reviewer_verdict
        if req_failures and outcome == "PASS":
            outcome = "BLOCKED" if any(g["status"] == "BLOCKED" for g in req_failures) else "REWORK"
            response["findings"].append("Reviewer PASS rejected: required gates lack passing evidence")
        response.update({"verdict": outcome, "reviewer_verdict": reviewer_verdict,
                         "reviewer_process": rv,
                         "attempt": attempt, "snapshot": snapshot,
                         "gate_source_fingerprint": pre_gate_snapshot["fingerprint"],
                         "failed_required_gates": [g["id"] for g in req_failures]})
        return record(folder, response)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return record(folder, {"verdict": "NOT_VERIFIED", "next_action": "Fix invalid or stale review evidence",
                               "reason": redact_text(str(e), parent_env), "attempt": attempt, "snapshot": snapshot,
                               "reviewer_process": rv, "gate_source_fingerprint": pre_gate_snapshot["fingerprint"]})


def record(folder: Path, outcome: dict) -> dict:
    # Bind every verdict (including failures) to the exact task contract.
    contract = load_json(folder / "contract.json")
    outcome["contract_hash"] = sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    outcome["task_id"] = contract["task_id"]
    save_json(folder / "verdict.json", outcome)
    save_json(folder.parent / "latest.json", outcome)
    return outcome


def cycle(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    contract = load_json(Path(args.contract).resolve())
    validate_contract(contract)
    try:
        work_snapshot(repo)
    except (OSError, RuntimeError) as exc:
        raise ValueError("target must be an initialized git repository: " + str(exc)) from exc
    executor = parse_command(args.executor or os.environ.get("AI_NATIVE_EXECUTOR_CMD"))
    reviewer = parse_command(args.reviewer or os.environ.get("AI_NATIVE_REVIEWER_CMD"))
    max_attempts = contract.get("budgets", {}).get("max_attempts", 3) if executor else 1
    deadline = time.monotonic() + contract.get("budgets", {}).get("max_total_seconds", 900)
    outcome = None
    final_decision = None
    for attempt in range(1, max_attempts + 1):
        if time.monotonic() >= deadline:
            break
        outcome = attempt_cycle(repo, contract, attempt, executor, reviewer, deadline)
        attempt_folder = repo / ".ai-native" / "runs" / contract["task_id"] / f"attempt-{attempt:02}"
        try:
            gate_record = load_json(attempt_folder / "gates.json")
        except (OSError, ValueError):
            gate_record = {}
        current_fingerprint = outcome.get("snapshot", {}).get("fingerprint")
        if not current_fingerprint:
            current_fingerprint = work_snapshot(repo)["fingerprint"]
        executor_result = outcome.get("executor", {})
        reviewer_process = outcome.get("reviewer_process", {})
        reviewer_verdict = outcome.get("reviewer_verdict")
        if reviewer_verdict is None and outcome.get("reviewed_fingerprint"):
            reviewer_verdict = outcome.get("verdict")
        control_state = {
            "schema_version": 1, "task_status": outcome.get("verdict", "NOT_VERIFIED"),
            "attempt": attempt, "max_attempts": max_attempts,
            "deadline_exceeded": time.monotonic() >= deadline,
            "executor_configured": bool(executor),
            "executor_status": executor_result.get("status", "PASS"),
            "reviewer_configured": bool(reviewer),
            "reviewer_process_status": reviewer_process.get("status", "NOT_RUN"),
            "reviewer_verdict": reviewer_verdict,
            "reviewer_independent": bool(reviewer) and (not executor or reviewer != executor),
            "current_fingerprint": current_fingerprint,
            "reviewed_fingerprint": outcome.get("reviewed_fingerprint"),
            "contract_hash": canonical_hash(contract),
            "reviewed_contract_hash": outcome.get("reviewed_contract_hash"),
            "gates": gate_record.get("gates", []),
            "gate_source_fingerprint": gate_record.get("source_fingerprint", outcome.get("gate_source_fingerprint"))
        }
        final_decision = decide_cycle(control_state)
        outcome["control_plane_decision"] = final_decision
        save_json(attempt_folder / "verdict.json", outcome)
        save_json(attempt_folder.parent / "latest.json", outcome)
        print(f"Attempt {attempt}/{max_attempts}: {outcome['verdict']} | control={final_decision['action']}")
        if final_decision["action"] != "RETRY_EXECUTOR":
            break
    if outcome is None:
        print("BLOCKED: budget expired before first attempt")
        return 3
    if final_decision and final_decision["action"] == "ESCALATE_RETRY_LIMIT":
        print("Retry cap reached; deterministic control plane requires diagnosis or Architect escalation.")
    print("Control-plane decision:", final_decision["action"] if final_decision else "UNAVAILABLE")
    print("Evidence:", repo / ".ai-native" / "runs" / contract["task_id"] / "latest.json")
    return {
        "TASK_VERIFIED": 0,
        "RETRY_EXECUTOR": 2,
        "ESCALATE_RETRY_LIMIT": 2,
        "ESCALATE_NO_EXECUTOR": 2,
        "WAIT_EXTERNAL": 3,
        "ESCALATE_BUDGET": 3,
        "CONFIGURE_REVIEWER": 4,
        "REVIEW_REQUIRED": 4,
        "STOP_NOT_VERIFIED": 4,
        "STOP_INVALID_STATE": 5,
        "INVALID_STATE": 5,
    }.get(final_decision["action"] if final_decision else "ESCALATE_BUDGET", 4)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="cmd", required=True)
    v = subs.add_parser("validate", help="check contract without executing commands")
    v.add_argument("--contract", required=True)
    c = subs.add_parser("cycle", help="execute bounded executor -> gates -> reviewer loop")
    c.add_argument("--contract", required=True)
    c.add_argument("--repo", default=".")
    c.add_argument("--executor", help="optional CLI command to invoke actual executor agent")
    c.add_argument("--reviewer", help="CLI command to invoke an actual independent reviewer")
    args = p.parse_args(argv)
    try:
        if args.cmd == "validate":
            validate_contract(load_json(Path(args.contract)))
            print("Contract is valid; nothing has been executed.")
            return 0
        return cycle(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())

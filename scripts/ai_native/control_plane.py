#!/usr/bin/env python3
"""Pure deterministic control plane. No LLM, network, subprocess, or clock calls.

Inputs are observed state; this module decides the only permitted next action.
It is deliberately small enough to test exhaustively. A model may propose work,
but cannot override these rules or convert missing evidence into PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

RULESET_VERSION = "1.0.0"
VERDICTS = {"PASS", "REWORK", "BLOCKED", "NOT_VERIFIED"}
GATE_STATUSES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}
WORK_STATUSES = {"NOT_RUN", "RUNNING", "PASS", "FAIL", "BLOCKED_EXTERNAL", "HUMAN_REQUIRED", "NOT_APPLICABLE"}
OWNERS = {"AUTONOMOUS", "HUMAN_PHYSICAL", "HUMAN_DECISION", "HUMAN_CREDENTIAL", "EXTERNAL_RESOURCE"}
TERMINAL_WORK = {"PASS", "NOT_APPLICABLE"}
PROOF_LADDER = ("static", "unit", "integration", "e2e")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _decision(action: str, reasons: list[str], **details) -> dict:
    payload = {"schema_version": 1, "ruleset_version": RULESET_VERSION,
               "action": action, "reasons": reasons, **details}
    payload["decision_sha256"] = canonical_hash(payload)
    return payload


def _retry(state: dict, reason: str) -> dict:
    attempt, limit = state["attempt"], state["max_attempts"]
    if not state.get("executor_configured", False):
        return _decision("ESCALATE_NO_EXECUTOR", [reason, "executor_not_configured"],
                         attempt=attempt, max_attempts=limit)
    if attempt < limit:
        return _decision("RETRY_EXECUTOR", [reason, "retry_budget_remains"],
                         attempt=attempt, max_attempts=limit)
    return _decision("ESCALATE_RETRY_LIMIT", [reason, "retry_budget_exhausted"],
                     attempt=attempt, max_attempts=limit)


def _required_gate_problem(gates: list[dict], source_fingerprint: str, gate_source_fingerprint: str | None) -> tuple[str | None, list[str]]:
    required = [g for g in gates if g.get("required", True)]
    if not required:
        return "NO_REQUIRED_GATES", ["no_required_gates"]
    if not source_fingerprint or not gate_source_fingerprint or gate_source_fingerprint != source_fingerprint:
        return "STALE_GATE_EVIDENCE", ["gate_source_fingerprint_missing_or_mismatch"]
    for gate in required:
        gid = gate.get("id", "<missing-id>")
        status = gate.get("status")
        if status == "BLOCKED":
            return "BLOCKED_GATE", [f"required_gate_blocked:{gid}"]
        if status == "FAIL":
            return "FAILED_GATE", [f"required_gate_failed:{gid}"]
        if status != "PASS":
            return "INCOMPLETE_GATE", [f"required_gate_not_passed:{gid}"]
        if gate.get("source_fingerprint") and gate["source_fingerprint"] != source_fingerprint:
            return "STALE_GATE_EVIDENCE", [f"gate_fingerprint_mismatch:{gid}"]
    return None, []


def verify_decision(record: dict) -> bool:
    """Validate a deterministic decision record's canonical digest."""
    if not isinstance(record, dict) or not isinstance(record.get("decision_sha256"), str):
        return False
    payload = dict(record)
    claimed = payload.pop("decision_sha256")
    return HEX64.fullmatch(claimed) is not None and canonical_hash(payload) == claimed


def decide_cycle(state: dict) -> dict:
    """Decide the next action for one implementation/review cycle from observed facts."""
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        return _decision("INVALID_STATE", ["schema_version_must_be_1"])
    for key in ("attempt", "max_attempts"):
        if isinstance(state.get(key), bool) or not isinstance(state.get(key), int) or state[key] < 1:
            return _decision("INVALID_STATE", [f"{key}_must_be_positive_integer"])
    if state["attempt"] > state["max_attempts"]:
        return _decision("INVALID_STATE", ["attempt_exceeds_max_attempts"])
    for key in ("deadline_exceeded", "executor_configured", "reviewer_configured", "reviewer_independent"):
        if not isinstance(state.get(key), bool):
            return _decision("INVALID_STATE", [f"{key}_must_be_boolean"])
    task_status = state.get("task_status")
    if task_status not in {"PASS", "REWORK", "BLOCKED", "NOT_VERIFIED"}:
        return _decision("INVALID_STATE", ["invalid_task_status"])
    if not isinstance(state.get("current_fingerprint", ""), str):
        return _decision("INVALID_STATE", ["current_fingerprint_must_be_string"])
    if state["deadline_exceeded"]:
        return _decision("ESCALATE_BUDGET", ["wall_clock_budget_exhausted"],
                         attempt=state["attempt"], max_attempts=state["max_attempts"])

    executor_status = state.get("executor_status", "PASS")
    if executor_status == "BLOCKED":
        return _decision("WAIT_EXTERNAL", ["executor_dependency_unavailable"])
    if executor_status == "FAIL":
        return _retry(state, "executor_failed")

    gates = state.get("gates")
    if not isinstance(gates, list) or any(
        not isinstance(g, dict) or g.get("status") not in GATE_STATUSES
        or not isinstance(g.get("id"), str) or not g.get("id")
        or not isinstance(g.get("required", True), bool)
        for g in gates
    ):
        return _decision("INVALID_STATE", ["gates_must_have_valid_id_status_and_required_flag"])
    source_fingerprint = state.get("current_fingerprint", "")
    gate_problem, gate_reasons = _required_gate_problem(
        gates, source_fingerprint, state.get("gate_source_fingerprint"))
    if gate_problem == "BLOCKED_GATE":
        return _decision("WAIT_EXTERNAL", gate_reasons)
    if gate_problem == "FAILED_GATE":
        return _retry(state, gate_reasons[0])
    if gate_problem:
        return _decision("STOP_NOT_VERIFIED", gate_reasons)

    if task_status == "BLOCKED":
        return _decision("WAIT_EXTERNAL", ["task_marked_blocked"])
    if not state["reviewer_configured"]:
        return _decision("CONFIGURE_REVIEWER", ["independent_reviewer_not_configured"])
    if state.get("reviewer_process_status") != "PASS":
        if state.get("reviewer_process_status") in {"BLOCKED", "NOT_RUN"}:
            return _decision("CONFIGURE_REVIEWER", ["reviewer_process_unavailable"])
        return _decision("STOP_NOT_VERIFIED", ["reviewer_process_failed"])

    reviewer_verdict = state.get("reviewer_verdict")
    if reviewer_verdict == "BLOCKED":
        return _decision("WAIT_EXTERNAL", ["reviewer_reported_external_blocker"])
    if reviewer_verdict == "REWORK":
        return _retry(state, "independent_reviewer_requested_rework")
    if reviewer_verdict == "NOT_VERIFIED":
        return _decision("STOP_NOT_VERIFIED", ["reviewer_could_not_verify"])
    if reviewer_verdict != "PASS":
        return _decision("STOP_NOT_VERIFIED", ["reviewer_verdict_missing_or_invalid"])
    if not state["reviewer_independent"]:
        return _decision("STOP_NOT_VERIFIED", ["reviewer_is_not_independent"])
    if not HEX64.fullmatch(source_fingerprint):
        return _decision("STOP_NOT_VERIFIED", ["current_source_fingerprint_missing_or_invalid"])
    if state.get("reviewed_fingerprint") != source_fingerprint:
        return _decision("REVIEW_REQUIRED", ["review_does_not_match_current_source"])
    contract_hash = state.get("contract_hash")
    if not isinstance(contract_hash, str) or not HEX64.fullmatch(contract_hash):
        return _decision("STOP_NOT_VERIFIED", ["contract_hash_missing_or_invalid"])
    if state.get("reviewed_contract_hash") != contract_hash:
        return _decision("REVIEW_REQUIRED", ["review_does_not_match_current_contract"])
    if task_status != "PASS":
        return _decision("STOP_NOT_VERIFIED", ["task_verdict_not_pass"])
    return _decision("TASK_VERIFIED", ["required_gates_passed", "independent_review_passed",
                                      "source_and_contract_fingerprints_match"],
                     source_fingerprint=source_fingerprint, contract_hash=contract_hash)


def _validate_work(work: list[dict]) -> list[str]:
    errors: list[str] = []
    if not work:
        return ["empty_work_registry"]
    by_id: dict[str, dict] = {}
    for item in work:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            errors.append("every work item requires a nonempty id")
            continue
        ident = item["id"]
        if ident in by_id:
            errors.append(f"duplicate_work_id:{ident}")
        by_id[ident] = item
        if item.get("status") not in WORK_STATUSES or item.get("owner") not in OWNERS:
            errors.append(f"invalid_owner_or_status:{ident}")
        if not isinstance(item.get("depends_on"), list):
            errors.append(f"dependencies_must_be_list:{ident}")
        elif any(not isinstance(dep, str) or not dep for dep in item["depends_on"]):
            errors.append(f"invalid_dependency_id:{ident}")
        required = item.get("requires", [])
        if (not isinstance(required, list) or any(
                not isinstance(name, str) or not name.strip() for name in required)
                or len(required) != len(set(required))):
            errors.append(f"invalid_required_capabilities:{ident}")
        priority = item.get("priority", 100)
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not math.isfinite(priority):
            errors.append(f"priority_must_be_finite_number:{ident}")
        if item.get("status") == "PASS" and (not isinstance(item.get("evidence"), list) or not item["evidence"]):
            errors.append(f"pass_without_evidence:{ident}")
        if item.get("status") == "NOT_APPLICABLE" and not (item.get("evidence") or item.get("reason")):
            errors.append(f"not_applicable_without_reason_or_evidence:{ident}")
        if item.get("status") == "BLOCKED_EXTERNAL" and not item.get("blocked_by"):
            errors.append(f"blocker_not_named:{ident}")
        if item.get("status") == "HUMAN_REQUIRED":
            if not str(item.get("owner", "")).startswith("HUMAN_"):
                errors.append(f"human_task_without_human_owner:{ident}")
            for field in ("automation_blocker", "procedure", "expected_return", "pass_criteria"):
                if not str(item.get(field, "")).strip():
                    errors.append(f"human_task_missing_{field}:{ident}")
            if item.get("owner") == "HUMAN_DECISION" and not item.get("quantified"):
                errors.append(f"human_decision_not_quantified:{ident}")
    graph = {ident: ([dep for dep in item.get("depends_on", []) if isinstance(dep, str)]
                       if isinstance(item.get("depends_on", []), list) else [])
             for ident, item in by_id.items()}
    for ident, dependencies in graph.items():
        for dep in dependencies:
            if not isinstance(dep, str) or not dep:
                errors.append(f"invalid_dependency_id:{ident}")
            elif dep not in by_id:
                errors.append(f"unknown_dependency:{ident}:{dep}")
    active, done = set(), set()

    def visit(ident: str) -> None:
        if ident in active:
            errors.append(f"dependency_cycle:{ident}")
            return
        if ident in done:
            return
        active.add(ident)
        for dep in graph.get(ident, []):
            if dep in graph:
                visit(dep)
        active.remove(ident)
        done.add(ident)

    for ident in graph:
        visit(ident)
    return errors


def decide_frontier(work: list[dict], clean_passes: int = 0,
                    state_fingerprint: str | None = None,
                    registry_fingerprint: str | None = None) -> dict:
    """Compute eligible work and the deterministic continue/handoff/stop decision."""
    if not isinstance(work, list):
        return _decision("INVALID_STATE", ["work_must_be_a_list"])
    if isinstance(clean_passes, bool) or not isinstance(clean_passes, int) or clean_passes < 0:
        return _decision("INVALID_STATE", ["clean_passes_must_be_nonnegative_integer"])
    if state_fingerprint is not None and not isinstance(state_fingerprint, str):
        return _decision("INVALID_STATE", ["state_fingerprint_must_be_string"])
    if registry_fingerprint is not None and not isinstance(registry_fingerprint, str):
        return _decision("INVALID_STATE", ["registry_fingerprint_must_be_string"])
    errors = _validate_work(work)
    if errors:
        return _decision("INVALID_STATE", errors)
    by_id = {item["id"]: item for item in work}
    eligible, blocked, human, done = [], [], [], []
    for item in work:
        status = item["status"]
        if status in TERMINAL_WORK:
            done.append(item["id"])
            continue
        if status == "HUMAN_REQUIRED":
            human.append(item)
            continue
        deps = [dep for dep in item["depends_on"] if by_id[dep]["status"] not in TERMINAL_WORK]
        if status == "BLOCKED_EXTERNAL" or deps:
            blocked.append({"id": item["id"], "blocked_by": item.get("blocked_by") or deps})
        elif item["owner"] == "AUTONOMOUS" and status in {"NOT_RUN", "FAIL"}:
            eligible.append(item)
        elif item["owner"] == "AUTONOMOUS" and status == "RUNNING" and item.get("lease_expired") is True:
            eligible.append(item)
        else:
            blocked.append({"id": item["id"], "blocked_by": "owner_or_status_requires_external_action"})
    eligible.sort(key=lambda item: (item.get("priority", 100), item["id"]))
    frontier = {"autonomous": eligible, "blocked": blocked, "human": human, "done": done,
                "AUTONOMOUS_ACTION_AVAILABLE": "YES" if eligible else "NO"}
    if eligible:
        action, reasons = "CONTINUE_FRONTIER", ["authorized_autonomous_work_remains"]
    elif human:
        clean = (clean_passes >= 2 and bool(registry_fingerprint)
                 and state_fingerprint == registry_fingerprint)
        action, reasons = (("HUMAN_HANDOFF", ["two_clean_passes_for_current_registry"])
                           if clean else ("DISCOVERY_REQUIRED", ["human_handoff_requires_two_clean_passes"]))
    elif blocked:
        action, reasons = "WAIT_EXTERNAL", ["no_unblocked_autonomous_work"]
    elif len(done) == len(work):
        action, reasons = "STOP_COMPLETE", ["all_work_items_terminal_with_evidence"]
    else:
        action, reasons = "STOP_NOT_VERIFIED", ["no_actionable_work_but_incomplete_items_remain"]
    return _decision(action, reasons, frontier=frontier,
                     registry_fingerprint=registry_fingerprint)


def decide_integration(state: dict) -> dict:
    """Distinguish task merge from project release; do not fake completion state."""
    if not isinstance(state, dict):
        return _decision("INVALID_STATE", ["integration_state_must_be_object"])
    scope = state.get("scope", "task")
    if scope not in {"task", "release"}:
        return _decision("INVALID_STATE", ["integration_scope_must_be_task_or_release"])
    keys = ("contract_approved", "scope_clean", "review_pass", "review_current",
            "contract_current", "required_gates_pass", "requirements_pass", "source_clean")
    if scope == "release":
        keys += ("frontier_complete", "human_clear", "external_clear")
    missing = [key for key in keys if not isinstance(state.get(key), bool)]
    if missing:
        return _decision("INVALID_STATE", [f"missing_boolean:{key}" for key in missing])
    failed = [key for key in keys if state[key] is not True]
    if failed:
        return _decision("NOT_ELIGIBLE", [f"precondition_failed:{key}" for key in failed])
    return _decision("READY_TO_RELEASE" if scope == "release" else "READY_TO_MERGE",
                     ["all_deterministic_" + scope + "_preconditions_passed"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cycle", "frontier", "integration"))
    parser.add_argument("--state", required=True, help="JSON observed-state packet")
    parser.add_argument("--expect-action", help="fail if the deterministic action differs")
    args = parser.parse_args(argv)
    try:
        data = json.loads(Path(args.state).read_text(encoding="utf-8"))
        if args.mode == "cycle":
            result = decide_cycle(data)
        elif args.mode == "frontier":
            result = decide_frontier(data.get("work"), data.get("clean_passes", 0),
                                     data.get("state_fingerprint"), data.get("registry_fingerprint"))
        else:
            result = decide_integration(data)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["action"] == "INVALID_STATE":
            return 2
        if args.expect_action and result["action"] != args.expect_action:
            print(f"Expected {args.expect_action}, got {result['action']}", file=sys.stderr)
            return 3
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"action": "INVALID_STATE", "reasons": [str(exc)]}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

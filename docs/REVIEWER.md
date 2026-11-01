# Independent reviewer

The reviewer is a fresh, read-only Claude Opus 5.5 session at high effort. It shares no
conversation with the agent that wrote the change, but it is the same kind of model, so it
can share the same blind spots. Use a different vendor's reviewer when that matters.
A PASS counts only when the deterministic control plane accepts it, so a reviewer's words
never override a failed gate or stale evidence.

## Run a review

```bash
.venv-gpu\Scripts\python.exe scripts\review.py verification\contracts\TASK.json
```

Add `--gates-only` to run the checks without a model call; the verdict stays `NOT_VERIFIED`.
Each full run sends the diff and repository files to the model service and uses the Claude
account's allowance. The session can only read files and run `git status`, `diff`, `log` and
`show`. It is denied `assets/`, `outputs/`, the model weights and `~/.ssh`. These settings
are a permission boundary inside Claude Code, not an OS sandbox. Evidence is written to
`.ai-native/runs/<task-id>/latest.json`.
Set `AI_NATIVE_REVIEWER_MODEL` and `AI_NATIVE_REVIEWER_EFFORT` to change the model. A review
that ran on any other model is rejected.

The reviewer uses the `claude` command line's default profile, which signs in separately from
the desktop app. If a review reports an authentication error, run `claude auth login` in your
own terminal, then run the review again.

| Exit code | Meaning |
|---|---|
| 0 | `TASK_VERIFIED`: gates passed and the review matches this exact tree and contract |
| 2 | Rework needed, or a required gate failed |
| 3 | Blocked on a named external dependency |
| 4 | `NOT_VERIFIED`: no usable review |
| 5 | Invalid contract or controller failure |

## What makes a PASS invalid

The controller rejects a PASS when a required gate failed, the source changed during gates or
review, the reviewer returned the wrong source fingerprint or contract hash, or the reviewer
is the same command as the executor. Task verification is not merge authorization: run
`scripts/ai_native/governance.py eligible` on the committed tip with
`verification/requirements.json`. These are local process checks, not a security boundary, and
GitHub branch rules must be configured separately.

## Project review checklist

Inspect the actual diff and files, not the author's summary.

- **Safety:** any weakening of the consent attestation, explicit-content refusal, screening of
  every target frame and the reference image, disabled model downloads, the `AI-ALTERED` label
  or metadata, or a change to upstream FaceFusion safety code, is at least `REWORK`.
- **Fail closed:** a failed check must stop the job and publish no output. Look for silent CPU
  fallback, swallowed errors, partial output and temporary media left behind.
- **Evidence level:** a fake-worker test is not face inference, a CPU check is not GPU proof, and
  a short clip is not a full-length movie. Report untested levels as `NOT_VERIFIED`.
- **Test honesty:** look for weakened assertions and tests that prove less than their names claim.
- **Privacy:** no credentials, personal media or private paths in Git, logs or error messages.
- **Spending and downloads:** no new GPU rental or model download path without owner approval.
- **Docs:** status and benchmark claims must match what was measured.

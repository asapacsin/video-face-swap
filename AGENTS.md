# Project operating instructions

Use the existing checkout: cloud tasks are already isolated. Do not create a worktree
unless the user explicitly requests one. Preserve existing user files and history.

Planner: record a bounded milestone in verification/contracts/<task>.json, with acceptance,
exact checks, scope, a time limit and at most three correction attempts.
Executor: make the smallest testable change, using economical tools where possible.
Reviewer: a fresh read-only Claude Opus 5.5 high-effort session runs through scripts/review.py
and follows docs/REVIEWER.md, returning PASS, REWORK, BLOCKED or NOT_VERIFIED. The session
that wrote a change never reviews it. Run a review before claiming a task is complete or committing it as
reviewed; state plainly when no review ran. A PASS counts only with exit code 0, because
scripts/ai_native/control_plane.py owns that decision. Never edit a contract, gate, test
or the requirements in verification/requirements.json to obtain a PASS. Task PASS does
not authorize a merge; use scripts/ai_native/governance.py eligible on the committed tip.
Stop gate: a CPU check never establishes GPU inference. Project PASS requires
the user's complete end-to-end criteria, including labeling and retained audio.
Follow authorized independent work until the bounded milestone is complete or blocked.

Do not rent a GPU, download model weights or invoke a face-processing service without
the user's explicit approval of size/cost/reason. Total planned GPU budget is about
RMB 200. Never modify upstream safety checks, automatically replace system drivers,
or log personal media/credentials. No new remote without an exact user-supplied URL.

These instructions were authored for this project from the user's requested roles.
The owner's ai-native-engineering reviewer core is copied at fa223b5; see
scripts/ai_native/PROVENANCE.md. Each review uses the owner's Claude allowance and runs
only when invoked; no hook starts one automatically.

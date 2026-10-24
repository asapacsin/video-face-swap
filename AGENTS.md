# Project operating instructions

Use the existing checkout: cloud tasks are already isolated. Do not create a worktree
unless the user explicitly requests one. Preserve existing user files and history.

Planner: record a bounded milestone in docs/TASK_CONTRACT.json, with acceptance,
exact checks, scope, a time limit and at most three correction attempts.
Executor: make the smallest testable change, using economical tools where possible.
Reviewer: independently inspect changes and execute checks, then return PASS,
REWORK, BLOCKED or NOT_VERIFIED with actual evidence and scope.
Stop gate: a CPU check never establishes GPU inference. Project PASS requires
the user's complete end-to-end criteria, including labeling and retained audio.
Follow authorized independent work until the bounded milestone is complete or blocked.

Do not rent a GPU, download model weights or invoke a face-processing service without
the user's explicit approval of size/cost/reason. Total planned GPU budget is about
RMB 200. Never modify upstream safety checks, automatically replace system drivers,
or log personal media/credentials. No new remote without an exact user-supplied URL.

These instructions were authored for this project from the user's requested roles.
The available ai-native-engineering checkout was inspected at b9deb834ce9ff192bc0c9628442308bdaed8d426.
No license file was present; its scripts/templates/hooks were not copied.
No external reviewer CLI or paid model workflow is automatically configured.

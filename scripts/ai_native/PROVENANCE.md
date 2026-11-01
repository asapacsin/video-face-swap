# Provenance

Copied from the owner's repository
[asapacsin/ai-native-engineering](https://github.com/asapacsin/ai-native-engineering) at
`fa223b52d1e91ea2526409b2dfd3cac211147f35`: `ai_native.py`, `control_plane.py`,
`process_security.py` and `governance.py`. `claude_bridge.py` is new to this project, written
in the shape of the upstream `codex_bridge.py`, so the reviewer runs as Claude Opus 5.5.

One change was made to the copied files, and it should be proposed upstream: in `ai_native.py`,
the fixed 240-second reviewer limit is now `budgets.reviewer_timeout_seconds` (default 240),
which is passed to the reviewer process as `AI_NATIVE_REVIEWER_TIMEOUT`.

Not copied: the Stop and keep-going hooks, because each automatic stop could start a billed review
and they need a work registry; the pre-tool guard, supervisor and frontier, which need a host
runner; and `discover_work.py`, `delivery_gate.py`, `capabilities.py` and the Android and Aurel
adapters, which this review flow does not use.

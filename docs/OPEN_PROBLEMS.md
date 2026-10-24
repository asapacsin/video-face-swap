# Outstanding work

- M2: implement FaceFusion subprocess adapter, both-input safety screening, visible
  labeling, metadata, audio retention, output verification and private temp cleanup.
- Review full OpenRAIL-AS terms and each exact model's rights; prepare download sizes,
  origins/checksums and purpose, then obtain user approval before downloading weights.
- Official hosted FaceFusion installation docs returned HTTP 403; inspect them when
  access becomes available. Pinned upstream installer/CI were inspected instead.
- Python package and CUDA installer execution on the remote host remain unverified.
  CUDA kernel check is supplied, but no GPU is available in Codex Cloud.
- User supplies provider/SSH access securely, then approves exact rental setup and cost.
  Do not request secret values in chat or put them in Git.
- Need an authorized adult reference face and short licensed/synthetic non-explicit
  face clip with audio. Test color frames are only media-inspection fixtures, not face tests.
- ai-native-engineering at b9deb83 has no license file. Its workflow conventions were
  inspected; original project-specific instructions are supplied. Importing tooling,
  CLI adapters and hooks requires clarified reuse terms and project-specific integration.
- No project remote configured; no source has been pushed. Cloud snapshot publication
  is separate from this local preparation and has not been verified in a fresh task.

Highest-value next action: complete the CPU-testable M2 safety/output adapter and
model/license/download manifest before considering a GPU rental.

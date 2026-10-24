# Outstanding work

- The CPU pipeline and worker orchestration are implemented and covered by synthetic
  fake-worker FFmpeg tests. Real FaceFusion execution, models, classifier behavior on
  real face media, and remote GPU behavior remain NOT_VERIFIED.
- Review full OpenRAIL-AS terms and each exact model's rights; prepare download sizes,
  origins/checksums and purpose, then obtain user approval before downloading weights.
- Model manifest records source-derived metadata, but exact byte sizes and SHA-256
  values remain unknown because metadata access was blocked; resolve before download.
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
- Project remote is configured and prior commits are pushed. Latest M2 CPU changes
  remain under independent review. Cloud snapshot restoration in a fresh task remains
  unverified.

Highest-value next action: clear CPU reviewer findings and verify model rights/sizes
before preparing a GPU approval packet. Actual GPU inference remains blocked on GPU,
model weights, authorized test media and spending approval.

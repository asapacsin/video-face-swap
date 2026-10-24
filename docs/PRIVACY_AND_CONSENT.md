# Privacy and consent

Only consensual adult, non-explicit footage is in scope. The operator must affirm
that every depicted individual whose identity is used is an adult and consented to
this specific use. Rights to the clip/reference and relevant model licenses must
also permit processing. No celebrity impersonation, harassment, deceptive presentation,
minors or sexual deepfakes. A command-line attestation records an assertion, not proof.

Known explicit content is refused. Dry-run does not classify media or approve its
content. Actual processing must retain upstream content checks and fail if they cannot
run. The finished pipeline must visibly label every output as AI-altered and include
metadata where feasible. M1 generates no altered videos or sample face identities.

No analytics, third-party face services or uploads are built into the CLI. FFprobe
is restricted to local file/pipe protocols. Avoid untrusted media; media parsers have
their own security boundary. Manual sync to a GPU host is an operator-controlled upload:
approve the host, access permissions and deletion plan first. Do not sync private inputs
with code by default. Use private directories (umask 077), encrypted SSH, and remove
inputs, outputs and temporary media from rented storage when no longer needed.
Do not retain personal filenames, frames, embeddings, credentials or raw stderr in logs.
Keep inputs, weights, credentials and generated media out of Git. Gitignore is a guard,
not a substitute for inspecting staged files. Storage deletion is not guaranteed forensic erasure.

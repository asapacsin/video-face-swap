# Face Swap App — product spec

Status date: 2026-11-02. This spec defines the product and its delivery order. It replaces the
delivery sequence in [PROJECT_PLAN.md](PROJECT_PLAN.md). Safety rules remain as in
[PRIVACY_AND_CONSENT.md](PRIVACY_AND_CONSENT.md).

## Goal

A Windows desktop app where the owner picks a movie and a photo, clicks once, and later gets a
720p MP4 of that movie with the photo's face on one chosen character, **rendered properly** as
defined in [Rendering quality](#rendering-quality). A run that finishes but fails those checks is a
failed run.

**Build rule:** reuse FaceFusion, FFmpeg, existing libraries and community practice wherever they
fit. Write custom code only when nothing existing does the job or the safety rules require it.

## Where work runs

| Phase | GPU | Purpose |
|---|---|---|
| **1. Build and prove (now)** | Rented server, reached through the `GPU_HOST` SSH alias | All GPU development, testing and benchmarks |
| **2. Move to the laptop (later)** | Owner's RTX 4050 Laptop: 6 GB VRAM, i5-12450H, 32 GB RAM | Final movie runs |

- **The local GPU is not used during phase 1.** Work on the laptop is limited to editing code, CPU
  tests with fake workers, and checking the window. No local face inference or GPU benchmark runs.
- **Phase 2 starts only when the owner asks.**
- **Cloud results do not prove the 4050 works.** The server has 32 GB VRAM, CUDA 13 and Linux, while
  the laptop has 6 GB, CUDA 12 and Windows. Phase 2 must repeat the stability and speed tests.

## Cloud rules

- **Budget:** about RMB 200 in total. The current server is estimated at ¥3.26/hour, about 60 hours;
  the owner confirms the rate in the console.
- **Billing:** only the owner can rent, stop or release a server. Each session reports when GPU work
  ends so the owner can stop it. Report spending against the budget at each milestone.
- **Long runs:** jobs run in the background on the server, so the laptop can be closed.
- **Media:** tests use FaceFusion's sample media by default. The owner's movie or photo is uploaded
  only after the owner approves it, and is deleted from the server when the test ends.
- **Credentials:** the SSH key stays on the owner's PC, and no secret enters chat, Git or logs.
- **Server loss:** a stopped or released server may lose its files, so setup must be repeatable from
  a script and results are downloaded before it stops.

## User flow in the finished app

1. Double-click `FaceSwap.bat`.
2. Choose the movie and the photo.
3. Tick the consent confirmation.
4. Press **Scan**. The app converts the movie, screens it, and shows the character it chose.
5. Confirm that character, or choose another.
6. Press **Start**. The laptop can run unattended, overnight if needed.
7. Open the finished MP4 from the app. If the job was interrupted, reopen the app and resume it.

During phase 1, the same pipeline runs on the server from the command line.

## Functional requirements

Status means: **Tested** has real evidence on short clips, **Partial** is incomplete, and
**Missing** is not built. Evidence comes from the rented server unless noted; none of it is from the
current code on the 4050.

| ID | Requirement | Status |
|---|---|---|
| F1 | Choose the movie, photo and save location in a window | Partial: built, file pickers not clicked by hand |
| F2 | Accept any readable movie file and convert it to a constant-rate H.264/AAC MP4 at 720p, never upscaling | Tested on MKV and variable-rate clips |
| F3 | Default to the face seen most often; allow a time to choose another character | Default tested on real clips; the time option has CPU tests with fake workers only |
| F4 | Show the chosen character and require confirmation before the long swap starts | Tested in the window with a fake job; not clicked by hand with a real job |
| F5 | Replace only faces matching the chosen character | Partial: checked by eye on sampled frames of one two-person clip, choosing each person; not frame by frame |
| F6 | Keep the first audio track and the original timing | Tested on clips up to 10 minutes; joins with audio tested over 2 segments only, so drift across a full film is unmeasured |
| F7 | Show stage, percent, elapsed time and estimated time left | Tested with fake workers; the bar moves per finished segment, so it stays still during the scan and each segment; time left appears in the log |
| F8 | Cancel a job, keeping progress or removing it | Tested with a fake job: removal on cancel and on closing the window. Keeping progress is offered in both but not tested in the window |
| F9 | Save progress and resume after a crash, power loss or cancel | Tested for segments: a real job killed after segment 1 of 2 resumed. Scan-checkpoint resume has CPU unit tests only; power loss not tested |
| F10 | Refuse to start without enough free disk space | Comparison tested with fake workers; the estimate comes from a CPU encode of the film clip, not a measured job |
| F11 | Show a specific error when a job stops | Partial: real pipeline failures not all exercised |

**F9 design:** process the 720p movie in segments of about 5 minutes using FaceFusion's frame-trim
options. Record screening and swap progress per segment, then join the segments at the end. The
reference frame stays numbered against the whole movie, so the character stays the same across
segments. Resuming reuses finished segments only if the movie, photo, character and app version
are unchanged. A segment refused by screening stays refused.

## Rendering quality

"Rendered properly" means every check below passes on the output. Thresholds marked *proposed* are
calibrated on the 15-second film clip first, and the owner confirms them before they become pass
marks.

**File and timing (automatic, every run):**

| ID | Requirement | How it is checked |
|---|---|---|
| R1 | The MP4 decodes from start to end with no errors | FFmpeg full decode with `-xerror` (already built) |
| R2 | Same frame count, frame rate and 720p size as the converted movie, never upscaled | FFprobe comparison (already built) |
| R3 | Audio stays in sync: duration within one frame of the source, and the audio and video start together | FFprobe stream start times and durations; today only duration is checked, within 0.25 s |
| R4 | The `AI-ALTERED` label is visible on every frame and in the metadata | Label drawn by FFmpeg (built); a check that reads the label region on sampled frames is missing |

**Face replacement (automatic, using FaceFusion's own face detector and recogniser):**

| ID | Requirement | How it is checked |
|---|---|---|
| R5 | Coverage: in frames where the chosen character is visible, the face is replaced | Share of those frames whose face now matches the photo. *Proposed:* at least 95% |
| R6 | Likeness: replaced faces resemble the photo | Recogniser distance between each replaced face and the photo, below FaceFusion's own matching limit (0.3) on at least 90% of replaced frames (*proposed*) |
| R7 | Nobody else changes | Faces of other people match the source frame. *Proposed:* no other person's face matches the photo in any sampled frame |
| R8 | Background untouched | Outside the replaced face boxes, sampled frames differ from the source only by encoding noise |

**Look (owner's judgement on a contact sheet and short clips):**

| ID | Requirement | How it is checked |
|---|---|---|
| R9 | No visible seams, boxes, colour patches or blurred blocks around the face | The app writes a contact sheet of frames every 30 seconds, plus three 10-second clips with the character on screen, and the owner rates them |
| R10 | No flicker or jumping between the photo's face and the original in a continuous shot | Owner watches the 10-second clips; R5's per-frame results show any gaps |
| R11 | Hard cases are reported, not hidden: side profiles, partly covered faces, dark scenes and small faces | The report lists frames where R5 or R6 failed, with times, so the owner can see where it struggles |

**Render report:** each finished job writes a short report beside the result, listing R1–R8
results, the contact sheet and the hard-case times. The face checks reuse FaceFusion's detector and
recogniser and run on the GPU after the swap, on frames sampled every second (*proposed*), so they
add minutes, not hours. The report holds times and scores only, never face images beyond the
contact sheet the owner opens.

Not measured yet: none of R4–R11 has been checked on a real run. The 15-second film clip was
checked by eye on sampled frames only.

## Performance and stability

| ID | Requirement | Cloud (phase 1) | RTX 4050 (phase 2) |
|---|---|---|---|
| P1 | Run a whole job without a GPU driver crash | Met: the same CUDA error 700 appeared here with 4 screening threads; with serial screening, the 65-second clip ran twice and the 10-minute job ran clean | **Failing until retested:** 4-thread screening hit CUDA error 700 partway through (see BENCHMARKS.md); serial screening has run here only on a 10.8-second clip |
| P2 | Stay on the GPU; stop with an error if any model falls back to the CPU | Partial: a forced no-GPU run was refused at start with no output; a fallback during a job is NOT_VERIFIED | Partial: checked at start only |
| P3 | Record peak VRAM; on the 4050 it must stay within 6 GB | 6093 MiB on the film clip (see docs/BENCHMARKS.md); 8128 MiB before the screening models were freed | 4.2–5.4 GB on short clips |
| P4 | Finish a 2-hour, 24 fps movie, with the overall rate recorded | Not run, by owner decision. Measured 2.3–3.6 frames/s on clips of 65 seconds and 10 minutes | Within 24 hours, at least 2 frames/s overall |
| P5 | Pause when the GPU reaches 85°C and continue once cooled | Not required | Missing |
| P6 | Stop the laptop sleeping during a job | Not applicable | Built, not tested over a long run |

P4's 24-hour limit and P5's 85°C are proposed targets. See [BENCHMARKS.md](BENCHMARKS.md) for the
measurements. What sets the remaining 6093 MiB peak was not measured, so phase 2 must test what the 4050 needs.

## Safety requirements

These stay in force everywhere and cannot be turned off.

- Consent is confirmed for each job.
- The photo and every movie frame are screened using FaceFusion's unchanged safety code. One
  refused frame stops the job, and some movies will therefore not process.
- If screening cannot run, the job does not run.
- Every output has a visible `AI-ALTERED` label and metadata.
- Nothing is overwritten. A failed check publishes no output.
- A refusal is final: its media is removed, and every-frame screening has CPU unit tests. No real
  frame has been refused in a recorded run.
- No analytics or online face services. In phase 2, media stays on the laptop.
- Logs and errors hold no credentials or personal media.
- Some models are non-commercial or research-licensed, so output is for personal use only until
  the full license terms are reviewed.

## Out of scope

Resolutions above 720p, more than one replaced character, more than one photo per job, batch jobs,
web or mobile versions, live video, and model training. A window that controls the cloud server is
also out of scope: the window drives local jobs, and the server is used from the command line.

## Milestones

Each milestone ends with its tests and an independent review through `scripts/review.py`.

**Phase 1: rented GPU**

| # | Deliverable | Done when |
|---|---|---|
| C0 | Repeatable server setup: Python 3.12, FFmpeg with a label font, CUDA 13 packages and models | `scripts/setup_cloud.sh` succeeds, a real GPU convolution passes, and the sample clip swaps end to end |
| C1 | Stable long runs (P1, P2) | The 65-second clip finishes twice with no driver error, and CPU fallback during a job stops it |
| C2 | Confirm the character (F3, F4, F5) | A clip with two different people replaces only the chosen one, checked frame by frame |
| C3 | Resume (F9) | A job killed mid-swap resumes and gives a valid result with matching frame count and audio |
| C4 | Long-job features (F7, F10, F11) | Estimated time, disk check and specific errors are each tested; the window shows them, tested with a fake worker |
| C5 | Real-film proof (P3) | A 15-second clip of a real film, with several people and a scene change, finishes with speed, VRAM and cost recorded. By owner decision, phase 1 does not run a full movie; the full-length proof is phase 2's final acceptance |
| C6 | Render check (R1–R11) | The render report is built from FaceFusion's detector and recogniser, the thresholds are calibrated on the 15-second film clip and confirmed by the owner, and that clip passes R1–R11 |

Status on 2026-11-01. Evidence is in place for C1–C5, but none has passed independent review, so no
milestone is complete:

- **C0, partly met:** the real GPU convolution runs when a job starts, not in the script, and FFmpeg is
  a moving build. In an empty folder, the script's packages installed and it finished with
  exit 0, and the film clip swapped end to end. The model download did not complete: GitHub gave
  about 3 MB/minute from this server, so the verified models were copied in from the server's
  existing folder. Downloads now resume after a timeout.
- **C1, partly met:** the 65-second clip finished twice with no driver error. CPU fallback was
  refused at start, but stopping on a fallback during a job is NOT_VERIFIED.
- **C2, partly met:** sampled frames were checked by eye, not frame by frame.
- **C3, evidence recorded:** see F9 above.
- **C5, evidence recorded:** the clip has a scene change and the scan found at least 3 candidate
  characters. See [BENCHMARKS.md](BENCHMARKS.md).
- **C4, evidence recorded with fake workers:** the window's Scan, confirm and Start flow ran on a real
  window with a fake job, not by hand.

**Phase 2: RTX 4050, only when the owner asks**

| # | Deliverable | Done when |
|---|---|---|
| L1 | Laptop stability and limits (P1–P3, P5) | The fixes run on Windows, the 65-second clip finishes twice within 6 GB, and the temperature pause is tested |
| L2 | Final acceptance | All items below pass |

## Final acceptance

The project passes only when all of these have recorded evidence on the RTX 4050:

1. A full-length movie, which the owner is allowed to use, runs from the window to a finished 720p
   MP4 within the P4 limit.
2. The result opens in a normal player, with audio in sync and the label visible.
3. The movie passes every rendering-quality check, R1–R11, and the owner accepts its render report.
4. A job interrupted partway resumes and finishes correctly.
5. All tests pass, and the review returns exit code 0 on the exact commit.

Cloud results, short clips and fake-worker tests do not count as evidence for items 1–4.

## Open items for the owner

- **Server:** confirm the hourly rate, and whether stopping keeps its files and still charges for disk.
- **Limits:** confirm the 24-hour limit and 85°C pause before phase 2.
- **Test movie:** for L2, provide a full-length, non-explicit movie you are allowed to use, with a
  photo of someone who has agreed.
- **Render thresholds:** confirm the proposed pass marks in R5–R7 once they are calibrated.

# Complete assignment for the user's Claude execution sessions

You are the implementation and execution worker for PhaseDx. Work from this repository's `docs/execution` queue. The user wants all useful ready work advanced in parallel and has abandoned automatic Codex-to-Claude hourly dispatch. Do not create another linked supervisor or wait for a new mini-prompt between task cards.

## Responsibilities

Implement, prepare inputs, execute the specified experiments, produce machine outputs, recover source/config facts, and mechanically render scientifically approved material. The user and Codex retain scientific design, interpretation, independent verification, manuscript writing and final approval. Do not turn your own passing tests into independent scientific verification. Do not select new endpoints, cohorts, model arms, thresholds or claims because of observed results.

Read once: `README.md`, `AUDIT.md`, `INPUTS.md`, `DATASETS.md`, `contracts/EXTERNAL_PROTOCOL_20261014.md`, and your ready task cards. Use the concise queue thereafter. The two copied scientific contracts are historical decisions, not new data-access authorization. Their expired cycle rules do not prevent independent code/source work; each real-data task uses the current execution scope.

## First actions

1. Inspect Git status and the queue. Preserve other people's changes and completed outputs. Check for an already active owned executor/job before launching duplicates.
2. Resolve local roots as described in `INPUTS.md`. The October implementation is not yet a portable public package: E01 imports selected source with provenance. Do not substitute the older `pipeline/run_all.py` workflow for the current study.
3. Start all independent ready tasks that fit the host. E06 may prepare its correction in a separate scratch path while E01 imports the original source. Merge the correction under E07; do not race writes to the same module.
4. Give each deterministic process a task/attempt/output directory. Backfill immediately when a slot becomes available. Only integration waits for its dependencies.

## Parallel execution and resources

Initial guidance is four lightweight one-thread CPU jobs, **not a fixed cap**. Increase concurrency from actual memory, CPU, free-space and swap-out measurements. Separate parsing/network/CPU/model-inference lanes. Do not run 100 processes just to match a task count, and never report queue length as measured concurrency. Record child start/end overlap, peak RSS/device use and failed attempts. Share the Mac and account with other projects; do not stop their jobs.

Use the user's declared session window and stop/checkpoint at its deadline. If no window or actual HPC resource allocation has been supplied, do not invent one. E05 prepares a Slurm option; it is not an HPC submission. Battery percentage is informational; do not change OS sleep, VPN or security settings. Normal Claude sandbox and approval controls remain intact.

Actual service exhaustion is a wait/resume condition within an authorized session, not successful completion. Preserve state and the same conversation where possible. Observe actual availability before resuming; do not use paid overage, reset redemption or account/provider switches unless separately instructed. Do not promise that the service will run a monitor while unavailable. This repository handoff does not create a background service.

## Study rules that prevent wasted work

- Reuse reviewed MRI/breast/internal CT results and the fixed completed simulations. Do not rerun a suite to make a new package or fill runtime.
- Make the three acquisition corrections in E06 once. Any later repair must address an observed execution-blocking or result-changing problem, with a narrow regression and preserved failure. Do not grow a general validator project.
- Preserve the full CQ500 source roster, technical failures, clinical unknowns and missing scores. Keep selection and prediction blind to reference outcomes.
- The frozen external primary is seed42 maximum-slice image logit versus **transported physical-rank reconstructed frozen G2**. It is not original-G-star equivalence. Seed123 is descriptive. Follow the copied scientific contract exactly.
- The future primary clinical bootstrap is 2000 paired draws, seed 2026101327, linear percentiles, with undefined draws retained. No fresh draws during implementation tests, no discarded unfavorable runs, no outcome-driven extensions.
- Historical instructions such as “run everything,” old architecture scripts, or outcome-enriched download suggestions do not authorize a new scientific design. The task cards identify the actual remaining work.
- If a clinical source fact is unknown, preserve it as unknown and work on independent tasks. Do not invent clinical expertise, pathology confirmation, patient linkage or ethics approval.

## Data and publication boundary

This repository is public. Never commit raw images, restricted derived tables, case/subject identifiers, individual reference labels or scores, private download links, keys, credentials or unapproved checkpoints. Register sources and permitted aggregate results instead. A public source URL is not permission to redistribute every downloaded artifact. Dataset-specific agreements still apply.

Existing Research storage/download approvals are recorded in the task cards. Do not repeatedly ask the user to choose a disk. E07/H02 resolve the concrete image-only operation and trustworthy cumulative baseline. Image acquisition and reference-label access are separate stages. Source `prediction_probabilities.csv` is excluded entirely. No agreements, messages, submissions, deposits or external account changes are performed merely because they appear in a task card.

## Commands and completion

The queue utility exists now. Clinical CLI commands in `CLI_CONTRACT.md` are **required interfaces to implement**, not a claim they already exist. Verify source entry points before running them. Do not paste guessed shell flags into a real data command.

After each meaningful deliverable, update its entry in `tasks.json` with status and evidence references and run `render`. Use `doing`, `blocked`, `done`, `awaiting_scientific_review` or `optional` as appropriate. Execution can be done while scientific acceptance remains pending; state that explicitly. Do not mark H tasks done yourself.

Return a compact report organized by **new study evidence and completed deliverables**, with actual commands, output identities, missing inputs and failures. Report measured timing and overlap separately from elapsed session length. Commit code and safe documentation in coherent units. Follow the user's GitHub publishing authorization for that session; this assignment is not permission to expose private data. No hundred-file victory lists, optimistic completion percentages or fabricated overnight compute claims.

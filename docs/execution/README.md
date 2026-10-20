# PhaseDx: execution backlog to a defensible submission

**Current checkpoint: October 20, 2026.** Automated Codex–Claude dispatch is ended. The user has authorized Codex to execute dependency-ready CPU work; the same repository also supports independently launched Claude sessions. There are no collaborator compute assignments. See [current evidence](CPU_EVIDENCE_20261020.json) and [progress](CPU_PROGRESS_20261020.md).

There are **50 execution task cards** and **15 scientific/author task cards**. Five execution cards are conditional alternatives or repairs, and one scientific card concerns the optional historical screen. These are work packages with real outputs, not a promise of 65 simultaneous workers. Each card now distinguishes completed implementation/evidence assembly from pending clinical execution. Card counts are not a measure of the paper’s scientific completeness.

Start with [the complete executor assignment](CLAUDE_EXECUTOR.md). Use [the task index](TASK_INDEX.md), [machine-readable queue](tasks.json), [dataset registry](DATASETS.md), [audit](AUDIT.md), [input map](INPUTS.md) and [CLI contract](CLI_CONTRACT.md). Every task has inputs, owned deliverables, execution steps, dependencies, resource profile and acceptance evidence.

## What moves the study forward

The next major empirical result is the **frozen external CT image-versus-transported-geometry comparison**. Its critical path is:

1. Portable implementation and targeted acquisition repairs are complete; B31 already acquired491 image archives and extracted490, with one source-corrupt archive retained in the denominator.
2. Finish the lossless-inclusive header inventory E10 once Research is readable. The repaired reader checkpoints I/O outages rather than interpreting them as clinical exclusions.
3. Resolve clinical series compatibility and source/reference mapping E11–E15, H03–H05.
4. Freeze actual model/input identities E16–E22; execute independent image and geometry work E23–E25.
5. Produce the specified primary, descriptive secondary and missingness results E26–E29; independently verify and interpret them H07–H09.
6. Integrate the evidence in the paper H10, then mechanically build and review submission files E38–E44, H14.

**Run independent work while this path progresses:** MRI/breast/CT Methods recovery, retained-cohort chronology, accepted result assembly, existing simulation reproducibility, primary-source extraction and reporting evidence. They are not held for another master prompt. A blocked clinical decision blocks its descendants, not the entire queue.

## Work that is already done

The two fixed synthetic experiments, selected retained MRI/breast/CT uncertainty checks, and v14 reading PDF exist. Reuse them. Do not burn a night regenerating accepted draws, complete simulation matrices, old architecture sweeps, repeated packages or generic validation suites. The detailed [completed-evidence table](AUDIT.md#completed-evidence-to-reuse) states what was actually checked.

## Running the queue

```sh
python3 tools/execution_backlog.py validate
python3 tools/execution_backlog.py ready
python3 tools/execution_backlog.py show E06
```

This small tool reads the queue; it does **not** launch compute, download data, call models or approve science. Update `tasks.json` with evidence and regenerate the index with `python3 tools/execution_backlog.py render` as work proceeds. `ready` means dependencies recorded complete, not proof that private inputs or permissions exist on a different machine. Check only the applicable task's inputs.

Read current card evidence before rerunning anything. The portable CPU environment, 54 synthetic/fixture tests, source-bound Methods, model/input registry and completed synthetic evidence are available. Fresh neural dependencies and strict loading of both original checkpoints on CPU also passed. The resumed header inventory is supervised separately. Clinical series qualification, reference-key interpretation, external execution and final author/release inputs remain explicit tasks. Do not repeat the completed test suite or outcome checks merely to fill a session.

## Definition of completion

“100%” means the chosen study is **ready for responsible submission**: the required evidence is executed and independently assessed; Methods and clinical limitations are honest; the manuscript, figures, applicable reporting checklist, permissions, disclosures and author approvals are complete. It does not mean a favorable external result or guaranteed journal acceptance. See [submission requirements](SUBMISSION.md).

The older `paper/tex/rsna` manuscript and the October revision are different evidence/scoping versions. Neither file existence nor the old checklist's “ready” wording establishes that the revised paper is ready. Preserve historical work, but use this handoff for new execution.

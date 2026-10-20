# Portable execution interfaces

Implemented October 20, 2026 in `src/phasedx_external`. These commands use explicit private inputs; they do not supply clinical qualification, authorize publication, or silently create a new study. Historical task-card command shapes are superseded by this file and actual `--help`.

## Install and smoke

```sh
python3.13 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements/cpu.lock.txt
.venv-cpu/bin/python -m pip install --no-deps .
.venv-cpu/bin/python -m phasedx_external --help
.venv-cpu/bin/python -m phasedx_external smoke --out /private/new-smoke
.venv-cpu/bin/python -m pytest tests/external -q
```

The smoke uses six fabricated cases and handwritten scores. It checks admission, signed paired point statistics, separate secondary support and missing-score bounds without images, neural inference, random draws or network access. Choose a new output directory. Install `requirements/inference.lock.txt` only where inference is needed; this is a CPU path, not a CUDA installation. See [environment observations](ENVIRONMENT.md) for tested versus untested installation scope.

## Registered queue

Real-data entry points require a prospective task receipt. Do not fabricate environment tokens to bypass it. Create a **private** jobs JSON with `id`, `command` (argument array), `cwd`, `script` (the entry module's actual path), optional `config`, `inputs`, `outputs`, `dependencies`, `memory_bytes`, `reserve_bytes`, and optional `exclusive` resource keys or `max_seconds`.

```sh
python tools/run_external.py --jobs /private/jobs.json \
  --run-root /private/new-run --deadline-utc 2026-10-20T12:30:00Z --workers 4
```

Use the deadline of the actually authorized session; the example date is not an extension. `--dry-run` reports the plan without admission. Paths in jobs must resolve on the executing machine. The queue writes unique attempt logs and append-only register/start/end receipts, records failures, backfills ready dependencies, and checkpoints only its positively identified children on STOP/deadline/resource exhaustion. Put `STOP_REQUESTED` in the run root to stop admission. All worker numerical thread variables are fixed to one. Four workers is a configurable starting point, not a claim about achieved overlap.

A task can be resumed when its source/config/input identity and prior successful output hashes match. Failed attempts remain visible. Exit zero is process completion, not scientific acceptance; inspect each task's domain status. A disk-disconnected scan must never turn into a clinically excluded population.

For `python -m phasedx_external manifest`, register `src/phasedx_external/build_manifest28.py` as the script. For `select`, the guarded implementation is also `build_manifest28.py`. `lock`, `receipt`, `reference` and `score` register `commands.py`. Image/geometry entry points register their respective modules below. These CLI entries declare no config to the entry guard: put every design/config file in the job's `inputs`, omit `config`. The separate header-shard tool declares its plan as `config`.

## Actual commands

Uppercase paths are placeholders, not provided clinical data.

| Stage | Arguments after `python -m phasedx_external` |
|---|---|
| Read-only acquisition plan | `acquire --operation OP.json --dry-run` |
| Image-only transfer | `acquire --operation OP.json --deadline-utc DEADLINE` |
| Source-bound extraction | `extract --operation OP.json --deadline-utc DEADLINE` |
| Full roster and technical inventory | `manifest --roster ROSTER.csv --root IMAGES --policy POLICY.json --out MANIFEST [--header-cache CACHE.json]` |
| Apply documented blinded clinical review | `select --manifest MANIFEST --review REVIEW.csv --images IMAGES --out SELECTED` |
| Bind the approved design to actual selected components | `lock --design DESIGN.json --manifest SELECTED --out EXECUTION.json` |
| Primary image predictions | `predict-image --design-lock DESIGN.json --execution-lock EXECUTION.json --role primary_image --manifest-dir SELECTED --checkpoint-dir MODEL42 --cycle-control CONTROL.json --out RUN42` |
| Descriptive image predictions | Same image command, with `--role secondary_image`, its own checkpoint directory and output |
| Transported frozen geometry | `predict-geometry --design-lock DESIGN.json --execution-lock EXECUTION.json --manifest-dir SELECTED --model G2.pkl --out RUNG2` |
| Freeze finished prediction runs | `receipt --execution-lock EXECUTION.json --manifest SELECTED --runs RUNS.json --out RECEIPT.json` |
| Reference conversion after prediction admission | `reference --design DESIGN.json --execution-lock EXECUTION.json --receipt RECEIPT.json --runs RUNS.json --manifest SELECTED --reads READS.csv --schema SCHEMA.json --out REFERENCE` |
| Primary and descriptive scoring | `score --design DESIGN.json --execution-lock EXECUTION.json --receipt RECEIPT.json --runs RUNS.json --manifest SELECTED --reference READS.csv --schema SCHEMA.json --out RESULT.json [--grouping GROUPS.json]` |
| Private flow and completion bounds | `flow --result RESULT.json --out FLOW.json` |
| Allowlisted aggregate candidate | `export --result RESULT.json --out CANDIDATE` |

`score --reference` consumes the authenticated **original three-reader CSV**, not the transformed two-column output from `reference`. It performs its own admission before opening reader rows. `RUNS.json` maps each role to `[run_directory, predictions_csv]`. An image checkpoint directory contains `best_model.pt`. Failed/missing predictions stay in the full intended population; they are not silently removed from denominator accounting.

The clinical scorer fixes 2,000 paired draws, seed 2026101327 and linear percentiles through the approved design. It has no runtime endpoint/resample tuning option. It retains undefined draws explicitly and reports the secondary seed on its own support. Completion bounds describe possible values in the finite intended eligible labeled cohort; they are not sampling confidence intervals.

## Decoder and source handling

Default image policy is `uncompressed_v1`. The explicit `--decoder-policy lossless_sv1_pylibjpeg_v1` extension admits JPEG Lossless Process 14 SV1 only, alongside the previously supported uncompressed syntaxes; it rejects declared lossy content. The design and manifest must name the same policy. The design also binds decoder and adapter source hashes and installed decoder package versions. Changing the decoder policy requires an outcome-blind design amendment before predictions; it does not refit the model or change the canonical HU/window/float16 preprocessing path.

`tools/check_lossless_samples.py --plan PLAN.json --out OUT` implements a fixed, registered diagnostic comparing two independent decoders. Do not expand its sample based on outcomes. `tools/scan_header_shard.py --plan PLAN.json --out OUT` reads each planned member's bytes for identity and parses headers without decoding pixels. I/O errors exit nonzero and leave the source pending. Authenticated non-DICOM files are retained as parse failures. Complete shards can be bound into a cache index containing plan/summary/record paths and SHA-256 values, image root and expected file count. The manifest rejects partial, changed or identity-failed caches. Header-cache reuse authenticates the earlier scan; it does not claim the drive's current bytes are unchanged. Prediction admission still verifies selected source bytes.

## Privacy and release

Keep source IDs, image/label rows, per-case predictions, person mappings, model checkpoints, private machine paths and full logs outside Git. `flow` remains private because it can contain source keys. `export` creates an aggregate candidate, not a publication authorization. No code execution substitutes for clinical compatibility, target interpretation, ethics or author approval.

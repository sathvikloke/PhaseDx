# Inputs, private roots and source handoff

## Current repository versus local artifacts

The older `pipeline/`, `trivialbaselines/` and August `paper/` tree are historical. The portable external implementation now lives in `src/phasedx_external/`; the completed synthetic studies are under `reproducibility/synthetic_b25/`. Private images, checkpoints and individual scores remain outside the public handoff.

Configure these roots locally, outside Git:

| Variable | Meaning / discovery hint |
|---|---|
| `PHASEDX_REVISION_ROOT` | Existing `nine_hour_20261014T022606Z` directory under the local Codex project's `work/`; holds B25–B30 and v14 |
| `PHASEDX_WORK_ROOT` | Parent historical `work/` containing earlier October cycle directories |
| `PHASEDX_LEGACY_ROOT` | Existing PhaseDx checkout with `pipeline_out/` and local `revamp/`; do not assume it is the new checkout |
| `PHASEDX_DATA_ROOT` | Authorized local fastMRI/source archive root, historically Research's `fastmridatasets`; verify actual mount/location |
| `PHASEDX_EXTERNAL_ROOT` | Already chosen Research project area `PhaseDx_revamp_data`; dataset folders remain private |
| `PHASEDX_RUN_ROOT` | New, non-overwriting execution output directory for the user's current session |

No private absolute machine paths are required in public code. The user supplies these roots to their authorized executor; a remote/cloud worker that cannot see them must obtain a permitted source-only handoff or authorized private inputs, not fabricate their contents. Do not upload restricted inputs to make a cloud worker convenient.

`source_map.json` records 56 specific source/document locators with content hashes from this audit. Resolve each relative path under `PHASEDX_REVISION_ROOT`. A file can be scientifically limited even if its hash matches. Do not upload everything in this map: some paths locate private configuration/source lists.

Earlier evidence discovery anchors:

- `nine_hour_20261012T024929Z`: B19–B23, reviewed MRI table/manuscript inputs and original CT uncertainty recovery.
- `nine_hour_20261011T155442Z`: B14–B18 and reviewed breast uncertainty table.
- `nine_hour_20261012T131057Z`: B24 and earlier source/methods extraction.
- Local `revamp/batches/`: original MRI/CT implementation, source locks and retained scores/checkpoints.

E02 reads existing manifests first to resolve exact files; these anchors are not license to guess a checkpoint or blindly rescan every image.

## Selected implementation starting points

| Component | Existing source | Required treatment |
|---|---|---|
| Acquisition | `revamp/batches/batch31_20261015T135412Z/code/acquire31.py` under the legacy root | Portable `acquire31.py` incorporates the actual B31 transfer repairs; three targeted portable regression checks pass. Original B30 remains historical |
| Source roster/operation | `batch30/acquisition/source_list_images_only.csv`, `operation_cq500_images_v2.json` | Local-only inputs; replace expired STOP/deadline paths; verify expected list/hash |
| Input/manifest binding | `batch28/manifest/build_manifest28.py`, `scripts/protocol28.py` | Preserve actual input/cohort identity checks |
| Image inference | `batch28/manifest/predict_image28.py`, `image_adapter27.py`, canonical recipe beside them | Preserve canonical precision/preprocessing and actual model binding |
| Geometry | `batch28/geometry/predict_geometry28.py`, `geometry_adapter27.py`, `bgeom.py`, `bstats.py` | Frozen parameters remain private unless cleared; name adaptation explicitly |
| Clinical selection | `batch28/scripts/selection28.py` | Clinical qualification precedes fixed tie-break |
| Reference | `batch28/scripts/reference28.py`, `reference/REFERENCE_AND_SCHEMA_CONTRACT.md` | Exact authenticated schema; no arbitrary extra readers |
| Paired scoring | `batch29/scripts/scorer29.py` plus imported `scorer28.py`, `scorer26.py`, `protocol28.py`, `reference28.py` | Trace imports; preserve B29 person-map checks and B28 cohort binding |
| Completion bounds | `batch26/evaluation/scorer26.py` | Use copied design; do not confuse bounds with CIs |
| Completed synthetic work | `batch25/scripts/sim25.py`, `transport25.py`, `precision25.py`, `agg25.py` | Reuse results; make source portable without changing design |

**No implicit authorization from availability:** possession of an input does not authorize a new clinical exposure or publication. Conversely, lack of CQ500 access does not block code portability, source extraction or the accepted-result handoff.

## October 20 input resolution

The private current-run `ACTIVE_INPUT_REGISTRY.json` resolves 91 paths across eight active dataset IDs (including the shared RSNA parent/subset and the two synthetic designs). It cites the retained B18/B20 source locks, separates historical MRI from C2 and original CT from the later sensitivity fit, and records three frozen model files. Local path resolution found no missing recorded local files; six drive-based entries were unavailable while Research was disconnected. A disconnected volume is not evidence that a source file is missing.

Expected hashes copied from previous locks are labeled as such. Path presence does not mean fresh byte verification; the two CT checkpoints and geometry file were checked separately. No old arrays or outcome statistics were recomputed for this inventory. The private breast handoff includes the retained 70-case mapping and repeat-group findings with their producer-evidence limitations.

## Output conventions

Public: code, task state, synthetic fixtures, schemas, source citations and permitted aggregate tables. Private: per-person/per-case records, labels/scores, clinical manifests, source images, checkpoint files until cleared, full local logs and path configuration. Keep public code/configs separated from private result directories. Preserve source artifacts read-only and write new task attempts under the run root.

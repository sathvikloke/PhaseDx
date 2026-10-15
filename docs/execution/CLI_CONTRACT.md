# Execution interfaces to implement

These interfaces are **specifications**, not currently installed commands. E01 creates the portable package; component tasks implement the subcommands. Do not run a guessed clinical command before its task and applicable real-data decision are complete. Existing historical scripts must first be inspected for their actual arguments and import dependencies.

Each subcommand must support `--help`, explicit input/config/output paths, nonzero exit on invalid input, no side effects on import and a read-only `--dry-run` where relevant. Use private paths supplied through a local configuration file. Never hard-code this handoff's machine paths.

| Required command shape | Implemented by | Main output |
|---|---|---|
| `python -m phasedx_external acquire --operation OP.json --dry-run` | E01/E06/E07 | Read-only source/budget/scope plan |
| `python -m phasedx_external acquire --operation OP.json --run-root RUN` | E08 | Verified image archives and transfer receipts |
| `python -m phasedx_external extract --operation OP.json --run-root RUN` | E09 | Source-bound extraction/member state |
| `python -m phasedx_external manifest --source-list LIST.csv --images IMAGE_ROOT --out MANIFEST_DIR` | E10 | Full roster, parsed instances, series, conflicts |
| `python -m phasedx_external select --manifest MANIFEST_DIR --review REVIEW.csv --protocol PROTOCOL.json --out SELECTED_DIR` | E12 | Fixed clinically qualified selection |
| `python -m phasedx_external lock --manifest SELECTED_DIR --models MODELS.json --protocol PROTOCOL.json --out LOCK.json` | E22 | Immutable actual-input/model execution identity |
| `python -m phasedx_external predict-image --lock LOCK.json --seed 42 --shard SHARD.json --out OUT_DIR` | E17/E23 | Source-bound seed42 logits/aggregate failures |
| Same image command with `--seed 123` | E17/E24 | Descriptive seed123 outputs |
| `python -m phasedx_external predict-geometry --lock LOCK.json --shard SHARD.json --out OUT_DIR` | E18/E25 | Frozen transported-geometry predictions |
| `python -m phasedx_external reference --schema SCHEMA.json --reads READS.csv --lock LOCK.json --out REF_DIR` | E14 | Source-matched reader majority/missingness |
| `python -m phasedx_external score --lock LOCK.json --predictions PRED_DIR --reference REF_DIR --grouping GROUPS.json --out RESULT_DIR` | E19/E26/E27 | Fixed primary and descriptive result artifacts |
| `python -m phasedx_external flow --lock LOCK.json --manifest MANIFEST_DIR --reference REF_DIR --predictions PRED_DIR --out FLOW_DIR` | E20/E28 | Full-source flow and eligible-labeled completion bounds |
| `python -m phasedx_external export --results RESULT_DIR --spec APPROVED_FIGURES.json --out PUBLIC_CANDIDATE_DIR` | E29/E38 | Candidate aggregate tables/figures |

`OP`, `RUN` and uppercase names above describe arguments; they are not literal input paths. Implementers must replace them from local configuration before execution. No endpoint or bootstrap setting is a free runtime tuning parameter: the scorer enforces the frozen protocol. Optional HPC scripts call the same interfaces after source/environment staging; do not create a separate scientific implementation.

## Minimum identity and output contracts

- **Candidate record:** source case key, archive/member identity, download/parse status, source-to-study/series mapping, selection status and reason. Private keys never enter public logs.
- **Prediction record:** source exam key, selected input identity, model/recipe identity, seed/arm, slice logit or fixed exam aggregate, success/failure reason, execution lock. No reference labels in inference input.
- **Reference record:** authenticated source exam key, reader-field availability and fixed majority result, source/schema identity, missingness reason. Not source-model probabilities.
- **Grouping record:** exam-to-person mapping plus source evidence, or explicit examination fallback. Hashing a claimed map alone is not source evidence.
- **Result record:** family/seed/endpoint/unit, intended/eligible-labeled/paired counts, positive/negative counts, both AUROCs and signed difference, interval or descriptive-only status, resampling recipe and undefined draws, original draw identity, grouping caveat.
- **Task receipt:** task/attempt, command, source/config/input identities, time/exit code, output references, failures and resource observations. Receipt existence is not successful scientific acceptance.

E21 exercises these interfaces with fake responses and point fields. New real inference/resamples occur only in E23–E27 after their dependencies. A test that uses the implementation to generate its own expected values is not independent scientific verification.

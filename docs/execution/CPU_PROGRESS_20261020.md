# CPU execution checkpoint — October 20

This checkpoint covers new portable execution code, input/source recovery and Methods work. It does not establish external clinical performance or submission readiness.

## Delivered

- Portable manifest, selection, frozen image/geometry prediction interfaces, reference admission, paired scoring, completion bounds, cohort flow and aggregate export.
- A clean-source installation, isolated installed-module smoke and **54 CPU fixture tests** passed. The complete command chain was exercised with fabricated cases and stub neural responses, without clinical inference or new bootstrap outcomes. The final receipt filename failure remains documented in [ENVIRONMENT.md](ENVIRONMENT.md).
- Tested isolated CPU and neural-dependency locks, strict loading of both frozen CT checkpoints on CPU, a guarded resumable queue, actual worker overlap/STOP tests, and three focused acquisition regression checks.
- An explicit JPEG-Lossless SV1 amendment. Twelve fixed compressed source images produced exact pixel/HU agreement across two decoders; four uncompressed images were skipped. This supports the bounded decoder change, not all-series clinical acceptance.
- A private registry resolves 91 paths across eight active dataset IDs, including both original CT checkpoints, frozen transported geometry, original retained resamples, historical/C2 MRI separation, breast mapping and completed synthetic inputs.
- Source-backed Methods from 15 inspected files and the canonical 20-run C2 record; retained breast mapping, 14 full-author references, exposure chronology and a 44-topic reporting map. Unknown executed-cache history and clinical facts remain explicit.
- The two completed synthetic studies have portable source/configuration and aggregate outputs. Their experiments were not rerun.
- A private manuscript v15 integrates the recovered Methods and corrects the obsolete C2 evidence summary. Existing numerical Results are unchanged.

The task queue currently records 24 execution deliverables complete, four in progress, 17 blocked and five conditional. Four bounded scientific/design tasks are recorded complete; manuscript revision remains in progress. These counts include reused prior acquisition evidence and are **not a percentage of the paper completed**.

## Active work and remaining clinical path

B31 already reports all 491 image archives acquired and verified, 490 extracted, and one source-corrupt archive preserved in the denominator. The expanded lossless-inclusive header scan is running after Research reconnected. The first October 20 scan attempt incorrectly classified absent-drive I/O as completed source failures; it read zero bytes and is rejected. Its code/logs/receipts remain. The repaired version pauses pending files on I/O loss and resumes without inventing exclusions.

A later pass found four changed extracted DICOM copies. The saved archive members matched the frozen inventory, so the four members were recovered into a separate input overlay while the original copies were preserved. The scan resumes from 21,522 previously verified records. See [input recovery](../study/INPUT_RECOVERY_20261020.md) for the exact scope and the verified storage move that relieved internal disk pressure.

The next external steps are the complete technical packet, blinded clinical qualification/selection, authenticated source-key/reference interpretation, frozen clinical input package, model predictions and the prespecified comparison. Technical decodability alone cannot fill unknown noncontrast, reconstruction-kernel or brain-coverage fields. Reference outcome rows remain unopened.

The neural dependencies now install and load the actual checkpoints on CPU; a clinical forward run still awaits the scientific input prerequisites. Project ethics/permitted-use statements, individual clinical ascertainment where required, author roster/approvals and final publication permissions remain separate factual inputs.

## Evidence boundaries

See [CPU_EVIDENCE_20261020.json](CPU_EVIDENCE_20261020.json), [Methods](../study/RETAINED_METHODS.md), [decoder amendment](../study/DECODER_AMENDMENT_20261020.md), [CLI contract](CLI_CONTRACT.md) and the regenerated [task index](TASK_INDEX.md).

Private receipts identify commands, input/source hashes, failed attempts and measured child times. Previously reviewed outcomes retain their original verification scope; copying them is not a new independent reproduction. No case identifiers, clinical images, outcome rows, weights or private machine paths are included in this code/task handoff.

## October 20 follow-up

A targeted [novelty and venue assessment](../study/NOVELTY_AND_JOURNAL_FIT_20261020.md) limits the contribution against primary prior art and corrects stale journal file-format guidance. A shorter private abstract candidate is prepared for author review, without new statistics.

The header writer now keeps one append handle open while retaining per-record flushing and all identity/STOP/resume checks. Five focused scanner tests passed. A 150-row identical-byte synthetic writing probe took 2.807 seconds with repeated open/close versus 0.292 seconds with a persistent handle; this measures record-writing overhead, not whole-scan acceleration. The prior scanner was checkpointed and versioned; 67,311 authenticated rows were reused with lineage in a separate V4 queue. Source inputs and original failures remain preserved.

## Preliminary clinical-review handoff

A private outcome-blind preliminary packet now contains 459 series from 178 exams whose entire planned source inventory is authenticated in closed shards. Its progress roster retains all 491 source exams: 312 remain pending scan and one has no extracted files. These statuses are not clinical exclusions or eligible-cohort counts. Clinical noncontrast/axial/kernel/coverage judgments, reviewer identity and evidence are blank. The packet allows early qualified review but cannot freeze a clinical input or substitute for final whole-inventory conflict reconciliation. Exact denominator, subset identity, blank clinical fields and output identities were checked. No source images or outcome rows were read during packet assembly.

The first packet attempt exposed exFAT AppleDouble plan sidecars. Its failed attempt is preserved. A synthetic regression reproduced the same failure in the final index builder; a narrow sidecar exclusion passed six cache/index tests. The active scanner was unchanged, and prior full-suite results were not rerun.

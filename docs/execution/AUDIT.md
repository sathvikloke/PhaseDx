# Scope and remaining-work audit — October 15, 2026

## Decision

The current project is **not yet submission-ready for the planned externally evaluated study**. The main missing empirical evidence is the frozen external CT result. There is useful completed internal and synthetic work; restarting all historical compute would waste it. The current implementation also needs a finite portability/acquisition completion step before real data execution.

This audit inspected repository `origin/main` at `68ba769`, the current manuscript source, October14 protocol/decisions, cycle report and B30 closure review, selected source interfaces and earlier scoped-review summaries. It also checked official journal/dataset pages. It did **not** run new clinical calculations, inspect all raw images, reproduce all training, independently repeat the historical audits or visually re-audit every figure. Existing numerical-check claims below are attributed to their saved reviews, not newly performed in this audit. Hashes/locators are in `source_map.json`.

No open issues or pull requests were returned by the repository query during this audit. The checkout had an untracked local `revamp/` tree; it was left untouched. This handoff was prepared in a separate worktree.

## The study and its evidence boundary

The October revision asks how evaluation unit, positional comparators and target/population definitions affect MRI/CT benchmark conclusions, adds a fixed synthetic transport counterexample, and plans an independent CT image-versus-transported-geometry comparison. It does not establish clinical benefit, causal shortcut use, equivalence or universal image irrelevance.

The public August RSNA manuscript instead centers a six-dataset zero-image audit and frozen-holdout stack-depth result. It is a different paper version, not a more recent completion of the October external study. It also explicitly excludes the old prevalence screen. Current GitHub text must therefore not be used as the sole record of current scope or readiness.

## Completed evidence to reuse

| Evidence | Saved status | Limit | Queue consequence |
|---|---|---|---|
| C2 prostate MRI retained scores/uncertainty | Prior reviews checked exact fold points, retained arrays and selected original count/weight calculations; 20 run records found | Not full raw-to-cache/training reproduction; some row-level historical receipts missing | E30/E34 recover Methods and bind existing results; no blanket retraining |
| Breast case-level contrasts | B18 selected review: 38 interval rows, 26 original-report mappings, 42 selected raw-count checks; all 24 paired intervals include zero | Does not establish equivalence or pathology/person mapping | E31/H06 address source interpretation, not interval reruns |
| Original internal CT comparison | B20 selected reconstruction/interval checks; v14 reports 283 studies/250 recorded patients | Internal subset, not external evaluation; later geometry sensitivity distinct | E32 recovers Methods; preserve original result |
| Controlled transport simulation | B25 returned complete 24,000-record grid; fixed 12 cells × 200 replicates | Constructed counterexample, not proof a retained CNN uses shortcuts | E35 reuse; H08 interpret |
| Precision simulation | B25 returned 108 cells × 500 replicates; selected arithmetic corroborated | Hypothetical assumptions/precision proxy, not achieved clinical power or coverage | E35 reuse; no expansion |
| Combined B25 selected checks | Saved review accepts 265 arithmetic checks | Not full independent generator/matrix replication | Cite exact scope |
| Image/geometry adapter fixtures | B26 producer reports 16-pair fidelity; later protocol/input-binding repairs exist | Restricted development geometry agreement does not prove external original-counter equivalence | E17/E18 port and make only changed-path checks |
| External source identification | Saved review matched torrent infohash, 493 file entries and webseed composition; 491 image ZIPs identified | HEAD results producer-only; no clinical download/performance established | E07–E10 acquire/parse after actual access decision |
| Manuscript v14/PDF | Full working reading draft exists | Clinical/Methods/author dependencies; not final journal files | H10 writes; E42 mechanically builds approved text |

## Consequential findings and closure conditions

| ID | Finding and evidence | Decision impact | Repair / closure |
|---|---|---|---|
| A01 | Public repo lacks the current portable October execution package; source lives in batch folders | A GitHub-only executor can run the wrong study or miss current work | E01/E02 + current entry point; preserve provenance |
| A02 | No external clinical predictions/results are established by the inspected final report | Empirical external discrimination/transport claim unsupported | E08–E29 with the fixed protocol; H07/H09 assess regardless of outcome |
| A03 | B30 static review identifies post-read resource refresh, probe reservation and full-partial resume gaps | Real transfer reliability incomplete | E06 three targeted fake-response regressions; then proceed, no broad loop |
| A04 | Prior cumulative download baseline unestablished; image-only current operation not executed | Requires one concrete execution scope/baseline record | E07/H02; does not block independent source/code work |
| A05 | Clinical series compatibility, exact reference schema and source-exam/person map not fully resolved | Wrong series/target/unit can invalidate external inference | E11–E15, H03–H05; conditional examination fallback already specified |
| A06 | Original counter-based geometry lacks established external semantics | Cannot call adapted comparator unchanged original-G-star validation | Already adjudicated transported G2 contract; implement E18, disclose H10 |
| A07 | Historical executed raw-to-cache and breast/CT Methods evidence incomplete | Reproducibility/reporting claims must remain bounded | E30–E33; H06 chooses truthful wording or a specifically justified correction |
| A08 | Breast local cases not fully source-linked to unique people/clinical ascertainment | Independent-person and clinical disease interpretations may be overstated | E31/H06; no invented pathology confirmation or outside identity linkage |
| A09 | Public README/old manuscript claims and checklists refer to different estimators/scopes | Obsolete claims can leak into final paper | E36/H01; add current handoff notice, preserve old record |
| A10 | B27 extra real-model fixture/200 draws, B29 real-path metadata write, and timing discrepancies are preserved in reports | Blanket perfect-execution/registration claims unsupported | Retain deviation log; no cleanup bypass or retroactive repair claim |
| A11 | Author/ethics/release facts are not established by computation | Submission approvals cannot be fabricated | H12/H13/H14; E41 uses approved inputs only |

A03 is a known implementation defect from static inspection, not evidence that real images were corrupted: no real transfer was established. A05/A07/A08 include missing evidence, which is different from proof that every underlying label or computation is wrong.

## Central claim coverage

| Claim | Status for this audit | What changes the assessment |
|---|---|---|
| Retained MRI/breast/internal CT comparisons have the stated bounded numerical support | Supported by saved scoped reviews; not newly recomputed here | A material source/identity defect would trigger E46/H07, not routine rechecking |
| Internal image advantage guarantees external performance | Not supported; fixed synthetic counterexample addresses the guarantee | No single external result establishes a universal guarantee |
| Actual image model relies causally on shortcuts | Not established | Would require a separately justified design; not a hidden queued experiment |
| The adapted frozen model outperforms geometry on CQ500 | Not assessed; experiment not executed | Locked new predictions/reference analysis and independent review |
| All Methods and clinical source chains are established | Partly checked / incomplete | Source-bound executed records and clinical decisions |
| Current manuscript ready for submission | Not yet | Completion of the chosen evidence path and H14 |
| Journal acceptance or high-tier publication assured | Not assessable and not promised | Editorial decision is external to the task queue |

## Coverage limits and scope control

Study question, target/unit, selection/exposure chronology, reference provenance, resampling design, comparator adaptation, code portability, publication-data boundaries and submission dependencies were examined through available documents/interfaces. Raw-image quality, all released source rows, complete training reproduction, all statistical output cells and a full new literature novelty review were **not performed** here. H07/H11 are concrete future scientific work, not completed audit badges.

There is no evidence-based reason to rerun every old MRI architecture or reacquire every historical dataset. The older six-dataset audit, phase-confound controls, screening study and rank-inversion work are inventoried in E36. E47–E49 activate only if a retained claim actually needs their missing evidence. No new cohort is added just to obtain a better-looking result.

The previous subjective completion estimate is not converted into a task-count percentage. Fifty execution cards do not imply fifty equally sized scientific milestones. The external clinical result and qualified interpretation remain disproportionately important.

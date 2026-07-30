# Trivial fraction as a continuous measure — every benchmark, every comparator

Generated 2026-07-29T18:50:12+00:00 by `pipeline/audit_prep/trivial_fraction_distribution.py`. No baseline is re-fitted here; every value is read from the artefact named in the last column.

> trivial fraction = (best zero-image baseline − chance) / (published − chance)

**Rows are not independent.** One benchmark contributes several rows when a paper reports several systems in one table. Read the *strongest published comparator per benchmark-arm* line as the primary distribution: a stronger comparator makes the denominator larger and the fraction smaller, so it is the conservative choice.

## Distribution

| set | n | min | Q1 | median | Q3 | max | ≤0.05 | 0.30–0.70 | ≥1 |
|---|---|---|---|---|---|---|---|---|---|
| all rows | 24 | -0.002 | 0.469 | **0.512** | 0.910 | 1.655 | 1 | 16 | 4 |
| strongest published comparator per benchmark-arm | 11 | -0.002 | 0.452 | **0.480** | 0.562 | 0.981 | 1 | 8 | 0 |
| peer-reviewed comparators, all rows | 18 | -0.002 | 0.455 | **0.485** | 0.514 | 0.889 | 1 | 16 | 0 |
| peer-reviewed comparators, strongest per benchmark-arm | 9 | -0.002 | 0.437 | **0.469** | 0.490 | 0.613 | 1 | 8 | 0 |
| preprint comparator (Rempe et al.) only | 6 | 0.973 | 1.020 | **1.142** | 1.518 | 1.655 | 0 | 0 | 4 |

## Rows

| benchmark | arm | published | system | peer-reviewed? | zero-image baseline | **trivial fraction** [CI] | verdict (secondary) | source |
|---|---|---|---|---|---|---|---|---|
| DeepLesion | 8-class lesion type | 0.9050 | triplet + type + location + size | yes | 0.5571 [0.524, 0.578] | **0.480** [0.431, 0.511] | PARTIAL | `paper/audit_results.md section 3.2 (pipeline/audit_prep/deeplesion_yan_conditions.py)` |
| DeepLesion | 8-class lesion type | 0.8620 | multi-scale ImageNet feature | yes | 0.5571 [0.524, 0.578] | **0.513** [0.460, 0.546] | PARTIAL | `paper/audit_results.md section 3.2 (pipeline/audit_prep/deeplesion_yan_conditions.py)` |
| DeepLesion | 8-class lesion type | 0.5970 | their own image-derived Location feature baseline | yes | 0.5571 [0.524, 0.578] | **0.889** [0.799, 0.947] | PARTIAL | `paper/audit_results.md section 3.2 (pipeline/audit_prep/deeplesion_yan_conditions.py)` |
| LUNA16 | FP-reduction track | 0.9500 | combined challenge solutions | yes | 0.0006 — | **-0.002** — | NOT MATCHED | `paper/audit_results.md section 3.6 (pipeline/audit_prep/luna16_cpm.py)` |
| PI-CAI | case level | 0.9100 | AI system | yes | 0.6917 [0.626, 0.755] | **0.467** [0.307, 0.623] | PARTIAL | `pipeline_out/trivial_baselines/picai_case_level.json` |
| PI-CAI | case level | 0.8600 | 62 radiologists, PI-RADS 2.1 | yes | 0.6917 [0.626, 0.755] | **0.532** [0.350, 0.710] | PARTIAL | `pipeline_out/trivial_baselines/picai_case_level.json` |
| RSNA ICH | any | 0.9843 | ResNeXt-101+BiLSTM | yes | 0.7374 [0.735, 0.740] | **0.490** [0.485, 0.495] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | any | 0.9752 | ResNeXt-101 (no LSTM) | yes | 0.7374 [0.735, 0.740] | **0.500** [0.495, 0.505] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | any (their split geometry) *(variant, not counted)* | 0.9843 | ResNeXt-101+BiLSTM | yes | 0.7381 [0.727, 0.750] | **0.492** [0.469, 0.516] | PARTIAL | `pipeline_out/audit_logs/rsna_ich_burduja_conditions.log` |
| RSNA ICH | epidural | 0.9851 | ResNeXt-101+BiLSTM | yes | 0.7122 [0.700, 0.725] | **0.437** [0.411, 0.464] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | epidural | 0.9703 | ResNeXt-101 (no LSTM) | yes | 0.7122 [0.700, 0.725] | **0.451** [0.424, 0.478] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | epidural (their split geometry) *(variant, not counted)* | 0.9851 | ResNeXt-101+BiLSTM | yes | 0.7186 [0.649, 0.776] | **0.451** [0.307, 0.569] | PARTIAL | `pipeline_out/audit_logs/rsna_ich_burduja_conditions.log` |
| RSNA ICH | intraparenchymal | 0.9927 | ResNeXt-101+BiLSTM | yes | 0.7514 [0.747, 0.755] | **0.510** [0.502, 0.518] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | intraparenchymal | 0.9883 | ResNeXt-101 (no LSTM) | yes | 0.7514 [0.747, 0.755] | **0.515** [0.507, 0.523] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | intraparenchymal (their split geometry) *(variant, not counted)* | 0.9927 | ResNeXt-101+BiLSTM | yes | 0.7527 [0.732, 0.772] | **0.513** [0.471, 0.552] | PARTIAL | `pipeline_out/audit_logs/rsna_ich_burduja_conditions.log` |
| RSNA ICH | intraventricular | 0.9970 | ResNeXt-101+BiLSTM | yes | 0.8048 [0.802, 0.808] | **0.613** [0.607, 0.620] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | intraventricular | 0.9953 | ResNeXt-101 (no LSTM) | yes | 0.8048 [0.802, 0.808] | **0.615** [0.609, 0.622] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | intraventricular (their split geometry) *(variant, not counted)* | 0.9970 | ResNeXt-101+BiLSTM | yes | 0.8058 [0.791, 0.820] | **0.615** [0.586, 0.644] | PARTIAL | `pipeline_out/audit_logs/rsna_ich_burduja_conditions.log` |
| RSNA ICH | subarachnoid | 0.9821 | ResNeXt-101+BiLSTM | yes | 0.6905 [0.686, 0.695] | **0.395** [0.386, 0.404] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | subarachnoid | 0.9644 | ResNeXt-101 (no LSTM) | yes | 0.6905 [0.686, 0.695] | **0.410** [0.400, 0.419] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | subarachnoid (their split geometry) *(variant, not counted)* | 0.9821 | ResNeXt-101+BiLSTM | yes | 0.6920 [0.665, 0.710] | **0.398** [0.342, 0.436] | PARTIAL | `pipeline_out/audit_logs/rsna_ich_burduja_conditions.log` |
| RSNA ICH | subdural | 0.9682 | ResNeXt-101+BiLSTM | yes | 0.7195 [0.717, 0.723] | **0.469** [0.463, 0.476] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | subdural | 0.9576 | ResNeXt-101 (no LSTM) | yes | 0.7195 [0.717, 0.723] | **0.480** [0.473, 0.487] | PARTIAL | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` |
| RSNA ICH | subdural (their split geometry) *(variant, not counted)* | 0.9682 | ResNeXt-101+BiLSTM | yes | 0.7211 [0.707, 0.735] | **0.472** [0.442, 0.502] | PARTIAL | `pipeline_out/audit_logs/rsna_ich_burduja_conditions.log` |
| fastMRI Prostate | DWI | 0.8610 | image + k-space (gold standard) | **no — preprint** | 0.8514 [0.816, 0.887] | **0.973** [0.876, 1.073] | MATCHED | `pipeline_out/trivial_baselines/fastmri_prostate_dwi_published.json` |
| fastMRI Prostate | DWI | 0.8090 | PCA x2 magnitude + phase | **no — preprint** | 0.8514 [0.816, 0.887] | **1.137** [1.023, 1.253] | MATCHED | `pipeline_out/trivial_baselines/fastmri_prostate_dwi_published.json` |
| fastMRI Prostate | DWI | 0.7140 | R=16 PCA coil combination | **no — preprint** | 0.8514 [0.816, 0.887] | **1.642** [1.478, 1.810] | MATCHED | `pipeline_out/trivial_baselines/fastmri_prostate_dwi_published.json` |
| fastMRI Prostate | T2 | 0.8610 | image + k-space (gold standard) | **no — preprint** | 0.8542 [0.812, 0.891] | **0.981** [0.865, 1.084] | MATCHED | `pipeline_out/trivial_baselines/fastmri_prostate_t2_published.json` |
| fastMRI Prostate | T2 | 0.8090 | PCA x2 magnitude + phase | **no — preprint** | 0.8542 [0.812, 0.891] | **1.146** [1.011, 1.266] | MATCHED | `pipeline_out/trivial_baselines/fastmri_prostate_t2_published.json` |
| fastMRI Prostate | T2 | 0.7140 | R=16 PCA coil combination | **no — preprint** | 0.8542 [0.812, 0.891] | **1.655** [1.459, 1.829] | MATCHED | `pipeline_out/trivial_baselines/fastmri_prostate_t2_published.json` |

## Behaviour at the extremes of the definition

Run on the tool's own `trivial_fraction()` so the limits section is a measurement, not an assertion. Baseline CI is baseline ± 0.02 throughout.

| case | baseline | chance | published | value | clipped | note |
|---|---|---|---|---|---|---|
| published far above chance, baseline mid-range | 0.7374 | 0.5000 | 0.9843 | 0.4902 | 0.4902 |  |
| published just above chance (headroom 0.021) | 0.6000 | 0.5000 | 0.5210 | 4.7619 | 1.0000 | clipped copy differs from value |
| published exactly at chance | 0.6000 | 0.5000 | 0.5000 | **undefined** | — | published (0.500) is at or below chance (0.500); the fraction is undefined |
| published BELOW chance | 0.6000 | 0.5000 | 0.4500 | **undefined** | — | published (0.450) is at or below chance (0.500); the fraction is undefined |
| baseline ABOVE published | 0.8540 | 0.5000 | 0.7140 | 1.6542 | 1.0000 | clipped copy differs from value |
| baseline exactly at chance | 0.5000 | 0.5000 | 0.9000 | 0.0000 | 0.0000 |  |
| baseline BELOW chance | 0.4533 | 0.5000 | 0.9843 | -0.0964 | 0.0000 | clipped copy differs from value |
| baseline exactly equals published | 0.9000 | 0.5000 | 0.9000 | 1.0000 | 1.0000 |  |
| published near-perfect | 0.7374 | 0.5000 | 1.0000 | 0.4748 | 0.4748 |  |
| non-AUROC chance anchor (average precision, prevalence 0.1434) | 0.2600 | 0.1434 | 0.6000 | 0.2554 | 0.2554 |  |

## Verdict counts (secondary, kept not deleted)

| set | MATCHED | PARTIAL | NOT MATCHED |
|---|---|---|---|
| all rows | 6 | 17 | 1 |
| peer-reviewed comparators only | 0 | 17 | 1 |

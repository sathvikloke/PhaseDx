# Scientific positioning and journal fit — October 20

The defensible contribution is a concrete analysis of evaluation units, positional comparators and eligible-pair support in the retained MRI/CT benchmarks, supported by a controlled transport counterexample. General claims that medical imaging models exploit shortcuts, that slice-level leakage is dangerous, or that dataset auditing is new would overstate the contribution.

## Primary work that the paper must distinguish

| Primary source | Established overlap | Consequence for PhaseDx |
|---|---|---|
| [Zech et al., PLOS Medicine, 2018](https://doi.org/10.1371/journal.pmed.1002683) | Hospital-associated information and disease prevalence can support strong internal discrimination with poorer cross-hospital generalization. | Transport failure and acquisition-associated signals are established problems. The simulation is a bounded counterexample, not their discovery. |
| [DeGrave et al., Nature Machine Intelligence, 2021](https://doi.org/10.1038/s42256-021-00338-7) | Empirical radiographic COVID-19 systems can select acquisition-related shortcuts. | PhaseDx's geometry predictability does not itself establish which cues its retained CNNs use. |
| [Yagis et al., Scientific Reports, 2021](https://doi.org/10.1038/s41598-021-01681-w) | MRI slice-based splitting can cause leakage and misleading performance estimates. | Distinguish train/test contamination from PhaseDx's evaluation-unit and eligible-pair questions. Do not claim that slice dependence or patient-level evaluation is newly identified here. |
| [SpurBreast, MICCAI 2025](https://papers.miccai.org/miccai-2025/0862-Paper0408.html) | A curated breast MRI dataset studies spurious correlations. | Breast MRI shortcut auditing is established prior art; the retained PhaseDx case-label/representation results need their specific interpretation. This source is a comparator in the literature, not an additional dataset to acquire. |
| [G-AUDIT, npj Digital Medicine, 2026](https://doi.org/10.1038/s41746-026-02807-y) | The publisher-indexed description presents a generalized, modality-agnostic dataset audit using attributes and task labels. | Do not claim a first general or modality-agnostic auditing framework. Full-text method comparison remains necessary before asserting a distinction from its utility/detectability construction. |

The publisher/conference descriptions and available primary text were checked on October20. The G-AUDIT publisher landing page could not be opened directly, and its PMC page returned a browser challenge; its comparison here is limited to publisher-indexed text. No access restriction was bypassed. This is a targeted prior-art assessment, not an exhaustive systematic search or proof that the remaining empirical contribution is unique.

## Current contribution and limits

Lead with the observed benchmark findings: within-fold versus pooled ranking, slice versus subject/case aggregation, and the support lost by position-restricted comparisons. Explain that high geometry discrimination diagnoses target/population structure without attributing a mechanism to an image model. Keep the original internal CT comparison distinct from its later geometry sensitivity analysis and the explicitly transported external comparator.

The controlled simulation addresses the insufficiency of an internal advantage as a transport guarantee under its fixed generator. It cannot establish that a real CNN exhibits the constructed failure. The precision experiment is a planning calculation. Neither substitutes for the planned external CT result.

The external comparison can strengthen the empirical contribution regardless of sign if its denominator, clinical compatibility, target and missing predictions are handled honestly. Complete the selected external route before describing the project as an external validation study. A retrospective-audit-only submission would be a separate scope decision; the journal instructions do not imply that every methodological audit universally requires a new cohort.

## Working journal route

**Radiology: Artificial Intelligence, Original Research** is a plausible scope match for the full empirical audit. This is a fit assessment, not a prediction of acceptance. Its [current author instructions](https://pubs.rsna.org/page/ai/author-instructions) describe hypothesis-driven original research and currently specify 3,000 main-text words, a structured abstract of at most250 words, up to35 references, six figures and four tables. Technical Developments is less natural for the current contribution because the paper is not primarily introducing a new algorithm, device or dataset. A smaller AI in Brief submission would require a deliberate narrowing of the paper.

The instructions' January2026 update requires manuscript text as DOCX. An older preparation checklist still mentions PDF; follow the current instructions when preparing final submission files. The existing reading PDF is useful for review but does not establish a final submission package.

The current v15 draft counts263 abstract words and3,267 main-body words after excluding Markdown table rows, or3,512 including them under a conservative local count. Reduce the abstract and move appropriate technical detail to a source-backed supplement while preserving targets, units, uncertainty, selection and limitations in the main text. Counts depend on the journal's final convention; allow margin. Final authorship, ethics/permitted-use facts, scientific approval and the chosen evidence scope remain unresolved.

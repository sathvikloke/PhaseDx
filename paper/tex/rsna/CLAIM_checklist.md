# CLAIM 2024 checklist — *What a Slice-Level Benchmark Certifies without the Pixels: An Audit of Six Public Imaging Datasets*

Completed against the **Checklist for Artificial Intelligence in Medical Imaging (CLAIM):
2024 Update** (Tejani AS, Klontzas ME, Gatti AA, Mongan JT, Moy L, Park SH, Kahn CE Jr, and
the CLAIM 2024 Update Panel. *Radiology: Artificial Intelligence* 2024;6(4):e240300;
doi:10.1148/ryai.240300), the 44-item list published at `pubs.rsna.org/page/ai/claim`.
Item numbering and section structure follow that update, including its renaming of the
*Ground Truth* section to **Reference Standard** and its renumbering relative to CLAIM 2020.

**This file is uploaded separately, and it is anonymized**, on the assumption that a
reviewer may see it. It therefore names no author, institution, repository, archive DOI or
software package. Where an item is satisfied on a document that is *not* anonymized — the
Full Title Page or the cover letter — the row says so and stops there.

**How to read the "Where" column.** Page numbers are pages of the compiled manuscript
document — the single anonymized file, 22 pages. The journal requires that pages not be
numbered, so these are counted from the first page of that file:

| Pages | Contents |
|---|---|
| 1 | Abbreviated Title Page — title, article type, Summary Statement, three Key Points, abbreviations, keywords |
| 2 | Structured abstract |
| 3 | Introduction |
| 4–6 | Materials and Methods — *Study Design*, *Benchmark Eligibility*, *The Zero-Image Baseline Family*, *Estimator, Aggregation, Intervals, and Nulls*, *The Comparison Statistic*, *Statistical Analysis* |
| 6–9 | Results — *One Locked Score Vector, Read at Two Units*, *The Mechanism Is Stack Depth*, *Pipeline Uncertainty over 24 Holdouts*, *Aggregation, and the Rest of the Baseline Family*, *The Same Reading on Every Other Benchmark-Arm*, *How Much of a Published Margin Is Reachable with No Pixels* |
| 9–11 | Discussion, with Limitations second-to-last and a summary paragraph last |
| 11 | Acknowledgments — large-language-model disclosure naming the tool; data and code availability |
| 12–15 | References (25) |
| 16–17 | Figure Legends (3) |
| 18–21 | Tables 1–4 — **Table 1 is the benchmark-eligibility table**, cited in Materials and Methods, which is why it is numbered first; Table 2 is the flagship two-unit table; Table 3 is estimator, aggregation and baseline sensitivity; Table 4 is every audited benchmark-arm |

**Updated 2026-08-13 for the frozen-holdout rewrite.** The estimator behind every flagship
number changed, the primary baseline was locked before the held-out data were touched, the
cross-study comparison was demoted to descriptive, and the tables were renumbered — the
benchmark-eligibility table is now Table 1 because it is cited in Materials and Methods.
Section names and table numbers below are the current ones. Explicit page numbers have been
removed rather than left stale; the compiled manuscript is 22 pages and the page map above
is current.

Figures themselves are in the separate combined figure document — 3 pages, one figure and
its legend per page. Legends there are byte-identical to the Figure Legends section of the
manuscript.

**Why so many items are marked not applicable.** This is a **retrospective secondary
analysis of publicly released label files**. It develops no diagnostic model, offers no
model for clinical use, and downloads no pixel data. The models it fits are five deliberately
trivial *pixel-blind null* models — a constant prevalence predictor, a 20-bin estimate of
label rate given relative slice position, a volume-size score, a depth-limited classification
tree over the label file's remaining non-image columns, and a combined position-plus-metadata
tree — used as audit instruments against published performance numbers. Every "not
applicable" below carries the reason on the same line, as the journal's instructions ask.
No item is left blank.

**Status key:** **Y** = reported, with the location given · **P** = partly reported, with
what is missing stated · **NA** = not applicable, with the reason stated · **ACTION** = a
human must supply something before upload.

---

## Title and Abstract

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 1 | Identification as a study of AI methodology, specifying the category of technology used | **P** | Abstract, and Key Points, identify the technology explicitly: five pixel-blind models — a constant predictor, a binned positional estimator, a volume-size score, a classification tree over non-image columns, and a combined tree. **The title does not carry the words "artificial intelligence" or a technology category**, because the study's object is an evaluation protocol rather than an AI system; the title names the audit and the seven datasets instead. **ACTION** — a deliberate choice, flagged here so the editors can require a title change rather than discover the gap. |
| 2 | Summary of study design, methods, results, and conclusions | **Y** | Structured abstract — Purpose / Materials and Methods / Results / Conclusion, four sections, 248 words. |

## Introduction

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 3 | Scientific and/or clinical background, including the intended use and role of the AI approach | **Y** | Introduction. The intended use of the models is stated as an audit instrument, not a clinical one; the Discussion, restates that every claim is about an evaluation protocol and that no claim is made about any published model's internals. |
| 4 | Study aims, objectives, and hypotheses (if not a data-driven approach) | **Y** | Introduction, final paragraph ("The purpose of this study was to quantify how much published slice-level performance … is reachable from published non-pixel fields under a subject-disjoint split, and whether the same score vector, read per patient, retains it"), and Abstract Purpose. |

## Methods — Study Design

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 5 | Prospective or retrospective study | **Y** | *Study Design*, and Abstract Materials and Methods: retrospective secondary analysis of publicly released, de-identified label files from benchmarks released 2016–2024; analyses ran July 2026. |
| 6 | Study goal | **Y** | *Study Design*, with the two questions stated at the end of the Introduction. The design is a measurement study, not model development or a diagnostic accuracy study of a proposed model. |

## Methods — Data

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 7 | Data sources | **Y** | *Benchmark Eligibility*: seven named public benchmarks across seven label files, each cited to its release paper. The slice ordering and header columns for the largest benchmark are declared to come from a third-party, MIT-licensed, pixel-free tabular mirror of the same release rather than from the official release, with the join key and the reconciliation given (). |
| 8 | Inclusion and exclusion criteria | **Y** | *Benchmark Eligibility*: a dataset entered if four fields — subject identifier, slice index, label, train/test assignment — were obtainable without pixel data. One further exclusion is recorded with its measurement: another challenge's release failed the slice-ordering run-length test and was excluded on that measurement. |
| 9 | Data preprocessing | **Y** | *The Zero-Image Baseline Family*: relative position defined as (slice − minimum)/(maximum − minimum) within a volume; columns that are the label under another name, or derived from the images, excluded, with every column recorded either way. The join of the official labels to the third-party ordering is described on. |
| 10 | Selection of data subsets | **Y** | No arm is selected. All eight DeepLesion body-part arms are listed in Table 4 and **all eight are plotted in Figure 2**; the earlier version carried one of them into the figure and disclosed the selection, and the revision removed the selection instead. The primary zero-image baseline was likewise **locked before any held-out value was computed** rather than chosen as the strongest of five (*The Zero-Image Baseline Family*; Table 3, block C). |
| 11 | De-identification | **Y** | *Study Design*: publicly released, de-identified label files; no data obtained through intervention or interaction with any living individual; no identifiable private information used; no pixel data downloaded. The same paragraph records that participant age and sex are not distributed with these files. |
| 12 | Missing data | **P** | Handled where it changes a reported value and stated there: the patient-level AUC is *undefined* for Duke Breast because 922 of 922 patients are positive (Results; Table 3; Figure 2 legend, which draws the two one-unit-only arms in the margins so a value that does not exist cannot be read as a low one). Limitations, records that fastMRI+ knee covers 199 of 1,173 roster volumes, that age and sex are absent from every file, and that PI-CAI's published comparison is on a hidden testing cohort while the baseline is on the development set. **Missing from the manuscript:** the row counts dropped from the two fastMRI Prostate arms as validation or unrecognized split values (1,458 for DWI, 1,462 for T2) are in the released artifacts but are not stated in the text. |
| 13 | Image acquisition protocol | **NA** | No images were used — no pixel archive was downloaded for any benchmark (*Study Design*), so this study applied no acquisition protocol; each benchmark's own release paper, cited on, describes the acquisition of its images. |

## Methods — Reference Standard

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 14 | Definition of method(s) used to obtain the reference standard | **Y** | The reference standard is each benchmark's **own released label column**, used unmodified; *Benchmark Eligibility*, names the label file for each benchmark and cites its release. Table 1, enumerates the six official label columns of the largest benchmark. |
| 15 | Rationale for choosing the reference standard | **Y** | *Benchmark Eligibility*, with the Introduction: the study measures what a benchmark's published label file certifies, so the released label is not a proxy for the reference standard — it *is* the object under audit, and substituting any other standard would change the question. |
| 16 | Source of reference standard annotations | **Y** | Third-party annotations from the benchmark releases, cited individually on; no annotation was performed by these authors. |
| 17 | Annotation of test set | **Y** | Test rows carry the same released labels as training rows; the split is the benchmark's own published train/test assignment where one exists, and on the flagship benchmark one frozen patient-disjoint holdout with a single fit (*Estimator, Aggregation, Intervals, and Nulls*). |
| 18 | Measures of inter- and intrarater variability of features described by the annotators | **NA** | No annotation was performed for this study, and the public label files carry a single consensus label per row with no per-reader records, so inter- and intrarater variability cannot be computed from them. Where a release reports its own reader agreement, that is a property of the cited release and not a measurement of this study. |

## Methods — Data Partitions

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 19 | How data were assigned to partitions | **Y** | *Estimator, Aggregation, Intervals, and Nulls*: a benchmark's own published train/test assignment where one exists; on RSNA ICH, which publishes none, one frozen patient-disjoint holdout of 30% of patients at a seed fixed **before any held-out value was computed**, single fit, no pooling. The pooled out-of-fold five-fold estimator used in an earlier round is retained only as a labelled sensitivity analysis and on four arms that could not be recomputed, and Table 4 names them. Table 3, block A, reports the whole procedure repeated over 24 independent holdouts and a second implementation's own eight. |
| 20 | Level at which partitions are disjoint | **Y** | *Estimator, Aggregation, Intervals, and Nulls*: **subject-disjoint throughout**. This is the item the paper exists to press on; the Introduction, distinguishes this study from the leakage literature precisely on it. Table 1's note, and Table 4's note, additionally report the constant-predictor floor that pooling out-of-fold predictions across folds of differing training prevalence introduces, per row, rather than assuming it is zero. |

## Methods — Testing Data

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 21 | Intended sample size | **NA** | No target sample size was set and no power calculation was performed: every eligible row of every eligible public label file was used, giving 752,802 slices from 18,938 patients on the largest benchmark, of which 13,257 patients train the model and 5,681 are held out (Abstract; *Benchmark Eligibility*; Table 2 note). |

## Methods — Model

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 22 | Detailed description of model | **Y** | *The Zero-Image Baseline Family*, describes all five models in full — a constant prevalence predictor, a 20-bin estimate of label rate given relative slice position, a volume-size score, a depth-limited classification tree over the label file's remaining non-image columns, and a combined position-plus-metadata tree — together with a fit-free variant, the negated absolute deviation of relative position from 0.5, that uses no training data. The tree's depth and its input columns for the one benchmark where the tree is the reported baseline are given in the Table 4 note (depth 3; patient age, prostate-specific antigen level, center, scan year, and the tree splits on all four). |
| 23 | Software libraries, frameworks, and packages | **Y** | *Statistical Analysis*: Python 3.14 (Python Software Foundation, Wilmington, Del) with NumPy 2.4 and pandas 3.0, and, for the tree baselines, scikit-learn 1.8. These are the versions of the environment that produced the frozen-holdout artefacts. |
| 24 | Initialization of model parameters | **NA** | There are no parameters to initialize: four of the five models are closed-form summaries of the training rows (a prevalence, a 20-bin histogram, a slice count, and their combination) with no iterative optimization, and the fifth is a depth-limited tree fitted deterministically. The seed governing every stochastic step that does exist — fold assignment and bootstrap resampling — is recorded per run (*Statistical Analysis*). |
| 25 | Details of training approach | **Y** | *The Zero-Image Baseline Family*: each model is fit on training rows and scores test rows; the positional model's one hyperparameter, the bin count, is fixed at 20 and swept over 5/10/20/50 as a sensitivity analysis (Results; Table 2 note). No optimizer, learning rate, epoch count or augmentation exists to report. |
| 26 | Method of selecting the final model | **Y** | **No selection remains.** (a) The numerator of the comparison statistic is the **locked** positional baseline, fixed before any held-out value was computed, not the strongest of five; *The Comparison Statistic* says so and says why — a maximum over five correlated baselines on the same test data is a winner's-curse estimate biased upward, which was the previous version's defect. The other four baselines are reported as secondary in Table 3, block C, where the locked model is also the strongest on all six labels, so the lock costs nothing on this benchmark; the one row it moves anywhere is PI-CAI, from 0.467 to exactly 0.000. (b) No DeepLesion arm is selected: all eight body-part arms are listed in Table 4 and all eight are plotted in Fig 2. |
| 27 | Ensembling technique | **NA** | No ensembling. Each reported value comes from a single model; the one model that combines inputs, the position-plus-metadata tree, is a single tree over both column groups rather than an ensemble, and it is reported against its own permutation null in the Table 2 note. |

## Methods — Evaluation

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 28 | Metrics of model performance | **Y** | *Estimator, Aggregation, Intervals, and Nulls*: every score vector is read at **both** units — slice-level AUC, and patient-level AUC after the **pre-specified mean** aggregation, with six alternative operators reported as sensitivity analyses, each with the number of distinct patient scores it produces so a near-degenerate operator cannot be mistaken for a measurement (Table 3, block B). *The Comparison Statistic* defines the trivial fraction, its chance anchors (0.5 for AUC, the majority-class rate for multiclass accuracy) and the conditions under which it is unclipped, and states that it is **descriptive only**: the four benchmarks sit on four metrics against four chance anchors, so no average is taken across them. The one arm scored on a detection metric, sensitivity at one false positive per scan, is reported with its random-score reference of 0.0027 in Results and in the Figure 3 legend. |
| 29 | Statistical measures of significance and uncertainty | **Y** | *Statistical Analysis*: an estimation study; no significance test was pre-specified and no *P* values are reported. Every AUC carries a 95% percentile bootstrap CI resampling **subjects**, replicate count and seed recorded per run, and **every primary value carries a second uncertainty statement beside it** — the spread of the whole procedure over 24 independent holdouts, which unlike the bootstrap conditions on neither the fitted model nor the split. The two are not ordered in a fixed direction and the paper says so and reports both (*Estimator…*; Results; Table 3, block A). Table 4's note gives the replicate count per row. |
| 30 | Robustness or sensitivity analysis | **Y** | Seven, all reported, all on the same frozen holdout. A bin sweep at 5/10/20/50 at both units and a fit-free variant that uses no training data (Table 2 note); a within-series permutation null preserving prevalence, subject clustering and stack depth, computed unstratified **and at fixed depth** (Results; Table 2 note); the constant-predictor floor reported per row rather than assumed, and printed in bold on the four arms where it is not 0.500 (Table 4); the whole procedure repeated over **24 independent holdouts**, plus a second implementation's own eight (Table 3, block A); **seven aggregation operators** with distinct-score counts (Table 3, block B); the four secondary zero-image baselines beside the locked primary one (Table 3, block C); and a stack-depth-stratified re-reading at exact depth and in 5-slice strata (Results; Table 2 lower block). |
| 31 | Methods for explainability or interpretability | **NA** | No post hoc explainability method was applied, and none is needed: the models are a 20-value lookup table, a slice count, a prevalence and a depth-3 tree, each fully inspectable as written. Where a result's mechanism was in question the paper measures it directly instead — the sub-chance patient-level reading is traced to stack depth entering through the mean operator, and maximum aggregation is shown to be **exactly** degenerate under a single fit (Results; Table 2 note; Table 3, block B). |
| 32 | Evaluation on internal data | **Y** | *Estimator, Aggregation, Intervals, and Nulls*, with the results in Tables 2–4. Every value is an out-of-sample reading: a benchmark's own held-out split where one exists, otherwise a frozen patient-disjoint holdout. Apparent-versus-held-out training AUC is reported for the flagship model (0.738 against 0.738) in the Table 2 note, on that same holdout. |
| 33 | Testing on external data | **NA** *as CLAIM means it* | No model is proposed for use anywhere, so there is no model to transfer to an external institution. The nearest analogue is reported and is arguably stronger for this study's purpose: the identical construction was run unchanged on **seven independently released benchmarks across seven label files**, from different institutions, modalities and organs, and every arm is reported including those that do not fire (Results; Table 4; Fig 3). The candidate universe those seven were drawn from, and every exclusion, are in **Table 1**. |
| 34 | Clinical trial registration | **NA** | Not a clinical trial and not a prospective study of any kind; no patient was recruited and no intervention was assigned (*Study Design*). Nothing here is registrable under the ICMJE clinical trial registration statement. |

## Results — Data

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 35 | Numbers of patients or examinations included and excluded | **P** | Included counts are given for every arm: 752,802 slices, 21,744 series and 18,938 patients for the largest benchmark (Abstract; *Benchmark Eligibility*; Table 2 note); 665 patients for DeepLesion and 1,476 development cases for PI-CAI in the Table 4 note; 922 of 922 positive patients for Duke Breast; 199 of 1,173 roster volumes for fastMRI+ knee, Limitations. Exclusion at the *benchmark* level is given with its measurement. **Missing from the manuscript:** a per-arm accounting of excluded rows — specifically the fastMRI Prostate rows dropped as validation or unrecognized split values, and how the 199 available fastMRI+ volumes were determined and whether they differ systematically from the remaining 974. |
| 36 | Demographic and clinical characteristics of cases in each partition and dataset | **NA** | **Age and sex are not distributed with these public label files and therefore cannot be reported.** This is stated three times rather than passed over: Abstract Materials and Methods; *Study Design*; and Limitations, which records that no subgroup analysis was possible in consequence. The one benchmark whose file does carry clinical variables — age and prostate-specific antigen level — has them reported, because there they are inputs to a baseline (Results; Table 4 note). |

## Results — Model Performance

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 37 | Performance metrics and measures of statistical uncertainty | **Y** | Results, and Tables 1–4. Every point estimate is printed with a 95% subject-clustered bootstrap interval at a stated replicate count, at three decimals throughout — the most the smallest replicate count supports (Table 2 note). Results reports no percentage of this study's own beyond the 95% interval level and the 30% holdout fraction, which is given with both counts (13,257 training and 5,681 held-out patients of 18,938). Every primary value additionally carries the spread of the whole procedure over 24 independent holdouts (Table 3, block A). |
| 38 | Estimates of diagnostic performance and their precision | **Y**, in the only sense that applies | No diagnostic model is proposed, so there is no diagnostic performance of this study's own to estimate; what is reported instead, with precision, is the performance of deliberately trivial baselines against published comparators. Discrimination with intervals: Tables 2 and 4. The one operating-point estimate in the paper, sensitivity at one false positive per scan on LUNA16, is reported in Results and in the Fig 3 legend, which states that no interval is available for it. Sensitivity, specificity and predictive values of a proposed model are **not applicable** — there is no proposed model. |
| 39 | Failure analysis of incorrect results | **Y** | The paper's largest single analysis is a failure analysis of its own headline number. The sub-chance patient-level reading is traced to stack depth entering through the mean operator and stops being sub-chance when depth is held fixed (Results; lower block and note of Table 2); maximum aggregation is shown to be **exactly** degenerate under a single fit and is labelled so rather than quoted as a robustness check; the constant-predictor floor introduced by pooling across folds was the reason the estimator was changed, and on the four arms that could not be recomputed it is printed in bold as a floor (Table 4); the benchmarks on which the measure does not fire are reported at equal prominence (Results; Fig 3); and the paper states where its own frozen holdout sits inside its family of 24 rather than presenting it as a tighter estimate (Results; Table 3 note). |

## Discussion

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 40 | Study limitations | **Y** | Limitations paragraph, placed second-to-last as the journal requires. It carries: the construction is prior art; no peer-reviewed comparison is matched and the only matched rows rest on a preprint comparator; the headline patient-level value is not robust to the analysis choices, running 0.415 to 0.652 over 5 to 50 bins crossed with six aggregation operators; two benchmarks are not pixel-free in the same clean sense; the PI-CAI comparison is across cohorts; age and sex are unavailable; and coverage is confined to benchmarks whose label fields were obtainable without a pixel data-use agreement. |
| 41 | Implications for practice, including intended use and/or clinical role | **Y** | Discussion, in three parts: what a reader of an imaging AI paper should ask of a reported AUC; what benchmark publishers can do at no cost with three fields they already release; and what this protocol cannot do, namely detect shortcuts that live in the pixels. The scope of the clinical claim is stated rather than implied — no device summary or regulatory filing was examined, and nothing in these data speaks to the unit at which any cleared product was validated (). |

## Other Information

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 42 | Provide a reference to the full study protocol or to additional technical details | **P — ACTION** | Additional technical detail is in the released archive, described in Acknowledgments. The one pre-specified analysis choice the paper relies on, 20 bins, is declared pre-specified "in a protocol that will be identified on acceptance" (Limitations) — deferred **only** because naming the protocol document would identify the authors under double-anonymized review. **Human action:** supply the protocol reference on acceptance, or now if the editors would prefer it. |
| 43 | Statement about the availability of software, trained model, and/or data | **P — ACTION** | Acknowledgments states that the audit tool, the per-benchmark output payloads, the label-table preparation scripts, the second implementation sharing no code with the first, and the cross-study comparison assembler are released under a Creative Commons license in a public repository and archived with a persistent identifier, that no benchmark's pixel data are redistributed, and that the fastMRI label files are available only by application under that initiative's own agreement. **The repository address and the commit identifier are withheld from the manuscript for anonymized review.** They are given in full on the Full Title Page and in the cover letter, and the cover letter tells the editor they can be written into Materials and Methods instead if that is preferred. This is a live conflict between the journal's Algorithm and Code Transparency policy and its double-anonymization instruction; it is flagged rather than papered over. |
| 44 | Sources of funding and other support; role of funders | **Y**, on the non-anonymized upload | Not in the manuscript, because a funding statement is identifying. Stated on the **Full Title Page**: no funding of any kind — no grant, fellowship, award, institutional allocation, industry contract, equipment loan, compute donation or in-kind support — and therefore no funder role in design, analysis, interpretation, writing or the decision to submit. Repeated in the cover letter under Declarations. |

---

## Two disclosures that CLAIM does not itemize and this journal requires

Recorded here so that a reader of this checklist alone does not have to reconstruct them.

- **Use of large language models.** Assistance was used in two ways — in drafting and editing
  the manuscript, and in **writing the analysis code that produces every number in the
  Results**. No reported value was generated by a language model: every number is a
  deterministic output of the released code run on a published label file, and the flagship
  estimate was reproduced by a second implementation sharing no code with the first, which
  ran the same pre-specified protocol on its own holdouts and agrees to 0.0003 AUC per slice
  and 0.005 per patient, with the constant predictor at exactly 0.500 in every draw of both.
  The disclosure appears in the Acknowledgments of the manuscript (anonymized), on the Full
  Title Page and in the cover letter, in the same words, and **the tool is named in all
  three**: Claude (Opus 5), Anthropic, accessed 2026-07-27 to 2026-08-12. It is named in the
  manuscript at submission because the journal requires name, version, manufacturer and
  dates of access there, and naming it identifies nobody.
- **Third-party slice ordering for the flagship benchmark.** Every RSNA intracranial
  hemorrhage number rests on a slice ordering the official release does not contain. Its
  source, licence, join key, reconciliation and falsifiable verification test are in
  Materials and Methods — not in a late clause of the Discussion — together with the
  fact that another challenge's release failed the same test and was excluded.

## What a human must still supply before this checklist is uploaded

1. **Item 1** — decide whether to leave a title that does not name a technology category, or
   retitle. Nothing else in the package depends on the choice.
2. **Items 42 and 43** — the protocol reference, the repository address and the commit
   identifier, all currently withheld for anonymization. Decide with the editor whether they
   go into Materials and Methods now or on acceptance.
3. **Items 12 and 26** — two gaps this checklist declines to overstate: the fastMRI Prostate
   dropped-row counts, and how the 199 available fastMRI+ knee volumes were determined. Each
   is a sentence of text in the manuscript, not new analysis. **Item 35's gap is closed**: the
   numerator is no longer a maximum over five correlated baselines chosen on the same test
   data. The primary baseline is locked before the held-out data are touched, the other four
   are reported as secondary, and the one row the lock moves — PI-CAI, from 0.467 to exactly
   0.000 — is stated in Results, in the Table 4 note and in the Figure 3 legend.

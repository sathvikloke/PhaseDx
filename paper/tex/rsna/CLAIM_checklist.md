# CLAIM 2024 checklist — *What a Slice-Level Benchmark Certifies without the Pixels: An Audit of Seven Public Imaging Datasets*

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
| 4–6 | Materials and Methods — *Study Design* (4), *Benchmark Selection* (4–5), *The Zero-Image Baseline Family* (5), *Evaluation, Intervals, and Nulls* (5–6), *The Comparison Statistic* (6), *Statistical Analysis* (6) |
| 6–9 | Results — *One Pixel-Blind Score Vector, Read at Two Units* (6–8), *How Much of a Published Margin Is Reachable with No Pixels* (8–9) |
| 9–11 | Discussion, with Limitations second-to-last (10–11) and a summary paragraph last (10) |
| 11 | Acknowledgments — large-language-model disclosure; data and code availability |
| 12–16 | References (25) |
| 17–18 | Figure Legends (3) |
| 19–22 | Tables 1–4, one per page |

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
| 1 | Identification as a study of AI methodology, specifying the category of technology used | **P** | Abstract, p 2, and Key Points, p 1, identify the technology explicitly: five pixel-blind models — a constant predictor, a binned positional estimator, a volume-size score, a classification tree over non-image columns, and a combined tree. **The title does not carry the words "artificial intelligence" or a technology category**, because the study's object is an evaluation protocol rather than an AI system; the title names the audit and the seven datasets instead. **ACTION** — a deliberate choice, flagged here so the editors can require a title change rather than discover the gap. |
| 2 | Summary of study design, methods, results, and conclusions | **Y** | Structured abstract, p 2 — Purpose / Materials and Methods / Results / Conclusion, four sections, 248 words. |

## Introduction

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 3 | Scientific and/or clinical background, including the intended use and role of the AI approach | **Y** | Introduction, p 3. The intended use of the models is stated as an audit instrument, not a clinical one; the Discussion, p 9, restates that every claim is about an evaluation protocol and that no claim is made about any published model's internals. |
| 4 | Study aims, objectives, and hypotheses (if not a data-driven approach) | **Y** | Introduction, final paragraph, p 3 ("The purpose of this study was to quantify how much published slice-level performance … is reachable from published non-pixel fields under a subject-disjoint split, and whether the same score vector, read per patient, retains it"), and Abstract Purpose, p 2. |

## Methods — Study Design

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 5 | Prospective or retrospective study | **Y** | *Study Design*, p 4, and Abstract Materials and Methods, p 2: retrospective secondary analysis of publicly released, de-identified label files from benchmarks released 2016–2024; analyses ran July 2026. |
| 6 | Study goal | **Y** | *Study Design*, p 4, with the two questions stated at the end of the Introduction, p 3. The design is a measurement study, not model development or a diagnostic accuracy study of a proposed model. |

## Methods — Data

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 7 | Data sources | **Y** | *Benchmark Selection*, pp 4–5: seven named public benchmarks across eight label files, each cited to its release paper. The slice ordering and header columns for the largest benchmark are declared to come from a third-party, MIT-licensed, pixel-free tabular mirror of the same release rather than from the official release, with the join key and the reconciliation given (p 4). |
| 8 | Inclusion and exclusion criteria | **Y** | *Benchmark Selection*, p 4: a dataset entered if four fields — subject identifier, slice index, label, train/test assignment — were obtainable without pixel data. One further exclusion is recorded with its measurement, p 4: another challenge's release failed the slice-ordering run-length test and was excluded on that measurement. |
| 9 | Data preprocessing | **Y** | *The Zero-Image Baseline Family*, p 5: relative position defined as (slice − minimum)/(maximum − minimum) within a volume; columns that are the label under another name, or derived from the images, excluded, with every column recorded either way. The join of the official labels to the third-party ordering is described on p 4. |
| 10 | Selection of data subsets | **Y** | Which arm of a multi-arm benchmark is carried into the figures is disclosed rather than assumed: Results, p 8, and the Table 3 note, p 21, state that eight DeepLesion body-part arms were computed, that all eight are listed, and that the pelvis arm carried into the text and Figure 2 is the strongest of the eight and **was selected after all eight had been computed, not on a rule stated in advance**. The 1,500-patient subsample used by two of the four replication routes is identified in Table 2, p 20. |
| 11 | De-identification | **Y** | *Study Design*, p 4: publicly released, de-identified label files; no data obtained through intervention or interaction with any living individual; no identifiable private information used; no pixel data downloaded. The same paragraph records that participant age and sex are not distributed with these files. |
| 12 | Missing data | **P** | Handled where it changes a reported value and stated there: the patient-level AUC is *undefined* for Duke Breast because 922 of 922 patients are positive (Results p 8; Table 3, p 21; Figure 2 legend, which draws the two one-unit-only arms in the margins so a value that does not exist cannot be read as a low one). Limitations, pp 10–11, records that fastMRI+ knee covers 199 of 1,173 roster volumes, that age and sex are absent from every file, and that PI-CAI's published comparison is on a hidden testing cohort while the baseline is on the development set. **Missing from the manuscript:** the row counts dropped from the two fastMRI Prostate arms as validation or unrecognized split values (1,458 for DWI, 1,462 for T2) are in the released artifacts but are not stated in the text. |
| 13 | Image acquisition protocol | **NA** | No images were used — no pixel archive was downloaded for any benchmark (*Study Design*, p 4), so this study applied no acquisition protocol; each benchmark's own release paper, cited on pp 4–5, describes the acquisition of its images. |

## Methods — Reference Standard

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 14 | Definition of method(s) used to obtain the reference standard | **Y** | The reference standard is each benchmark's **own released label column**, used unmodified; *Benchmark Selection*, pp 4–5, names the label file for each benchmark and cites its release. Table 1, p 19, enumerates the six official label columns of the largest benchmark. |
| 15 | Rationale for choosing the reference standard | **Y** | *Benchmark Selection*, p 4, with the Introduction, p 3: the study measures what a benchmark's published label file certifies, so the released label is not a proxy for the reference standard — it *is* the object under audit, and substituting any other standard would change the question. |
| 16 | Source of reference standard annotations | **Y** | Third-party annotations from the benchmark releases, cited individually on pp 4–5; no annotation was performed by these authors. |
| 17 | Annotation of test set | **Y** | Test rows carry the same released labels as training rows; the split is the benchmark's own published train/test assignment where one exists and otherwise a five-fold subject-disjoint split scored out of fold (*Evaluation, Intervals, and Nulls*, p 5). |
| 18 | Measures of inter- and intrarater variability of features described by the annotators | **NA** | No annotation was performed for this study, and the public label files carry a single consensus label per row with no per-reader records, so inter- and intrarater variability cannot be computed from them. Where a release reports its own reader agreement, that is a property of the cited release and not a measurement of this study. |

## Methods — Data Partitions

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 19 | How data were assigned to partitions | **Y** | *Evaluation, Intervals, and Nulls*, p 5: a benchmark's own published train/test assignment where one exists, otherwise a five-fold split scored out of fold and pooled so every subject is tested once. Table 2, p 20, reports four independent routes through this step, including one with a different fold assignment and seed. |
| 20 | Level at which partitions are disjoint | **Y** | *Evaluation, Intervals, and Nulls*, p 5: **subject-disjoint throughout**. This is the item the paper exists to press on; the Introduction, p 3, distinguishes this study from the leakage literature precisely on it. Table 1's note, p 19, and Table 3's note, p 21, additionally report the constant-predictor floor that pooling out-of-fold predictions across folds of differing training prevalence introduces, per row, rather than assuming it is zero. |

## Methods — Testing Data

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 21 | Intended sample size | **NA** | No target sample size was set and no power calculation was performed, because there was nothing to sample: every eligible row of every eligible public label file was used, giving 752,802 slices from 18,938 patients on the largest benchmark (Abstract p 2; *Benchmark Selection*, p 4). The single subsample in the study, 1,500 patients, exists only as a replication route and is labeled as such in Table 2, p 20. |

## Methods — Model

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 22 | Detailed description of model | **Y** | *The Zero-Image Baseline Family*, p 5, describes all five models in full — a constant prevalence predictor, a 20-bin estimate of label rate given relative slice position, a volume-size score, a depth-limited classification tree over the label file's remaining non-image columns, and a combined position-plus-metadata tree — together with a fit-free variant, the negated absolute deviation of relative position from 0.5, that uses no training data. The tree's depth and its input columns for the one benchmark where the tree is the reported baseline are given in the Table 3 note, p 21 (depth 3; patient age, prostate-specific antigen level, center, scan year, and the tree splits on all four). |
| 23 | Software libraries, frameworks, and packages | **Y** | *Statistical Analysis*, p 6: Python 3.11 (Python Software Foundation, Wilmington, Del) with NumPy 1.26 and pandas 2.1, and, for the independent reimplementation only, scikit-learn 1.4. |
| 24 | Initialization of model parameters | **NA** | There are no parameters to initialize: four of the five models are closed-form summaries of the training rows (a prevalence, a 20-bin histogram, a slice count, and their combination) with no iterative optimization, and the fifth is a depth-limited tree fitted deterministically. The seed governing every stochastic step that does exist — fold assignment and bootstrap resampling — is recorded per run (*Statistical Analysis*, p 6). |
| 25 | Details of training approach | **Y** | *The Zero-Image Baseline Family*, p 5: each model is fit on training rows and scores test rows; the positional model's one hyperparameter, the bin count, is fixed at 20 and swept over 5/10/20/50 as a sensitivity analysis (Results, p 7; Table 1 note, p 19). No optimizer, learning rate, epoch count or augmentation exists to report. |
| 26 | Method of selecting the final model | **P** | Stated for both selections that occur. (a) The numerator of the comparison statistic is the **strongest of the five baselines** on each arm, defined in *The Comparison Statistic*, p 6, as "best zero-image baseline". (b) The DeepLesion arm carried into the figures was selected post hoc, and Results p 8 and the Table 3 note p 21 say so in those words. **Missing from the manuscript:** the maximum in (a) is taken over five correlated baselines on the same test data, so it is a winner's-curse estimate biased upward, and the interval printed with it is a bootstrap of the selected model rather than of the selection procedure. Neither the bias nor a nested-bootstrap correction is reported. |
| 27 | Ensembling technique | **NA** | No ensembling. Each reported value comes from a single model; the one model that combines inputs, the position-plus-metadata tree, is a single tree over both column groups rather than an ensemble, and it is reported against its own permutation null in the Table 1 note, p 19. |

## Methods — Evaluation

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 28 | Metrics of model performance | **Y** | *Evaluation, Intervals, and Nulls*, p 5: every score vector is read at **both** units — slice-level AUC, and patient-level AUC after mean aggregation, with a maximum-aggregated reading alongside. *The Comparison Statistic*, p 6, defines the trivial fraction, its chance anchors (0.5 for AUC, the majority-class rate for multiclass accuracy), and the conditions under which it is undefined or reported unclipped. The one arm scored on a detection metric, sensitivity at one false positive per scan, is reported with its random-score reference in Results p 8, and the Table 4 note, p 22, states that it is the only row in the primary set on that metric and that chance anchor and gives what the fraction would be against the alternative reference. |
| 29 | Statistical measures of significance and uncertainty | **Y** | *Statistical Analysis*, p 6: an estimation study; no significance test was pre-specified and no *P* values are reported. Every AUC carries a 95% percentile bootstrap CI resampling **subjects**, with the replicate count and seed recorded per run, degenerate replicates counted rather than dropped, and one reported range identified as a split-to-split spread rather than a bootstrap (Table 2 note, p 20). Table 3's note, p 21, gives the replicate count per row. |
| 30 | Robustness or sensitivity analysis | **Y** | Six of them, all reported. Bin sweep at 5/10/20/50 at both units (Results p 7; Table 1 note, p 19); a fit-free variant that uses no training data (p 5, p 19); a within-series label permutation null preserving prevalence, subject clustering and stack depth (p 5, Results p 7, Table 1 note); the constant-predictor floor measured per row rather than assumed to be 0.500 (Table 1 note p 19, Table 3 note p 21); four independent computational routes including a reimplementation sharing no code (Results p 7, Table 2 p 20); and a stack-depth-stratified re-reading of the patient-level result, at exact depth, in 5-slice strata and in deciles (Results pp 7–8, lower block of Table 1, p 19). |
| 31 | Methods for explainability or interpretability | **NA** | No post hoc explainability method was applied, and none is needed: the models are a 20-value lookup table, a slice count, a prevalence and a depth-3 tree, each fully inspectable as written. Where a result's mechanism was in question the paper measures it directly instead — the sub-chance patient-level reading is traced to stack depth entering through the mean operator, and the maximum-aggregated reading is shown to be degenerate (Results pp 7–8; Table 1 note, p 19). |
| 32 | Evaluation on internal data | **Y** | *Evaluation, Intervals, and Nulls*, p 5, with the results in Tables 1–3, pp 19–21. Every value is an out-of-sample reading: a benchmark's own held-out split where one exists, otherwise out-of-fold predictions on a subject-disjoint five-fold split. Apparent-versus-held-out training AUC is reported for the flagship model (0.738 against 0.738) in the Table 1 note, p 19. |
| 33 | Testing on external data | **NA** *as CLAIM means it* | No model is proposed for use anywhere, so there is no model to transfer to an external institution. The nearest analogue is reported and is arguably stronger for this study's purpose: the identical construction was run unchanged on **seven independently released benchmarks across eight label files**, from different institutions, modalities and organs, and every arm is reported including the one that does not fire (Results pp 8–9; Tables 3 and 4, pp 21–22). |
| 34 | Clinical trial registration | **NA** | Not a clinical trial and not a prospective study of any kind; no patient was recruited and no intervention was assigned (*Study Design*, p 4). Nothing here is registrable under the ICMJE clinical trial registration statement. |

## Results — Data

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 35 | Numbers of patients or examinations included and excluded | **P** | Included counts are given for every arm: 752,802 slices, 21,744 series and 18,938 patients for the largest benchmark (Abstract p 2; *Benchmark Selection*, p 4; Table 1 note, p 19); 665 patients for DeepLesion and 1,476 development cases for PI-CAI in the Table 3 note, p 21; 922 of 922 positive patients for Duke Breast, p 21; 199 of 1,173 roster volumes for fastMRI+ knee, Limitations, pp 10–11. Exclusion at the *benchmark* level is given with its measurement, p 4. **Missing from the manuscript:** a per-arm accounting of excluded rows — specifically the fastMRI Prostate rows dropped as validation or unrecognized split values, and how the 199 available fastMRI+ volumes were determined and whether they differ systematically from the remaining 974. |
| 36 | Demographic and clinical characteristics of cases in each partition and dataset | **NA** | **Age and sex are not distributed with these public label files and therefore cannot be reported.** This is stated three times rather than passed over: Abstract Materials and Methods, p 2; *Study Design*, p 4; and Limitations, pp 10–11, which records that no subgroup analysis was possible in consequence. The one benchmark whose file does carry clinical variables — age and prostate-specific antigen level — has them reported, because there they are inputs to a baseline (Results p 8; Table 3 note, p 21). |

## Results — Model Performance

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 37 | Performance metrics and measures of statistical uncertainty | **Y** | Results, pp 6–9, and Tables 1–4, pp 19–22. Every point estimate is printed with a 95% subject-clustered bootstrap interval at a stated replicate count, at three decimals throughout — the most the smallest replicate count supports (Table 2 note, p 20). Results reports no percentage of this study's own beyond the 95% interval level; the two coverage percentages are in the Figure 1 legend, p 17, and carry their numerators and denominators there (46.5%, 93 of 200; 91.5%, 183 of 200). |
| 38 | Estimates of diagnostic performance and their precision | **Y**, in the only sense that applies | No diagnostic model is proposed, so there is no diagnostic performance of this study's own to estimate; what is reported instead, with precision, is the performance of deliberately trivial baselines against published comparators. Discrimination with intervals: Tables 1 and 3, pp 19 and 21. The one operating-point estimate in the paper, sensitivity at one false positive per scan on LUNA16, is reported in Results p 8 with the note, p 22, that no interval is available for it. Sensitivity, specificity and predictive values of a proposed model are **not applicable** — there is no proposed model. |
| 39 | Failure analysis of incorrect results | **Y** | The paper's largest single analysis is a failure analysis of its own headline number. The sub-chance patient-level reading is traced to stack depth entering through the mean operator and dissolves when depth is held fixed (Results pp 7–8; lower block and note of Table 1, p 19); maximum aggregation is shown to be degenerate and is labeled so rather than quoted as a robustness check; the constant-predictor floor introduced by pooling across folds is measured per row and printed as a floor (Table 1 note p 19; Table 3 note p 21); the arm on which the measure does not fire is reported at equal prominence (Results p 8; Table 4, p 22); and the permutation null that disagrees between two fold draws is reported with both values and the pairing rule (Table 1 note, p 19). |

## Discussion

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 40 | Study limitations | **Y** | Limitations paragraph, pp 10–11, placed second-to-last as the journal requires. It carries: the construction is prior art; no peer-reviewed comparison is matched and the only matched rows rest on a preprint comparator; the headline patient-level value is not bin-robust, running 0.437 to 0.632 over 5 to 50 bins; two benchmarks are not pixel-free in the same clean sense; the PI-CAI comparison is across cohorts; age and sex are unavailable; and coverage is confined to benchmarks whose label fields were obtainable without a pixel data-use agreement. |
| 41 | Implications for practice, including intended use and/or clinical role | **Y** | Discussion, pp 9–10, in three parts: what a reader of an imaging AI paper should ask of a reported AUC; what benchmark publishers can do at no cost with three fields they already release; and what this protocol cannot do, namely detect shortcuts that live in the pixels. The scope of the clinical claim is stated rather than implied — no device summary or regulatory filing was examined, and nothing in these data speaks to the unit at which any cleared product was validated (p 10). |

## Other Information

| # | CLAIM 2024 item | Status | Where, or why not applicable |
|---|---|---|---|
| 42 | Provide a reference to the full study protocol or to additional technical details | **P — ACTION** | Additional technical detail is in the released archive, described in Acknowledgments, p 11. The one pre-specified analysis choice the paper relies on, 20 bins, is declared pre-specified "in a protocol that will be identified on acceptance" (Limitations, p 10) — deferred **only** because naming the protocol document would identify the authors under double-anonymized review. **Human action:** supply the protocol reference on acceptance, or now if the editors would prefer it. |
| 43 | Statement about the availability of software, trained model, and/or data | **P — ACTION** | Acknowledgments, p 11, states that the audit tool, the per-benchmark output payloads, the label-table preparation scripts, the independent reimplementation, and the trivial-fraction assembler are released under a Creative Commons license in a public repository and archived with a persistent identifier, that no benchmark's pixel data are redistributed, and that the fastMRI label files are available only by application under that initiative's own agreement. **The repository address and the commit identifier are withheld from the manuscript for anonymized review.** They are given in full on the Full Title Page and in the cover letter, and the cover letter tells the editor they can be written into Materials and Methods instead if that is preferred. This is a live conflict between the journal's Algorithm and Code Transparency policy and its double-anonymization instruction; it is flagged rather than papered over. |
| 44 | Sources of funding and other support; role of funders | **Y**, on the non-anonymized upload | Not in the manuscript, because a funding statement is identifying. Stated on the **Full Title Page**: no funding of any kind — no grant, fellowship, award, institutional allocation, industry contract, equipment loan, compute donation or in-kind support — and therefore no funder role in design, analysis, interpretation, writing or the decision to submit. Repeated in the cover letter under Declarations. |

---

## Two disclosures that CLAIM does not itemize and this journal requires

Recorded here so that a reader of this checklist alone does not have to reconstruct them.

- **Use of large language models.** Assistance was used in two ways — in drafting and editing
  the manuscript, and in **writing the analysis code that produces every number in the
  Results**. No reported value was generated by a language model: every number is a
  deterministic output of the released code run on a published label file, and the flagship
  estimate was reproduced by an independent reimplementation written separately from the
  primary implementation, the two agreeing to 0.003 AUC at both units on the full cohort. The
  disclosure appears in the Acknowledgments of the manuscript (p 11, anonymized), on the Full
  Title Page, and in the cover letter, in the same words; the tool name, version, maker and
  dates of access are on the two non-anonymized documents.
- **Third-party slice ordering for the flagship benchmark.** Every RSNA intracranial
  hemorrhage number rests on a slice ordering the official release does not contain. Its
  source, licence, join key, reconciliation and falsifiable verification test are in
  Materials and Methods, p 4 — not in a late clause of the Discussion — together with the
  fact that another challenge's release failed the same test and was excluded.

## What a human must still supply before this checklist is uploaded

1. **Item 1** — decide whether to leave a title that does not name a technology category, or
   retitle. Nothing else in the package depends on the choice.
2. **Items 42 and 43** — the protocol reference, the repository address and the commit
   identifier, all currently withheld for anonymization. Decide with the editor whether they
   go into Materials and Methods now or on acceptance.
3. **Items 12, 26 and 35** — three gaps this checklist declines to overstate: the fastMRI
   Prostate dropped-row counts, how the 199 available fastMRI+ knee volumes were determined,
   and the upward bias in a numerator that is a maximum over five correlated baselines chosen
   on the same test data. Each is a sentence of text in the manuscript, not new analysis.

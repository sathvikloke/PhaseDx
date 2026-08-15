# `paper/tex/rsna/` — the *Radiology: Artificial Intelligence* Original Research submission

This directory holds a **separate, self-contained, anonymised manuscript** built for
*Radiology: Artificial Intelligence* (RSNA) as an **Original Research** article.

It began as a cut-down of `paper/tex/main.tex`. It is no longer that. On 2026-08-13 the
estimator behind every flagship number changed — from a pooled out-of-fold five-fold split
to one frozen patient-disjoint holdout — and the paper was reorganised around the
two-unit reading and its stack-depth mechanism, with the cross-study comparison demoted to
descriptive. Every number here now comes from `revised_numbers.json`; the full manuscript
still carries the superseded estimator, so **numbers in this directory and numbers in
`paper/tex/main.tex` are no longer expected to agree**. `checklist.md` §14 is the record of
what changed and why.

**Nothing in `paper/tex/` outside this directory was modified.** `main.tex`,
`supplement.tex`, `refs.bib` and `figures/` remain the full manuscript, which is still
needed for a *Patterns* submission and for co-author outreach. The shared figure builder
`paper/tex/make_figures.py` was likewise left alone; this package builds its own figures.

---

## Files

| File | What it is |
|---|---|
| `main.tex` | The **single submission document**, in the order the journal requires: Abbreviated Title Page (anonymised) → Abstract → Main Body → Acknowledgments → References → Figure Legends → Tables (one per page). |
| `figures.tex` | The **separate figure document**. New submissions must combine all figures into one document with each legend immediately following its figure. |
| `refs.bib` | 25 entries; **19 are cited by `main.tex` and are the 19 that appear in the compiled bibliography.** The 6 uncited entries are shortcut/leakage citations left over from the prose that was cut and are inert (BibTeX emits only cited keys). Trimmed from the full `refs.bib`; the tool/archive entry was **deleted**, because it names the authors. |
| `figures/fig1_collapse.pdf` | **Figure 1.** A: two units, six labels. B: the stack-depth mechanism. C: the same four quantities over 24 holdouts. Built here. No identifying text. |
| `figures/fig2_unit_scatter.pdf` | **Figure 2.** Slice versus patient for every audited benchmark-arm, all eight DeepLesion arms plotted. Built here. No identifying text. |
| `figures/fig3_bin_agg_grid.pdf` | **Figure 3.** Bin count x aggregation grid. Built by `pipeline/audit_prep/frozen/rsna_bin_agg_grid.py`. No identifying text. |
| `figures/figS1_trivial_fraction.pdf` | **Figure S1, supplemental.** The descriptive cross-study comparison, demoted out of the main text. Built here. No identifying text. |
| `make_rsna_numbers.py` | Prints every number the manuscript uses, and every table body, straight out of `revised_numbers.json`. Run it before believing any figure in the text. Not submitted. |
| `make_rsna_figures.py` | Builds **all three** figures from `revised_numbers.json` and prints a source ledger. Run as `PHASEDX_METRIC_WORD=AUC venv/bin/python paper/tex/rsna/make_rsna_figures.py`. Not submitted. |
| `revised_numbers.json` | The number set every figure and every table reads from, with a 29-entry source ledger and a sha256 per input. **Carries `NOT_FOR_SUBMISSION`: it contains absolute local paths and must not be uploaded.** |
| `tables_appendix.tex` | Two optional supplemental tables: E1, the three-way classification of the non-image variables behind Table 4's one-letter key; E2, the mirror-provenance tests behind the Table 2 note. |
| `screening_table.md` | The working record behind Table 1. Not submitted. |
| `titlepage.tex` | The **Full Title Page**, *non*-anonymised, uploaded separately. 3 pp. |
| `cover_letter.tex` | The **cover letter** to the Editor, *non*-anonymised, uploaded separately. 4 pp. |
| `CLAIM_checklist.md` | The **CLAIM 2024 reporting checklist**, all 44 items, uploaded separately. Anonymised, because a reviewer may see it. |
| `checklist.md` | The submission checklist against the journal's own pages. Not submitted. |

Build with `tectonic main.tex`, `tectonic figures.tex`, `tectonic titlepage.tex` and
`tectonic cover_letter.tex` (compile in a scratch directory; there is no `pdflatex` on this
machine). All four compile clean — no errors, no overfull boxes, no undefined references.

### Still owed before submission

Every required file now exists. What is missing is information no artefact in this
repository establishes, and it was deliberately not invented; it is marked `TO SUPPLY` in
the file that needs it and listed in `checklist.md` §0. In short: the submission date and
the non-anonymised acknowledgments block. The language-model disclosure is complete and
identical in `main.tex`, `titlepage.tex` and `cover_letter.tex` — Claude (Opus 5),
Anthropic, accessed 2026-07-27 to 2026-08-12 — because the journal requires tool, version,
manufacturer and dates of access **at submission**, in both the manuscript and the cover
letter.

---

## Compliance, as built

Measured on the compiled PDF, 2026-08-13, after the frozen-holdout rewrite.

| Requirement | Limit | This manuscript |
|---|---|---|
| Body, Introduction → Discussion | ≤ 3000 words | **2978** — 22 tokens of margin, so recount after any edit |
| Structured abstract | ≤ 250 words, exactly four sections | **243** counting the heading word "Abstract" and the four section headings, 235 counting neither, **247** under the strictest comma-splitting convention; Purpose / Materials and Methods / Results / Conclusion |
| Summary Statement | ≤ 255 characters, one sentence, no abbreviations | **246 characters** |
| Key Points | ≤ 3 | 3 |
| References | ≤ 35 | 25 |
| Figures | ≤ 6 | 3 |
| Tables | ≤ 4 | 4 — at the cap, so anything added in revision must replace one |
| Abbreviations | ≤ 10 | 5 declared (AUC, CI, DWI, ICH, SD); IQR was removed with the median, SD added with the across-holdout spread |
| Title | no stated limit; journal median 13 words, Q3 15 | 15 words, colon structure |

Section budget actually used: Introduction 295 · Materials and Methods 1030 ·
Results 996 · Discussion 657. The compiled document is 21 double-spaced pages, with each of the four tables on a page of its own.
**The 400/800/1000/800 section split is not this journal's rule** — it is the
boilerplate of two sibling RSNA journals, and the *Radiology: AI* instructions
carry the same section-opening sentences with those caps deleted. The only
binding text limit is the 3,000-word total. Materials and Methods runs long for
two declared reasons: the third-party slice-ordering provenance and the
benchmark-eligibility criterion both live there, which is where reviewers asked
for them.

Other rules honoured: double-spaced, ragged right, 11 pt Times New Roman, pages not
numbered; Introduction is background only with hypothesis and purpose in its last
paragraph; Materials and Methods ends with statistical analysis naming software with
version and manufacturer; Results give numerators and denominators for every percentage;
Discussion opens by restating problem and primary results, has limitations as its
second-to-last paragraph, ends with a summary, and **cites no table or figure**; tables
and figures are **cited in ascending numeric order of first citation** (Table 1→4,
Fig 1→3); figure legends are **byte-identical** between `main.tex` and `figures.tex`; no
self-evaluative language ("novel", "unique"), no priority claim, and "significant" does
not appear.

---

## What was cut, and why

The scope decision was made upstream: this version is **the RSNA intracranial hemorrhage
two-unit reading with its stack-depth mechanism, plus a descriptive cross-study comparison
over the seven audited benchmarks**, and nothing else.

### 1. The pre-specified prevalence screen — **excluded entirely, pending a disclosure rework**

The full manuscript reports a protocol-frozen random-sample screen of a 9979-record
PubMed frame with four independent screeners, and uses it to state how often this
literature reports a zero-image baseline. **None of that appears here in any form** — not
as a result, not as a citation to our own screen, not as a motivating sentence, not in the
abstract, not as a limitation.

The reason is a disclosure defect, not a numerical one. An audit established that the four
"independent screeners" were language-model agents and that this is nowhere disclosed; the
supplement's own timeline shows batches submitted 49–55 minutes after protocol freeze at
36–37 records each, roughly 82–93 seconds per record against a protocol that requires
fourteen prescribed full-text searches plus a full Methods read for every record. That arm
is being reworked separately, and it must not be re-introduced into this manuscript until
that rework is finished and the disclosure is written.

The journal's own policy makes this sharper rather than softer: RSNA requires authors to
describe, in both the cover letter **and the manuscript**, how AI or AI-assisted
technologies were used **in the study**, not merely in drafting — with tool name, version,
manufacturer and date of access. Dropping the screen removes the reported-result problem;
it does not remove that duty for anything language-model-touched that remains in scope, or
for drafting assistance. A disclosure paragraph is therefore present in the
Acknowledgments — and it is the broad one: it leads with the verifiable claim that every
number is a deterministic output of released code and that **no reported value was
generated by a language model**, then states that assistance was used **in drafting and
editing the manuscript and in implementing that analysis code**. ``Implementing'' rather
than ``writing'': the protocol was pre-specified, the code implements it, and the earlier
phrasing implied a language model produced the numbers. The same wording is in
`cover_letter.tex` and in `titlepage.tex`, and the tool is **named in `main.tex` too, at
submission** — Claude (Opus 5), Anthropic, accessed 2026-07-27 to 2026-08-14. The manuscript previously deferred the name to acceptance, which
was a policy violation: RSNA requires name, version, manufacturer and dates of access at
submission, in both the manuscript and the cover letter, and naming the tool identifies
nobody. The argument that carries the weight is identical in all three and rests where it
belongs: no value is model-generated, every number is a deterministic output of the
released code run on a published label file, and the flagship estimate was reproduced by a
second implementation sharing no code with the first, whose family of estimates agrees to
0.0003 AUC per slice and 0.005 per patient.

Where the full manuscript motivates the work with "almost nobody checks", this version
motivates it from the published literature instead — Kapoor & Narayanan, Varoquaux &
Cheplygina, Roberts et al, Maier-Hein et al — and never from our own unreported screen.

### 2. The rank-inversion analysis — cut for scope

"Does the evaluation unit change which method wins?" returned a null across 447 pairs on
21 method configurations. It is a defensible result but it is a second study, it costs
roughly 700 words plus a table and a figure, and its own limitation is that the 21
configurations were trained by one group with one pipeline. Nothing in this version
depends on it.

### 3. The k-space phase case study — cut as a study

The five NYU fastMRI worked-example cohorts, the 102 training runs and 456 control runs,
the background-only control, the reconstruction-fidelity validation, and the nine-criterion
falsification suite are all out. The fastMRI **label files** stay, because they are two of
the seven label files in the trivial-fraction table; the fastMRI **imaging study** does
not. In the end the falsification suite did not earn even a sentence — the word budget went
to the two in-scope results instead.

The NYU fastMRI Data Sharing Agreement obligations survive the cut and are honoured:
the literal string "NYU fastMRI" in the abstract (¶6a), the mandated acknowledgement
paragraph in Materials and Methods (¶6c), and citations to Knoll et al and Zbontar et al
(¶6b). The comment block guarding that paragraph is intact — do not delete it.

### 4. Cut for length, from surviving material

- The FDA 510(k) scoping check (30 cleared radiology AI summaries).
- The DeepLesion window/level result (0.911 from the header column alone) and the
  release-batch confound from the worked-example cohorts.
- The position-stratified AUROC remedy and the seven-rule reporting protocol table.
- The RSNA-STR Pulmonary Embolism label-file measurement as a standalone result; only
  the general point that a benchmark can leave its slice unlocatable survives, attached to
  the RSNA ICH slice-ordering recovery.
- The flow figure, the rank-inversion figure, the case-study figure, and the
  qualitative-input figure. All three figures in this package are now built here, by
  `make_rsna_figures.py`, from `revised_numbers.json`; none is copied from the full
  manuscript, because the full manuscript still carries the superseded estimator.
- The naive-versus-clustered interval-width panel and its simulation coverage figures
  (old Figure 1B). Its numbers were computed under the superseded pooled estimator and the
  simulation was not re-run; that panel slot went to the stack-depth mechanism, which is
  the reviewer-facing result. See `checklist.md` §14.
- Per-benchmark detail on `--self-test`, the JSON payload format, the dependency argument,
  and the IRB paragraph's longer form.

### 5. Corrected relative to the full manuscript

The full manuscript says **two** benchmark-arms do not fire. Under the **locked** primary
baseline that is now the right count, for a reason the full manuscript does not give.
LUNA16 is at −0.002. PI-CAI's *locked positional* baseline is exactly 0.500 — the correct
registration of "inapplicable", since that label file has no slice index — so its primary
trivial fraction is exactly **0.000**. The 0.692 metadata tree that used to carry that row
was selected on the test data and two of its four inputs are clinical predictors; it
survives only as a labelled secondary reading (0.692; 95% CI: 0.626, 0.755) and supports no
acquisition-confounding inference. That is the one row the baseline lock moves anywhere in
the package.

**Second correction, made during the major revision.** That PI-CAI baseline is **not** an
acquisition-metadata model and this version no longer calls it one anywhere. It is a
depth-3 tree over the four non-image columns of that label file — patient age,
prostate-specific antigen level, centre and scan year — and it splits on all four. Age and
prostate-specific antigen are standard clinical predictors of clinically significant
prostate cancer, so the row is a **clinical-variable baseline** and supports no inference
about acquisition confounding; the manuscript says so and draws none. Refitting on centre
and scan year alone is not possible from the retained output, and the manuscript says that
too; read one column at a time those two reach 0.547 and 0.552 per patient against 0.636
and 0.638 for the clinical pair.

---

## Anonymisation

Double-anonymised review; the author anonymises. This manuscript contains no institution,
no funding statement, no author name or initials, no repository address, no archive DOI, no
software name, no repository file path, and no self-referential phrasing. `refs.bib` was
rebuilt from scratch for this directory and the entry that names the authors, the software
and the archive was dropped.

**The one unresolved conflict** is between the journal's anonymisation instruction and its
Algorithm and Code Transparency policy, which requires a link to the code and the unique
revision identifier *in the Materials and Methods*. RSNA publishes no rule reconciling the
two. The resolution taken here: Materials and Methods states that the tool is released
under a Creative Commons licence and that its address and commit identifier are withheld
for anonymised review and will be supplied on acceptance; the Acknowledgments repeat this
and offer both to the editors on request. Flag it in the cover letter.

The data-sharing statement lives in the Acknowledgments and promises release on
acceptance without naming the repository. It also records that no benchmark's pixel data
are redistributed and that the fastMRI label files are available only by application under
that initiative's own agreement.

---

## Rules that governed every number here

- Every value is traceable to `revised_numbers.json`, which carries a source ledger and a
  sha256 per input, or to `pipeline_out/trivial_baselines/`. Where a number and an artefact
  disagreed, the artefact won — including once where the artefact's own prose note
  disagreed with the artefact's own numbers, recorded in `checklist.md` §14.
- Every number in `main.tex` is machine-checked against `revised_numbers.json`: 162
  assertions, all passing.
- A null is never rounded toward significance: −0.002 stays −0.002, 0.500 stays 0.500.
- No claim that a model "learned nothing". Every claim is about an evaluation protocol.
- The only matched rows have a preprint comparator, and they are labelled as such every
  time they are mentioned.


## Revision of 2026-08-14

Four arms previously carried on the superseded pooled out-of-fold estimator were
resolved. Two -- Duke Breast and LUNA16 -- had their label tables rebuilt from the
public source releases (`Annotation_Boxes.xlsx` plus the TCIA `getSeries`
metadata; `candidates_V2.zip` from the LUNA16 Zenodo record) and verified
byte-identical against the SHA-256 recorded by the original run, then re-scored by
`pipeline/audit_prep/frozen/frozen_arm_holdout.py`. Two fastMRI+ knee arms could
not be rebuilt on their original cohort -- 155 of the 199 roster volumes remain
locally -- and are withdrawn. **No number in this submission now rests on the
pooled estimator, and the constant predictor is exactly 0.500 on every retained
arm.**

The primary estimate moved from the single frozen holdout to the mean over the
24-holdout family, because the pre-registered draw sits at the boundary of that
family on the patient-level and depth-conditioned readings. That draw is still
reported, as a pre-registered reference point with a bootstrap CI.

The cross-study margin ratio was demoted out of the abstract and Key Points and
its figure moved to `supplement.tex` as Figure S1; the bin x aggregation grid was
promoted to Figure 3. `figures.tex` is now GENERATED from main.tex by
`make_figures_tex.py`, which enforces the journal's rule that a legend follow its
figure immediately -- an earlier revision wrapped Figure 4 in a `[p]` float, which
separated the two.

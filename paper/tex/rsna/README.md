# `paper/tex/rsna/` — the *Radiology: Artificial Intelligence* Original Research submission

This directory holds a **separate, self-contained, anonymised manuscript** built for
*Radiology: Artificial Intelligence* (RSNA) as an **Original Research** article.

It is **derived from `paper/tex/main.tex` by cutting**, not by rewriting. Wherever a
sentence fit the new scope and the new word budget, its exact wording was preserved,
because that prose has been through repeated audit and its numbers are traceable to
artefacts.

**Nothing in `paper/tex/` outside this directory was modified.** `main.tex`,
`supplement.tex`, `refs.bib` and `figures/` remain the full manuscript, which is still
needed for a *Patterns* submission and for co-author outreach.

---

## Files

| File | What it is |
|---|---|
| `main.tex` | The **single submission document**, in the order the journal requires: Abbreviated Title Page (anonymised) → Abstract → Main Body → Acknowledgments → References → Figure Legends → Tables (one per page). |
| `figures.tex` | The **separate figure document**. New submissions must combine all figures into one document with each legend immediately following its figure. |
| `refs.bib` | 25 entries — only those cited by `main.tex`. Trimmed from the full `refs.bib`; the tool/archive entry was **deleted**, because it names the authors. |
| `figures/fig1_collapse.pdf` | Copied unchanged from `paper/tex/figures/`. Contains no identifying text. |
| `figures/fig2_trivial_fraction.pdf` | Copied unchanged from `paper/tex/figures/`. Contains no identifying text. |

Build with `tectonic main.tex` and `tectonic figures.tex` (compile in a scratch
directory; there is no `pdflatex` on this machine). Both compile clean — no overfull
boxes, no undefined references.

### Not in this directory, and still owed before submission

- **Full Title Page** — a separate, *non*-anonymised file: title, all authors with degrees
  and superscript-numbered affiliations, the institution where the work originated, the
  corresponding author's phone/email/postal address, funding, manuscript type, word count,
  unanonymised acknowledgments, data-sharing statement.
- **Cover letter** — must carry the title and full author list, any subject overlap with
  previously published work, conflicts/industry support, confirmation of sole submission,
  and the large-language-model disclosure.
- **Reporting checklist** — required at first submission or the paper is returned. CLAIM
  is the designated default for AI-in-medical-imaging manuscripts; mark inapplicable items
  "not applicable".

---

## Compliance, as built

| Requirement | Limit | This manuscript |
|---|---|---|
| Body, Introduction → Discussion | ≤ 3000 words | **2995** (2955 excluding section headings) |
| Structured abstract | ≤ 250 words, exactly four sections | **249**, Purpose / Materials and Methods / Results / Conclusion |
| Summary Statement | ≤ 255 characters, one sentence, no abbreviations | **228 characters** |
| Key Points | ≤ 3 | 3 |
| References | ≤ 35 | 25 |
| Figures | ≤ 6 | 2 |
| Tables | ≤ 4 | 4 |
| Abbreviations | ≤ 10 | 5 (AUC, CI, DWI, ICH, IQR) |
| Title | no stated limit; journal median 13 words, Q3 15 | 15 words, colon structure (used by ~half of the journal's titles) |

Section budget actually used: Introduction 415 · Materials and Methods 744 ·
Results 1037 · Discussion 799. The compiled document is 21 double-spaced pages.

Other rules honoured: double-spaced, ragged right, 11 pt Times New Roman, pages not
numbered; Introduction is background only with hypothesis and purpose in its last
paragraph; Materials and Methods ends with statistical analysis naming software with
version and manufacturer; Results give numerators and denominators for every percentage;
Discussion opens by restating problem and primary results, has limitations as its
second-to-last paragraph, ends with a summary, and **cites no table or figure**; no
self-evaluative language ("novel", "unique"), no priority claim, and "significant" does
not appear.

---

## What was cut, and why

The scope decision was made upstream: this version is **the RSNA intracranial hemorrhage
unit collapse plus the seven-benchmark trivial-fraction table**, and nothing else.

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
Acknowledgments.

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
the eight label files in the trivial-fraction table; the fastMRI **imaging study** does
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
- The PRISMA flow figure, the rank-inversion figure, the case-study figure, and the
  qualitative-input figure. Two figures remain.
- Per-benchmark detail on `--self-test`, the JSON payload format, the dependency argument,
  and the IRB paragraph's longer form.

### 5. Corrected relative to the full manuscript

The full manuscript says **two** benchmark-arms do not fire. That is wrong on this
statistic and is fixed here. **Only LUNA16 is at zero** (−0.002). PI-CAI's *positional*
baseline is exactly 0.500 — the correct registration of "inapplicable", since that label
file has no slice index — but its strongest zero-image baseline runs over acquisition
metadata and reaches 0.692, giving a trivial fraction of 0.467, which is mid-range. PI-CAI
is therefore **not** a null arm and is not described as one anywhere in this version.

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

- Every value is traceable to an artefact under `pipeline_out/trivial_baselines/` or
  `paper/trivial_fraction_distribution.json`. Where a number and an artefact disagreed,
  the artefact won.
- A null is never rounded toward significance: −0.002 stays −0.002, 0.500 stays 0.500.
- No claim that a model "learned nothing". Every claim is about an evaluation protocol.
- The only matched rows have a preprint comparator, and they are labelled as such every
  time they are mentioned.

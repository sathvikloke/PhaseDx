# Submission checklist — *Radiology: Artificial Intelligence*, Original Research

Tick through this before uploading anything to ScholarOne
(<https://mc.manuscriptcentral.com/rad-ai>).

Every requirement below was read from the journal's own pages on **2026-08-11**:
`pubs.rsna.org/page/ai/author-instructions` (cited **[AI]**) and
`pubs.rsna.org/page/policies` (cited **[POL]**). Both 403 on direct fetch; they
were retrieved through the `r.jina.ai` reader proxy. The instructions page's own
"What's New" latest entry is January 2026.

Status key: **OK** = done and verified · **ACTION** = a human must do something ·
**RISK** = done but could bounce, decide deliberately.

---

## 0. Blocking items — do these first

| # | Item | Why |
|---|---|---|
| 1 | **Recount the body and get it under 3,000 words.** See §3. | Measured 3,041 words in the compiled PDF. The README's figure of 2,995 does not survive a whitespace-token count of the rendered text. |
| 2 | **Complete the CLAIM checklist.** Not written yet. | *"Your paper will be sent back if this checklist is not included upon first submission."* **[AI]** |
| 3 | **Fill every `TO SUPPLY` marker** in `titlepage.tex` and `cover_letter.tex`. | Middle initials, telephone, ORCID iDs, language-model tool names/versions/dates, and who performed which analysis are not recorded anywhere in this repository and were deliberately not invented. |
| 4 | **Decide the abstract word count convention.** See §3. | 246 words without the section headings, 253 with. Limit is 250. |

---

## 1. Files to upload

**[AI]** splits the submission into separate uploads plus one combined manuscript file.

### Uploaded separately

| Required file | Built here | Status |
|---|---|---|
| Cover Letter | `cover_letter.tex` → 4 pp | **ACTION** — 3 `TO SUPPLY` markers |
| Full Title Page | `titlepage.tex` → 3 pp | **ACTION** — 11 `TO SUPPLY` markers |
| Checklist (CLAIM / STARD / CONSORT / PRISMA / STROBE) | — | **ACTION** — not written |
| Figures, combined into **one** document, each legend immediately after its figure | `figures.tex` → 2 pp | **OK** |
| Supplemental Materials | — | none submitted; optional |

### Uploaded as one single document — `main.tex` → 21 pp

Order required by **[AI]**, and the order in the file:

1. Abbreviated Title Page (anonymized) — **OK**
2. Abstract — **OK**
3. Main Body — **OK**
4. Acknowledgments (anonymized) — **OK**
5. References — **OK**
6. Figure Legends — **OK**
7. Tables, embedded, one per page — **OK**

All four documents compile under `tectonic` with no warnings, no overfull or
underfull boxes and no undefined references. Build each in a scratch directory:
`tectonic -X compile <file>.tex --outdir <scratch>`.

---

## 2. Hard limits

All from the Original Research block **[AI]**.

| Limit | Requirement | Measured | Status |
|---|---|---|---|
| Word count, Introduction→Discussion | ≤ 3,000 | **3,041** (see §3) | **ACTION** |
| Abstract | ≤ 250 words, structured, exactly 4 sections | 246 / 253 (see §3); sections correct | **ACTION** |
| References | ≤ 35 | **25** | OK |
| Figures (images, charts, graphs) | ≤ 6 | **2** | OK |
| Tables | ≤ 4 | **4** — at the cap | OK |
| Key Points | ≤ 3 | **3** | OK |
| Summary Statement | ≤ 255 characters | **228**, one sentence | OK |
| Abbreviations | ≤ 10 | 5 declared, 13 acronym forms appear (see §6) | **RISK** |
| Authors | ≤ 2 first authors; exactly one corresponding/senior author | 4 authors, 1 first, 1 corresponding | OK |

---

## 3. The two counts that are not settled

**Body.** Counting whitespace-separated tokens in the text extracted from the
compiled PDF, Introduction through Discussion is **3,041 words**. Removing the
9 standalone bracketed citation markers gives 3,032. Counting from the LaTeX
source instead — collapsing the displayed equation to one token and inline
maths to a few — gives 3,018 with section headings and 2,976 without. The
README in this directory reports 2,995; that number is inside the band but is
not robust, and two of the four conventions put the paper over the limit.

- **ACTION.** Cut roughly 45–60 words from the body so that it clears 3,000 on
  the strictest reading, then recount **on the file you actually upload, in the
  format you upload it in**, and write that number on the Full Title Page.
- Section budget as it stands (LaTeX-source count, headings included):
  Introduction 419 · Materials and Methods 747 · Results 1,043 · Discussion 809.
  A 2019 capture of the journal's preparation checklist suggested 400 / 800 /
  1000 / 800; that page is date-limited and is not restated in the current
  instructions, but it is a sane budget and the Introduction and Results are
  each slightly over it.

**Abstract.** 253 words including the word "Abstract" and the four section
headings; 246 without. The limit is 250 **[AI]**.

- **ACTION.** ScholarOne asks you to paste the abstract into a box at Step 1.
  Paste it, read the character/word counter the system shows you, and cut if it
  objects. Cutting 4 words now costs nothing and removes the question.

---

## 4. Abbreviated Title Page — the four required items

**[AI]** lists exactly four. All four are at the top of `main.tex`, anonymized.

- [x] **Title.** *What a Slice-Level Benchmark Certifies without the Pixels: An
      Audit of Seven Public Imaging Datasets.* 15 words.
      Note: Radiology: AI states **no title length limit** — the 15-word cap
      belongs to flagship *Radiology*, a different journal. 15 words sits at the
      third quartile of 120 Radiology: AI titles from 2025–2026 (median 13), and
      the colon structure is used by roughly half of them. Nothing to fix.
- [x] **Article Type.** Original Research.
- [x] **Summary Statement.** 228 characters, one sentence, boldface, no
      abbreviations. The Original Research block says "1–2 sentences" while the
      Abbreviated Title Page block says "a single sentence"; the stricter reading
      is used. **[AI]** also forbids abbreviations in the Summary Statement —
      there are none.
- [x] **Key Points.** 3, each carrying summary data, none repeating the Summary
      Statement verbatim.
  - **RISK.** **[AI]** asks you to "avoid using vague language and abbreviations
    in the Key Points. Obvious abbreviations like CT and MRI are fine." Key Point 1
    uses **CI**, Key Point 2 uses **IQR**. Neither is on the journal's "obvious"
    list. Cheap fix: write "95% confidence interval" and "interquartile range"
    once each — it costs about 6 words and removes an easy desk-edit comment.
  - **RISK.** Key Point 2 and the Summary Statement both carry the "about half of
    the published margin" finding. Not a verbatim repeat, but close enough that a
    reader may notice. Consider re-pointing Key Point 2 at the *tightness* of the
    distribution (8 of 9 arms between 0.30 and 0.70) instead.

---

## 5. Abstract structure

**[AI]** requires exactly four sections, in this order, and **no Background
section** — Background is a flagship-*Radiology* item and must not be imported.

- [x] Purpose / Materials and Methods / Results / Conclusion, in that order, no fifth heading.
- [x] Materials and Methods carries all seven required elements in order:
  1. **Retrospective** — stated.
  2. **Date range of study** — benchmarks released 2016–2024; analyses run July 2026.
  3. **Number of patients, with age and sex if appropriate** — 752,802 slices from
     18,938 patients; *"age and sex are not distributed."* Stating why they are
     absent, rather than omitting them, is what keeps this compliant.
  4. **Groups including controls** — five pixel-blind models named, with a
     within-series permutation null identified as the control.
  5. **Procedures performed** — subject-disjoint five-fold split, read at both units.
  6. **Specifics of evaluation paralleling Results** — trivial fraction defined.
  7. **One sentence on statistical analyses** — bootstrap CIs resampling subjects;
     no hypothesis testing.
- [x] Conclusion derives directly from the results, addresses the stated purpose,
      and does not elaborate on significance or implications.
- [x] Contains the literal string **"NYU fastMRI"** — required verbatim in any
      published abstract by the fastMRI Data Sharing Agreement ¶6(a). **Do not cut it.**

---

## 6. Main body — structural and prose rules

| Rule **[AI]** | Status |
|---|---|
| Headings are Introduction, Materials and Methods, Results, Discussion | OK |
| Introduction is background and why the study was done — **"No extensive literature review"** | **RISK** — the second paragraph runs through seven citations. It is compressed and every citation earns its place, but it is the longest stretch in the paper that could read as a review. If you need to cut words (§3), cut here first. |
| Introduction's final paragraph states the hypothesis and purpose | OK — both, explicitly |
| Materials and Methods includes every item that appears in Results | OK |
| M&M states retrospective, date range, subject group, selection criteria | OK — inclusion rule is the four-field test |
| M&M first paragraph addresses IRB approval and consent | OK — states approval not required, with the regulatory basis (45 CFR 46.102(e)) |
| M&M last paragraph: statistical methods, software **name + version + manufacturer**, and the *P* value used for significance | OK — Python 3.11 (Python Software Foundation, Wilmington, Del), NumPy 1.26, pandas 2.1, scikit-learn 1.4; states that no significance test was pre-specified and no *P* values are reported |
| Results give **numerators and denominators for every percentage** | OK — 46.5% (93 of 200), 91.5% (183 of 200); the only two percentages in Results |
| Discussion ¶1 restates the problem and the primary results | OK |
| Discussion **second-to-last paragraph is Limitations** | OK — Limitations is second-to-last, summary is last |
| Discussion final paragraph is a summary | OK |
| **Do not cite tables or figures in the Discussion** | OK — verified, zero Table/Figure references in the Discussion |
| Avoid self-evaluation: "novel," "unique," "ground-breaking" | OK — zero occurrences |
| Avoid claiming priority | OK — the Introduction concedes prior art before making any claim |
| Reserve "significant" for statistical significance | OK — the word does not appear anywhere |
| Text clear to a radiologist outside the specialty | Judgement call — reread once with that reader in mind |

### Abbreviations — the one number that is arguable

Five are declared: AUC, CI, DWI, ICH, IQR. The body additionally contains
**RSNA, RSNA-STR, PI-CAI, NYU, LUNA16, MRI, COVID-19, CFR, PMC**.

Most of those are proper names of datasets, organisations or diseases, and are
very unlikely to be counted against the limit of 10. **CFR** is the exception:
it is a plain abbreviation, it is used once, and it is not in the declared list.

- **ACTION (cheap).** Either add CFR to the abbreviation list — taking it to 6 of
  10 — or spell out the regulation. Do not leave it undeclared.

---

## 7. Full Title Page — `titlepage.tex`

The nine items **[AI]** requires, in the order it lists them:

- [x] Title of the manuscript
- [~] First and last names, **middle initials**, **academic degrees**, institutions;
      affiliations by superscript number in author order — **ACTION**: middle
      initials not on record; the degree field is empty because no author holds a
      degree, which is stated explicitly rather than left blank silently
- [x] Name and street address of the institution from which the work originated
- [~] Corresponding author: **telephone**, e-mail, complete postal address —
      **ACTION**: telephone not on record
- [x] Funding information — none, stated exhaustively
- [x] Manuscript Type — Original Research
- [~] Word Count for Text — **ACTION**, see §3
- [~] Unanonymized acknowledgments — **ACTION**: the language-model tool list needs
      names, versions, manufacturer and dates; anyone else acknowledged must send
      written permission, because being named signals agreement with the data and
      conclusions **[AI]**
- [x] **Data sharing statement** — required *on the full title page* **[POL]**, and
      one of five statements must appear **verbatim**. The first is used: *"Data
      generated by the authors or analyzed during the study are available at:"*
      followed by the Zenodo concept DOI, license, and the reuse restrictions the
      policy also asks for

Also on the page, required by **[POL]** rather than **[AI]**:

- [x] Conflicts of interest — all four authors declare none; each will file an
      ICMJE disclosure form
- [~] **ORCID iDs** — **ACTION**: the submitting author's ScholarOne account must be
      linked to an ORCID iD, and all authors of accepted manuscripts must link theirs

---

## 8. Cover letter — `cover_letter.tex`

**[AI]** requires five items; **[POL]** adds three more.

- [x] Title of the manuscript
- [x] Complete author list
- [x] **Subject overlap with previously published works (REQUIRED)** — none by these
      authors; the *patients* were all previously reported by the benchmark
      releases and the comparator papers, which is stated plainly and cited in M&M
      as **[AI]** also requires
- [x] **Conflict of interest / industry support (REQUIRED)** — none
- [x] **Confirmation of sole submission (REQUIRED)**
- [~] **Which author(s) had control of the data and performed the analyses
      (REQUIRED, [POL] Conflicts of Interest)** — **ACTION**: partly filled. Say
      explicitly whether the independent reimplementation in Table 2 was written by
      a different author. If it was the same person, say so — an independent
      reimplementation by one author is a weaker check and the editor is entitled
      to see that before a reviewer finds it
- [x] **Prior posting disclosure with DOI and licensing terms** ([POL] Preprint
      Servers) — §1 of the letter; see §9 below
- [~] **Language-model use, in the cover letter as well as the manuscript**
      ([POL] LLM guidelines) — **ACTION**: tool names, versions, manufacturer and
      dates of access. The list must match the Acknowledgments word for word
- n/a Dual first authorship / fast-track explanation — neither requested

---

## 9. Policies that bite on this particular paper

### Prior posting — disclosed, unresolved by design

**[POL]**: *"Authors should disclose details of preprint posting, including DOI
and licensing terms, upon submission."* And on redundant publication: *"authors
should include a letter informing the Editor of any potential overlap with other
already published material … and should also state how the manuscript submitted
to the journal differs substantially from this other material. Copies of such
material must be provided."*

- [x] The Zenodo deposit is disclosed in the cover letter with **concept DOI
      `10.5281/zenodo.21814952`**, version DOI `10.5281/zenodo.21814953`, and the
      **CC BY 4.0** license. Both DOIs were confirmed to resolve on 2026-08-11.
- [x] The letter states how this manuscript differs from the longer deposited one
      and offers to send the deposit or any part of it.
- [x] The letter commits to updating the Zenodo record with the publication
      reference and a link on acceptance, as **[POL]** requires.
- **RISK, and a human should decide this, not a script.** The deposit is public
  and its landing page describes material this submission does not report. An
  editor who follows the DOI will see it. The cover letter warns that the archive
  is broader than the submission, without characterising the extra material. If
  you would rather the editor have the specifics up front, that is a defensible
  choice — but make it deliberately, and make it before submission, not after a
  reviewer asks.

### Algorithm and code transparency — a genuine conflict, flagged not hidden

**[POL]** asks Materials and Methods to carry (1) a link to the code and (2) the
unique revision identifier. Double-anonymized review forbids both.

- [x] M&M states the tool is released under a Creative Commons license with the
      address and commit withheld for review and supplied on acceptance.
- [x] Acknowledgments repeat this and offer both to the editors on request.
- [x] The cover letter gives the editor the identifiers immediately and offers to
      write them into the manuscript instead if that is preferred.
- Note the policy's own wording is *"deposited in a publicly accessible repository
  **upon publication**"* — the deposit already exists, so only the in-text link is
  deferred.

### Large language models

**[POL]**: disclosure is required for use **in the study and/or manuscript
preparation**, in **both** the cover letter and the manuscript, with **name and
version of the tool, date of access, and manufacturer/creator**.

- [x] Routed to Acknowledgments in `main.tex` (anonymized), Acknowledgments in
      `titlepage.tex` (named), and the cover letter.
- **ACTION.** The specifics are unfilled in all three places. They must be
  identical in all three. The disclosure covers **study** use as well as drafting —
  read the policy's wording before deciding what belongs in the list.

### Reporting checklist

**[AI]**: required at first submission for Original Research involving human
subjects; the paper is returned without it.

- **ACTION.** Complete **CLAIM** — the designated default for AI manuscripts in
  medical imaging (`10.1148/ryai.2020200029`; 2024 updates at
  `pubs.rsna.org/page/ai/claim`). Mark inapplicable items "not applicable" **[AI]**.
  Expect many: this study audits an evaluation protocol rather than developing a
  diagnostic model, so most model-development and clinical-deployment items do not
  apply. Give a reason for each N/A.
- The cover letter already tells the editor this and offers STROBE instead if the
  editors prefer it for a study of this shape.

### Authorship

- [x] Four authors, one first author, **one** corresponding author. **[POL]**: *"The
      Radiology suite of journals does not allow multiple corresponding authors."*
      This is a cardinality rule, not a seniority rule — nothing in **[AI]**,
      **[POL]** or the ICMJE criteria RSNA adopts requires a faculty member or
      supervisor on the byline.
- [x] Author count is far below the consent thresholds (**[AI]** says >40 needs
      Editor consent; **[POL]** still says >20 — inconsistent, irrelevant at n=4).
- **ACTION (optional, cheap insurance).** One line to `ryai-editor@rsna.org` before
  submitting, confirming that an all-student author list with one designated
  corresponding author is acceptable. Nothing in the written rules forbids it; an
  unwritten editorial expectation cannot be verified from the public pages.
- **ACTION.** **[POL]** requires **CRediT** contributor roles for certain manuscript
  types (paragraph added October 2025). Agree the roles among the four authors
  before submission; ScholarOne will ask.

### fastMRI Data Sharing Agreement — non-negotiable, and already satisfied

- [x] ¶6(a): the literal string **"NYU fastMRI"** appears in the abstract.
- [x] ¶6(c): the mandated acknowledgment paragraph is in Materials and Methods,
      unabridged. It names no author and does not breach anonymization.
- [x] ¶6(b): Knoll et al and Zbontar et al are cited.
- [x] No fastMRI-derived image, cache file or slice is in `figures/`, and none is
      redistributed anywhere in the package.

---

## 10. Format

All from **[AI]** unless noted.

- [x] PDF is an accepted format for initial submission. LaTeX `.tex` is also
      accepted at initial submission, but *"LaTeX files (.tex) cannot be used for
      production of accepted articles"* — so a `.docx` will be needed if the paper
      is accepted. As of January 2026 `.doc` is no longer accepted.
- [x] Double-spaced — `\doublespacing`. Tables are set single-spaced, which is
      conventional.
- [x] Not right-justified — `\raggedright`.
- [x] 11 pt Times New Roman — `mathptmx`.
- [x] **Pages not numbered** — `\pagestyle{empty}`.
- [x] Tables embedded in the manuscript document, one per page.
- [x] Figures in their own combined document, each legend immediately following
      its figure.

---

## 11. Anonymization — verified on the compiled PDF, not on the source

Text was extracted from `main.pdf` and searched. Zero occurrences of: any
author surname as an author, the institution names, the city names, the
repository name, the software name, "GitHub", "Zenodo", `10.5281`, any DOI of
ours, any file path, and any reference to the excluded arm of the larger study.

- One benign hit: **"Johnson"** appears twice, both times as a third-party
  reference author (Patricia Johnson, in the fastMRI citations). A reviewer
  cannot infer authorship from a common surname in a reference list. No action.
- Figure PDFs carry only `Matplotlib v3.10.8` in their Creator/Producer metadata.
  No author, institution, filename or path. No action.
- **ACTION.** Re-run this check on the final PDF after you make the word-count
  cuts in §3. Anonymization is a property of the file you upload, not of the file
  you checked last week.

---

## 12. Files in this directory

| File | Role |
|---|---|
| `main.tex` | The anonymized single manuscript document. **Reviewers see this.** |
| `titlepage.tex` | Full Title Page. Not anonymized. Separate upload. Never merge into `main.tex`. |
| `cover_letter.tex` | Cover letter to the Editor. Not anonymized. Separate upload. |
| `figures.tex` | Combined figure document. Separate upload. |
| `figures/fig1_collapse.pdf` | Figure 1. Copied unchanged from the full manuscript. |
| `figures/fig2_trivial_fraction.pdf` | Figure 2. Copied unchanged. |
| `refs.bib` | 25 entries, all cited, all present in the parent bibliography. |
| `checklist.md` | This file. Not submitted. |
| `README.md` | What was cut from the full manuscript and why. Not submitted. |

`paper/tex/main.tex`, `supplement.tex`, `refs.bib` and `figures/` are untouched.

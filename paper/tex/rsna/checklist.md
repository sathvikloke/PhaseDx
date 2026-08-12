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
| 1 | ~~Recount the body and get it under 3,000 words.~~ **DONE** — **2,990** words, measured on the current `main.pdf`. See §3. | Grew from 2,904 during the major-revision pass; 10 tokens of margin remain, so recount after any edit. |
| 2 | ~~Complete the CLAIM checklist.~~ **DONE** — `CLAIM_checklist.md`, all 44 items of the CLAIM 2024 update, none left blank. | *"Your paper will be sent back if this checklist is not included upon first submission."* **[AI]** Three items in it still need a human: see §9 and the file's own closing section. |
| 3 | **Fill every `TO SUPPLY` marker**: **9** in `titlepage.tex`, **3** in `cover_letter.tex`. | Middle initials, telephone, ORCID iDs, the final language-model access date, who wrote the independent reimplementation, the submission date and the non-anonymized acknowledgments are not recorded anywhere in this repository and were deliberately not invented. Everything the repository *does* establish has now been filled in — see §7 and §9. |
| 4 | ~~Decide the abstract word count convention.~~ **DONE** — 248 words including the four section headings, against the 250 cap, so it clears under either convention. See §3. | Both readings are now under the limit; the earlier 253-word figure counted the word "Abstract" itself. |

---

## 1. Files to upload

**[AI]** splits the submission into separate uploads plus one combined manuscript file.

### Uploaded separately

| Required file | Built here | Status |
|---|---|---|
| Cover Letter | `cover_letter.tex` → 4 pp | **ACTION** — 3 `TO SUPPLY` markers |
| Full Title Page | `titlepage.tex` → 3 pp | **ACTION** — 9 `TO SUPPLY` markers |
| Checklist (CLAIM) | `CLAIM_checklist.md` | **OK** — all 44 CLAIM 2024 items answered, every inapplicable one carrying its reason; 3 items flag a human decision |
| Figures, combined into **one** document, each legend immediately after its figure | `figures.tex` → 3 pp | **OK** |
| Supplemental Materials | — | none submitted; optional |

### Uploaded as one single document — `main.tex` → 22 pp

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
| Word count, Introduction→Discussion | ≤ 3,000 | **2,990** (see §3) | OK, 10 tokens of margin |
| Abstract | ≤ 250 words, structured, exactly 4 sections | **248** counting the heading word "Abstract" and the four section headings, 241 counting neither (see §3); sections correct | OK |
| References | ≤ 35 | **25** | OK |
| Figures (images, charts, graphs) | ≤ 6 | **3** | OK |
| Tables | ≤ 4 | **4** — at the cap | OK |
| Key Points | ≤ 3 | **3** | OK |
| Summary Statement | ≤ 255 characters | **243**, one sentence | OK |
| Abbreviations | ≤ 10 | 5 declared, 14 acronym forms appear (see §6) | **RISK** |
| Authors | ≤ 2 first authors; exactly one corresponding/senior author | 4 authors, 1 first, 1 corresponding | OK |

---

## 3. The counts

**Body — RESOLVED, re-measured on the current file 2026-08-12.** Counting whitespace-separated tokens in the text
extracted from the compiled PDF, Introduction through end of Discussion is now
**2,990 words** against the 3,000 limit, after the major-revision pass that
added the RSNA ICH slice-ordering provenance to Materials and Methods, the
composition of the nine-arm primary set, the DeepLesion arm-selection
disclosure, the PI-CAI relabelling, the scoping of the clinical claim, and the
bin-robustness limitation. Section totals: Introduction 334 · Materials and
Methods 917 · Results 1,057 · Discussion 682, against indicative budgets of
400/800/1000/800. Methods is over its indicative budget because the slice-
ordering provenance now lives there; roughly 450 tokens of prose were cut
across all four sections to pay for the additions, with no result, interval,
null or caveat removed --- supporting values that moved out of the body were
moved into the Table 1, Table 3 and Table 4 notes and the Figure 1 legend,
which sit outside the Introduction-to-Discussion span the limit applies to.

**The last two edits were word-count neutral, deliberately.** Two changes were
made after the 2,990 count was taken, and both were built to leave it unchanged:
the Methods pointer `split-to-split spread, labeled as such (Table 2)` became
`split-to-split spread and is labeled as such` (13 tokens either way — see §6 on
why the pointer had to go), and `plotted in Figure 2` became `plotted in Fig 2`.
The body was re-measured after both and is still **2,990**.

**Figure numbering.** The two-unit scatter is **Figure 2**, not Figure 3,
because figures must be numbered in order of first citation and Results reaches
the two-unit reading of the other benchmark-arms before it reaches the trivial
fraction. The trivial-fraction figure is therefore Figure 3 here, while the full
manuscript calls the same image Figure 2; the file was renamed to match, not
rebuilt.

**One measurement caveat, stated rather than hidden.** An early pass recorded
3,041 tokens for the same span under a convention that could not be reproduced
afterwards. The boundary used for every count since runs from the Introduction
heading to the last sentence of the Discussion, immediately before
Acknowledgments, and both edges are printed by the counting script and checked.
Margin is 10 tokens, so **recount on the file you actually upload, in the format
you upload it in**, before submitting — in particular after any conversion to
`.docx`, which the journal will require on acceptance.

**Abstract.** **248 words** counting the heading word "Abstract" and the four
section headings; 247 counting the headings but not the word "Abstract"; 241
counting neither. The limit is 250 **[AI]**. One caveat worth knowing before you
paste it anywhere: the abstract contains two thousands-separated figures
(752,802 and 18,938), and an extractor that splits on the comma reads those as
two tokens each, taking the strictest count to exactly 250. It is inside the
limit under every convention tried, but with no margin under the strictest, so
**do not add words to the abstract.**

**Summary Statement.** **243 characters** against the 255 cap, one sentence, no
abbreviations.

**Section budget.** A 2019 capture of the journal's preparation checklist
suggested 400 / 800 / 1000 / 800 for Introduction / Methods / Results /
Discussion. That page is date-limited and is not restated in the current
instructions. Against it, Methods (917) and Results (1,057) are over and
Introduction (334) and Discussion (682) are under; Methods is over because the
third-party slice-ordering provenance now lives there, which is where a reviewer
asked for it.

- **ACTION.** ScholarOne asks you to paste the abstract into a box at Step 1.
  Paste it, read the word counter the system shows you, and cut if it objects.
  Cutting 3 or 4 words costs nothing and removes the question for good.

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
- [x] **Summary Statement.** **243 characters**, one sentence, boldface, no
      abbreviations. The Original Research block says "1–2 sentences" while the
      Abbreviated Title Page block says "a single sentence"; the stricter reading
      is used. **[AI]** also forbids abbreviations in the Summary Statement —
      there are none. It now carries the corrected reading of the patient-level
      result ("carried no patient-level information at fixed stack depth"), not
      the earlier "below chance", which was a reviewer blocker.
- [x] **Key Points.** 3, each carrying summary data, none repeating the Summary
      Statement verbatim.
  - ~~**RISK.** Key Point 1 uses **CI**, Key Point 2 uses **IQR**.~~ **FIXED.**
    **[AI]** asks you to "avoid using vague language and abbreviations in the Key
    Points. Obvious abbreviations like CT and MRI are fine." Both are now spelled
    out: Key Point 1 writes "95% confidence interval" twice, Key Point 2 writes
    "interquartile range".
  - **RISK, unchanged.** Key Point 2 and the Summary Statement both carry the
    "about half of the published margin" finding. Not a verbatim repeat — Key
    Point 2 gives the median, the interquartile range, and the fact that the nine
    arms are four benchmarks — but close enough that a reader may notice.
    **Do not "fix" this by re-pointing Key Point 2 at the tightness of the
    distribution.** An earlier draft of this file suggested exactly that; the
    tightness sentence ("8 of 9 rows lie between 0.30 and 0.70") was deleted from
    the manuscript as a reviewer blocker, because six of those nine rows are the
    six label columns of one cohort and the claim was a statement about one
    benchmark dressed as a statement about nine.

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
| M&M first paragraph addresses IRB approval and consent | OK — states approval not required, and states the two conditions it rests on (no data obtained through intervention or interaction with a living individual; no identifiable private information used). The bare regulatory citation "45 CFR 46.102(e)" was dropped when the paragraph was tightened, which also removed the one undeclared plain abbreviation in the paper |
| **All tables, references and figures cited in numeric order** **[AI]**, twice | OK — **fixed 2026-08-12**. First citation of each is now Table 1 → Table 2 → Table 3 → Table 4 and Fig 1A → Fig 1B → Fig 2 → Fig 3. Before the fix the first table citation in the body was **Table 2**, in the Statistical Analysis paragraph, ahead of Table 1's first citation in Results; the pointer was replaced by "and is labeled as such", which is word-count neutral and loses nothing, because Table 2's own note carries the full explanation of why that one range is a split-to-split spread rather than a bootstrap interval. References are numbered by `unsrtnat`, which numbers by order of first citation by construction |
| In-text figure citations styled consistently | OK — **fixed 2026-08-12**. One "Figure 2" mid-sentence in Results became "Fig 2"; the body now uses "Fig N" throughout and reserves "Figure N." for the legends, which the journal requires to begin that way |
| M&M last paragraph: statistical methods, software **name + version + manufacturer**, and the *P* value used for significance | OK — Python 3.11 (Python Software Foundation, Wilmington, Del), NumPy 1.26, pandas 2.1, scikit-learn 1.4; states that no significance test was pre-specified and no *P* values are reported |
| Results give **numerators and denominators for every percentage** | OK, and now vacuously so — re-checked 2026-08-12. Results contains **no percentage of this study's own** apart from the 95% interval level; the two coverage percentages moved into the Figure 1 legend during the major-revision cut, and they carry their numerators and denominators there: 46.5% (93 of 200) and 91.5% (183 of 200). The only other percentages in the body (29%–55%, 96%, 59.7%, 90.5%) are values quoted from cited papers, in the Introduction |
| Discussion ¶1 restates the problem and the primary results | OK |
| Discussion **second-to-last paragraph is Limitations** | OK — Limitations is second-to-last, summary is last |
| Discussion final paragraph is a summary | OK |
| **Do not cite tables or figures in the Discussion** | OK — verified, zero Table/Figure references in the Discussion |
| Avoid self-evaluation: "novel," "unique," "ground-breaking" | OK — zero occurrences |
| Avoid claiming priority | OK — the Introduction concedes prior art before making any claim |
| Reserve "significant" for statistical significance | OK — the word does not appear anywhere |
| Text clear to a radiologist outside the specialty | Judgement call — reread once with that reader in mind |

### Abbreviations — re-measured on the current PDF, 2026-08-12

Five are declared: **AUC, CI, DWI, ICH, IQR**. Scanning the compiled body
(Introduction → Discussion) for acronym-shaped tokens returns fourteen forms:
the five declared, plus **AI, COVID-19, LUNA16, MIT, MRI, NYU, PI-CAI, RSNA,
RSNA-STR**.

All nine undeclared forms are proper names — of datasets (LUNA16, PI-CAI,
RSNA-STR), organisations (NYU, RSNA), a licence (MIT), a disease (COVID-19), a
modality (MRI), or the field itself (AI) — and are very unlikely to be counted
against the limit of 10.

- ~~**ACTION.** CFR is undeclared.~~ **RESOLVED.** **CFR** and **PMC** appeared
  in earlier drafts and are both gone from the current body: CFR left with the
  tightened IRB sentence, PMC with the material that was cut. There is now no
  plain, non-proper-noun abbreviation in the body outside the declared five.
- **RISK, small and unchanged.** "AUROC" appears on the *y*-axis of Figure 1 and
  inside Figure 3, while the text, the legends and the tables all say "AUC".
  The two text-side occurrences ("case AUROC" in the Table 4 note, "0.003 AUROC"
  in the cover letter) were changed to AUC on 2026-08-12, so the remaining
  mismatch is **inside the two figure images**, which are vector PDFs generated
  by the plotting script and cannot be edited in place. Fixing it means
  regenerating Figures 1 and 3 with "AUC" axis labels. A reviewer raised this;
  it is cosmetic, but it is also the kind of thing a copy editor will catch.

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
- [x] Word Count for Text — 2,990, recorded on the Full Title Page; see §3
- [~] Unanonymized acknowledgments — **mostly filled 2026-08-12.** The
      language-model disclosure is now the honest one and is word-for-word
      consistent with the cover letter and with the anonymized Acknowledgments of
      the manuscript: assistance in drafting **and** in writing the analysis code;
      tool **Claude (Opus 5), Anthropic**; access from **2026-07-27**. Two
      **ACTION**s remain: the **final** date of access, which cannot be known
      until the file is uploaded, and the *other* acknowledgments block — anyone
      named there must send written permission, because being named signals
      agreement with the data and conclusions **[AI]**, and if there is nobody,
      the block must say "None." rather than be deleted
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
      ([POL] LLM guidelines) — **filled except one date.** Tool, version, maker and
      first date of access are in the letter and now match the Full Title Page and
      the manuscript's Acknowledgments word for word. **ACTION**: the final date
      of access, in both non-anonymized files
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
- [x] **The three now say the same thing, and it is the honest version.**
      `titlepage.tex` carried a *false* narrower disclosure until 2026-08-12 —
      "no analysis reported in the manuscript was produced by a large language
      model" — while `main.tex` and `cover_letter.tex` had already been corrected
      to say the assistance covered **both drafting and the writing of the
      analysis code**. All three now carry the broader claim, and all three carry
      the same load-bearing argument in the same words: no value is
      model-generated; every number is a deterministic output of the released code
      run on a published label file; and the flagship estimate was reproduced by
      an independent reimplementation, the two agreeing to 0.003 AUC at both units
      on the full cohort.
- [x] Tool, version and maker are recorded on the two non-anonymized documents:
      **Claude (Opus 5), Anthropic**, accessed from **2026-07-27**. The
      manuscript's own Acknowledgments state, as the anonymized document must,
      that the specifics are in the cover letter and will be named in that section
      on acceptance.
- **ACTION, and only this.** The **final** date of access, in `titlepage.tex` and
  `cover_letter.tex`. It must be the same date in both.
- **RISK, worth one minute of a human's attention.** The **start** date of
  2026-07-27 is the date recorded in the cover letter, and the repository history
  is consistent with it for the code that produces every reported number: the
  audit pipeline first enters the history on 2026-07-29. But the repository's
  first commits are from April 2026 and belong to an earlier part of the larger
  study that this submission does not report. If language-model assistance was
  used on that earlier work too, the honest start date is earlier than
  2026-07-27, and the disclosure should say so.

### Reporting checklist

**[AI]**: required at first submission for Original Research involving human
subjects; the paper is returned without it.

- [x] **DONE — `CLAIM_checklist.md`.** Completed against the **CLAIM 2024 update**
      (Tejani et al, *Radiology: Artificial Intelligence* 2024;6(4):e240300,
      `10.1148/ryai.240300`), the 44-item list published at
      `pubs.rsna.org/page/ai/claim`, fetched rather than recalled. Item numbering
      and section structure follow that update, including its renaming of *Ground
      Truth* to **Reference Standard** and its renumbering relative to CLAIM 2020
      (for example, clinical trial registration is item 34 in 2024 and was item 40
      in 2020).
- [x] Every one of the 44 items is answered. **No item is blank.** The tally is
      **29 reported, 6 partly reported, 9 not applicable** — items 13, 18, 21, 24,
      27, 31, 33, 34 and 36, each carrying its reason on the same line: no images
      were downloaded, no annotation was performed, no sample size was targeted
      because every eligible row was used, no parameters are initialized, no
      ensembling was used, no post hoc explainability method was needed for a
      lookup table and a depth-3 tree, no model is offered for external use, no
      clinical trial exists, and age and sex are not distributed with these label
      files. The 6 partial rows each state what is missing rather than rounding up.
      Every applicable item is mapped to a section **and a page** of the compiled
      manuscript, with a page map at the top of the file explaining that pages are
      counted from the first page because the journal forbids page numbers.
- [x] The file is **anonymized**, on the assumption a reviewer may see it: it
      names no author, institution, repository, archive DOI or software package,
      and where an item is satisfied only on a non-anonymized upload it says so
      and stops there.
- **ACTION, three items inside it.** (1) Item 1 — the title does not name a
  technology category; decide whether to retitle. (2) Items 42 and 43 — the
  protocol reference, the repository address and the commit identifier are
  withheld for anonymization; agree with the editor whether they go in now or on
  acceptance. (3) Items 12, 26 and 35 — three reporting gaps the checklist
  declines to overstate: the fastMRI Prostate dropped-row counts, how the 199
  available fastMRI+ knee volumes were determined, and the upward bias in a
  numerator that is a maximum over five correlated baselines selected on the same
  test data. Each is a sentence of manuscript text, not new analysis, and the body
  has 10 tokens of margin.
- The cover letter already tells the editor a CLAIM checklist is enclosed, says
  that many items are marked not applicable with reasons, and offers STROBE
  instead if the editors prefer it for a study of this shape.

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

**Re-run 2026-08-12 on the current build of `main.pdf` and `figures.pdf`.** Text
was extracted from both and searched. Zero occurrences of: any author surname as
an author, the institution names, the city names, the repository name, the
software name, "GitHub", "Zenodo", `10.5281`, any DOI of ours, and any file path.
Zero occurrences, too, of every term on the excluded arm's stop list.

- One benign hit: **"Johnson"** appears twice in `main.pdf`, both times as a
  third-party reference author (Patricia Johnson, in the fastMRI citations). A
  reviewer cannot infer authorship from a common surname in a reference list. No
  action.
- Document metadata: `main.pdf` carries `LaTeX with hyperref` / `xdvipdfmx`,
  `figures.pdf` carries `tectonic` / `xdvipdfmx`. The figure PDFs themselves carry
  only `Matplotlib` in their Creator/Producer fields. No author, institution,
  filename or path anywhere. No action.
- **ACTION.** Re-run this check on the final PDF you actually upload, and again
  after any conversion to `.docx`. Anonymization is a property of the file you
  upload, not of the file you checked last week. `CLAIM_checklist.md` was written
  anonymized for the same reason and should be re-checked with the rest.

---

## 12. Files in this directory

| File | Role |
|---|---|
| `main.tex` | The anonymized single manuscript document. **Reviewers see this.** |
| `titlepage.tex` | Full Title Page. Not anonymized. Separate upload. Never merge into `main.tex`. |
| `cover_letter.tex` | Cover letter to the Editor. Not anonymized. Separate upload. |
| `figures.tex` | Combined figure document. Separate upload. |
| `figures/fig1_collapse.pdf` | Figure 1. Copied unchanged from the full manuscript. |
| `figures/fig2_unit_scatter.pdf` | Figure 2. Built for this submission by `make_rsna_figures.py`. |
| `figures/fig3_trivial_fraction.pdf` | Figure 3. Copied unchanged (the full manuscript's Figure 2), renamed to match its number here. |
| `CLAIM_checklist.md` | The CLAIM 2024 reporting checklist, all 44 items. **Submitted, as a separate upload.** Anonymized. |
| `refs.bib` | 25 entries, all cited, all present in the parent bibliography. |
| `make_rsna_figures.py` | Builds Figure 2 and prints a source ledger. Not submitted. |
| `checklist.md` | This file. Not submitted. |
| `README.md` | What was cut from the full manuscript and why. Not submitted. |

`paper/tex/main.tex`, `supplement.tex` and `refs.bib` are untouched.

**One caveat on `paper/tex/figures/`.** On 2026-08-12 the independent
verification pass invoked `paper/tex/make_figures.py --help`; that script
regenerates on import rather than parsing arguments first, so all seven of the
full manuscript's figure PDFs were rewritten. They were checked afterwards and
are **content-identical** — same byte length, same page count, same extracted
text — differing only in the embedded PDF `CreationDate`. Nothing in this
submission uses them. Restore with `git checkout -- paper/tex/figures/` if a
clean `git status` matters.

---

## 13. Independent verification pass — 2026-08-12

A separate check of the whole package: recompile, recount, re-grep, and
re-verify every number against its artifact rather than against the record of
the previous pass.

**Clean.** All four documents compile under `tectonic` with **zero** errors,
warnings, overfull boxes, underfull boxes, undefined citations and undefined
references (22 / 3 / 3 / 4 pp). Counts re-measured on the compiled PDF and
unchanged: body **2,990** (2,998 splitting thousands separators), sections
334 / 917 / 1,057 / 682, abstract **248** (250 strict), Summary Statement
**243** characters, 3 Key Points, 5 declared abbreviations, 25 references all
cited, 3 figures, 4 tables. First-citation order is Table 1, Fig 1, Table 2,
Fig 2, Table 3, Table 4, Fig 3; references appear in strict 1–25 first-citation
order; no table or figure is first cited in the Discussion. Figure legends are
byte-identical between `main.tex` and `figures.tex`. Anonymization and
prevalence-screen greps on both submitted PDFs return **zero** hits.

**One defect found and fixed.** The Table 1 note gave the fold-to-fold range of
the patient-level permutation null as `0.523--0.560`, two sentences before
quoting the audit tool's own draw as **0.561**. The artifact value is
`0.5605683`, which is **0.561** to three decimals, so the range excluded a
value the same note reported. Changed to `0.523--0.561`. Word counts are
unaffected.

**Verified from scratch, not merely re-read.** The Table 1 lower block's two
load-bearing pair counts reproduce to the digit from
`rsna_ich_slices.csv` under the note's own definition of stack depth as a
patient's slices per volume: **10,065,308** within exact depth over 36 strata,
and **25,685,534** within 5-slice strata. Stack depth alone reproduces as
**0.402 (0.394, 0.410)**, matching all three printed digits including both
interval bounds, and maximum aggregation does collapse to exactly five distinct
patient scores, one per fold. The constant-predictor claim — that the five fold
training prevalences reproduce the pooled 0.492 in closed form with no other
input — reproduces to six decimals (**0.491816**), and the stated mechanism
holds exactly: correlation between a fold's training prevalence and its own
test prevalence is **−0.99998**. Table 2's independent-implementation subsample
row was recomputed end to end at 2,000 replicates and returns
**0.731 (0.723, 0.739)** and **0.458 (0.428, 0.486)**, the printed values.

**Three items a human should settle before upload.**

| # | Item | Status |
|---|---|---|
| 1 | **Software versions.** Materials and Methods claims Python 3.11, NumPy 1.26, pandas 2.1, scikit-learn 1.4. No environment record exists anywhere in the repository, `requirements.txt` pins only floors, and the current virtual environment is Python 3.14 / NumPy 2.4 / pandas 3.0 / scikit-learn 1.8. The claim may well be right for the July 2026 runs, but nothing here establishes it and the only live evidence disagrees. **ACTION** — confirm against the environment that produced the artifacts, or restate. Do not guess. |
| 2 | **Deciles pair count.** `\num{11936868}` in the Table 1 lower block does not reproduce. Under the same depth definition that reproduces the other four counts exactly, the within-decile pair count is 11,967,327; five other plausible binning conventions were tried and none returns the printed value. The row's AUC is unaffected in substance — every convention leaves it covering 0.500 — but the count itself is unconfirmed. **ACTION** — recompute or drop the count from that row. |
| 3 | **Fold seed for the Table 1 lower block.** The block is stated to run "under an independently drawn fold assignment whose unstratified patient-level AUC is 0.449". That draw's seed is recorded nowhere, so the third decimals of the four stratified readings cannot be reproduced exactly; independent draws matching the stated 0.449 give 0.487–0.498 within exact depth against the printed 0.497. The conclusion is robust — every draw tried leaves the stratified reading covering chance — but Statistical Analysis promises that a seed is "recorded per run". **ACTION** — record the seed. |

**Not defects, checked and cleared.** Series with hemorrhage do average
**33.9** slices against 35.1 without, at the series unit the sentence names (the
patient-unit reading of the same quantity is 33.6 against 35.1, which is why
this looks like a discrepancy and is not). The `0.497 (0.487, 0.508)` printed
for the depth-fixed reading is identical to the intraventricular row directly
above it in the same table; this is a genuine coincidence, not a copy-paste —
both were confirmed separately against their own sources. DeepLesion's **665**
patients is the official-split test count, not the 1,368 subjects in the file.
The three `Johnson` matches in an anonymization grep of `main.pdf` are Patricia
M. Johnson in the two cited fastMRI references, not a co-author leak.

# Authorship, contributions, competing interests, funding and availability

Scaffold for the submission's front and back matter. **Everything in this file is a template
until a named person fills it and confirms it.** Nothing here should be pasted into a
manuscript with a placeholder still in it.

---

## 0. The standard this must meet

**npj Digital Medicine enforces the ICMJE authorship criteria.** Springer Nature journals
require that every listed author meet **all four**, not a majority of them:

1. **Substantial contribution** to the conception or design of the work, **or** to the
   acquisition, analysis or interpretation of data;
2. **Drafting the work or revising it critically** for important intellectual content;
3. **Final approval of the version to be published** — the actual final version, not a draft;
4. **Accountability for all aspects of the work**, including agreement to investigate and
   resolve questions about the accuracy or integrity of any part of it.

Three consequences that decide real cases and are worth writing down before the author list is
settled:

- **Acquiring funding, providing data, or general supervision does not by itself justify
  authorship.** Those belong in Acknowledgements.
- **Criterion 4 is not a formality.** An author is accountable for the *whole* paper, not only
  their own section. An author who cannot answer for a number in a table they did not produce
  should not be an author of a paper containing it.
- **Every author must approve the final version.** Approval of a near-final draft is not
  approval of the version submitted.

The journal will also expect a **CRediT** taxonomy statement. CRediT records *what each person
did*; ICMJE decides *who may be an author*. They are not interchangeable, and satisfying CRediT
does not satisfy ICMJE. Both are drafted below.

**Current state of the author list.** `paper/DRAFT.md` still reads
*"Authors: [to be completed]"*. `paper/COLLABORATORS.md` identifies two roles that are required
and unfilled: a **senior radiologist co-author** (to write and stand behind the clinical
paragraph on whether the positional signal is anatomy) and a **biostatistics review** —
explicitly recommended as an *independent reviewer rather than a co-author*, so that the
acknowledgement can say the statistics were checked by someone with no stake in the result. If
the statistical review turns into substantive work, authorship follows; that decision must be
made on ICMJE criterion 1, not on gratitude.

`trivialbaselines/CITATION.cff` and `trivialbaselines/.zenodo.json` list four creators for the
**software**: Sathvik Loke, Ethan Johnson, Neeraj Movva, Aditya Raut. Software authorship and
manuscript authorship are separate determinations. Whoever appears on the manuscript must meet
all four ICMJE criteria for the *manuscript*.

---

## 1. Author list

Fill in submission order. Order is a decision to be made and recorded, not inferred.

| # | name | degrees | affiliation(s) | ORCID | email | corresponding? |
|---|---|---|---|---|---|---|
| 1 | Sathvik Loke | | Illinois Mathematics and Science Academy, Aurora, IL, USA | | sloke@imsa.edu | ☐ |
| 2 | Ethan Johnson | | | | ethanthekek21@gmail.com | ☐ |
| 3 | Neeraj Movva | | Illinois Mathematics and Science Academy, Aurora, IL, USA | | nmovva@imsa.edu | ☐ |
| 4 | Aditya Raut | | | | aditya.sraut09@gmail.com | ☐ |
| 5 | *[senior radiologist — required, unfilled]* | | | | | ☐ |
| 6 | *[biostatistician — only if criterion 1 is met; otherwise Acknowledgements]* | | | | | ☐ |

ORCID iDs are effectively mandatory for the corresponding author at Springer Nature and are
requested for all authors. Affiliations must be the affiliation *at the time the work was done*.

---

## 2. ICMJE criteria table — to be completed by each author individually

Each author ticks each box **for themselves**. A row with any unticked box is not an author.
Do not fill this in on anyone else's behalf; collect it in writing.

| author | 1. substantial contribution to conception/design **or** acquisition/analysis/interpretation | 2. drafted or critically revised for important intellectual content | 3. approved the final version to be published | 4. accountable for **all** aspects, incl. accuracy and integrity |
|---|---|---|---|---|
| Loke | ☐ | ☐ | ☐ | ☐ |
| Johnson | ☐ | ☐ | ☐ | ☐ |
| Movva | ☐ | ☐ | ☐ | ☐ |
| Raut | ☐ | ☐ | ☐ | ☐ |
| *[radiologist]* | ☐ | ☐ | ☐ | ☐ |
| *[biostatistician, if applicable]* | ☐ | ☐ | ☐ | ☐ |

---

## 3. CRediT contributor-roles table — to be completed

Mark **L** (lead), **E** (equal) or **S** (supporting) in each cell; leave blank where the role
does not apply. Every role that was performed must be attributed to someone; a role nobody
performed is left as an empty row and that is a finding about the work, not an embarrassment.

The role list is CRediT's fourteen, with the concrete artefact each one maps to in this project
so the table is filled against evidence rather than from memory.

| CRediT role | what it means **here** | Loke | Johnson | Movva | Raut | *[rad.]* | *[stat.]* |
|---|---|---|---|---|---|---|---|
| Conceptualization | the zero-image-baseline framing; the decision to pair an audit with a prevalence screen | | | | | | |
| Methodology | the null-model family; the verdict rule and trivial fraction; the screen's protocol, codebook and endpoints | | | | | | |
| Software | `trivialbaselines`; the `pipeline/` stages; the screen's analysis scripts | | | | | | |
| Validation | the 53 self-tests; the independent recomputation of the unit-collapse result; the `--verify` digest check | | | | | | |
| Formal analysis | the audit results; the pooled prevalence endpoints; agreement statistics; the bounding analyses | | | | | | |
| Investigation | benchmark label-file acquisition; **screening and extraction of the 100 sampled records** | | | | | | |
| Resources | data access, compute | | | | | | |
| Data curation | label-file provenance table; the frozen frame and permutation; the sealed screener files | | | | | | |
| Writing — original draft | `paper/DRAFT.md` | | | | | | |
| Writing — review & editing | critical revision; the clinical paragraph; the statistical review | | | | | | |
| Visualization | the PRISMA flow figure; the audit figures | | | | | | |
| Supervision | | | | | | | |
| Project administration | | | | | | | |
| Funding acquisition | | | | | | | |

### 3.1 Contributions statement, prose form

Replace every bracket. This is the paragraph the journal prints.

> **Author contributions.** [Initials] conceived the study and designed the zero-image baseline
> family and the verdict rule. [Initials] implemented `trivialbaselines` and the audit pipeline.
> [Initials] designed the prevalence screen protocol, the extraction codebook and the analysis
> plan. [Initials, Initials, Initials, Initials] screened and extracted records independently as
> screeners S1–S4. [Initials] performed the pooling, agreement and bounding analyses.
> [Initials] provided the clinical interpretation of the positional signal and of the anatomical
> alternative explanation. [Initials] reviewed the statistical methods. [Initials] drafted the
> manuscript. **All authors critically revised the manuscript for important intellectual
> content, approved the final version, and agree to be accountable for all aspects of the
> work.**

### 3.2 A disclosure that must be made, and made explicitly

The four screeners in this study are designated S1–S4 and their sealed submissions are released
in full. **The manuscript must state, in the Methods, who the four screeners were** — whether
they are among the listed authors, and whether any screener was also an author of the protocol
they screened against. Protocol §4.2 records that the ten pilot records were read by the
protocol author and permanently excluded from analysis for exactly this reason, and `D` §5.1
already states that screeners were not blinded to the study hypothesis. Naming the screeners
completes that disclosure. If any part of the screening or extraction was performed with
automated or AI assistance, that must be described in the Methods with the same specificity —
what tool, at which step, and what a human verified — and it does **not** confer authorship on
any tool. Springer Nature requires this disclosure and forbids listing an AI system as an
author.

---

## 4. Competing interests

Springer Nature requires a declaration from **every** author, financial and non-financial,
covering the 36 months before submission. "None" is a valid answer and must still be stated.

Non-financial interests are the ones people forget and are the ones that matter here: this paper
audits named public benchmarks and evaluates a named preprint. Anyone who is an author of an
audited benchmark, a co-author of a compared paper, an employee of an organisation that
publishes one, or a competitor of one, has a non-financial competing interest and must declare
it.

### 4.1 Per-author collection form

| author | financial (grants, consultancies, honoraria, patents, stock, expert testimony, paid speaking) | non-financial (advisory roles, unpaid memberships, personal or professional relationships, involvement with an audited benchmark or a compared publication) |
|---|---|---|
| Loke | | |
| Johnson | | |
| Movva | | |
| Raut | | |
| *[rad.]* | | |
| *[stat.]* | | |

### 4.2 Statement, if there are none

> **Competing interests.** The authors declare no competing interests.

### 4.3 Statement, if there are any

> **Competing interests.** [Author] reports [interest] from [entity], [relationship to the
> submitted work]. [Author] is a co-author of [publication/benchmark] that is evaluated in this
> manuscript; [they took no part in the coding of that record / the record was coded by an
> independent screener and is flagged in the released extraction file]. The remaining authors
> declare no competing interests.

### 4.4 Specific checks to run before signing this

- **`trivialbaselines` is released under MIT with no commercial interest attached.** If that
  changes — a company, a licence change, a paid support arrangement — it is a financial interest
  and must be declared.
- **The manuscript evaluates a 2024 fastMRI Prostate preprint and seven public benchmarks.**
  Confirm that no author is an author of any of them, and if one is, say so in the statement and
  in the Methods, and confirm that record or benchmark was not scored by that author.
- **`paper/COLLABORATORS.md` proposes recruiting a radiologist as a co-author and a
  statistician as an independent reviewer.** If the statistician later becomes an author, their
  review can no longer be described as independent, and the Acknowledgements must be rewritten
  rather than left standing.

---

## 5. Funding

> **Funding.** [Option A — none:] This research received no specific grant from any funding
> agency in the public, commercial or not-for-profit sectors.
>
> [Option B — funded:] This work was supported by [funder] under grant [number] to [author].
> The funder had no role in the design of the study; in the collection, analysis or
> interpretation of data; in the writing of the manuscript; or in the decision to submit it for
> publication.

Choose one and delete the other. If any author is supported by a fellowship, a stipend, or an
institutional programme that touched this work — including a secondary-school research
programme — it is disclosed here even if no money was granted for this specific project. The
statement of the funder's role is required whenever there is a funder, and "no role" must be
true.

---

## 6. Data and code availability

The manuscript already carries a long availability section (`paper/DRAFT.md` → *Data and code
availability*). What follows is the compact statement the journal prints, plus the four gaps
that must be closed before it is true as written.

### 6.1 Data availability

> **Data availability.** No pixel data were downloaded, generated or redistributed by this
> study. The audit reads only publicly released label tables; Table 1 gives the source URL,
> byte count, SHA-256 prefix and licence of every label file used, and each remains available
> from its original distributor under its original terms. All derived data supporting the
> findings — per-benchmark audit payloads, the frozen 9,979-PMID sampling frame with its
> SHA-256 digest, the seeded permutation, the drawn sample, the four sealed screener
> submissions with every verbatim quote and search string, the access-recovery records, and
> every analysis output — are in the public repository at
> https://github.com/sathvikloke/PhaseDx and archived at [Zenodo DOI]. Full texts retrieved
> during screening are **not** redistributed, because most are under publisher copyright or a
> NonCommercial licence; the retrieval URLs and the verbatim quotes in the extraction files are
> the audit trail. The 16 records whose full text could not be obtained are listed by PMID so
> that a reader with better access can complete the screen.

### 6.2 Code availability

> **Code availability.** The auditor is released as `trivialbaselines` (MIT licence; `numpy`
> and `pandas` only) at https://github.com/sathvikloke/PhaseDx/tree/main/trivialbaselines and
> archived at [Zenodo DOI]. It ships a self-test that includes a benchmark the baselines must
> **fail** to detect alongside ones they must detect, so a tool that always fires cannot pass.
> The complete analysis pipeline, the prevalence screen's protocol, codebook and analysis
> scripts, and the script that regenerates the PRISMA flow, are in the same repository. Every
> number in the manuscript is regenerated from the released inputs by a named command.

### 6.3 Gaps to close before either statement is true

1. **The Zenodo DOI does not exist.** `trivialbaselines/.zenodo.json` and `CITATION.cff` are
   prepared, but nothing has been deposited and no DOI has been minted. Every `[Zenodo DOI]`
   above, and every DOI reference in the manuscript, is a placeholder. Mint it, then use the
   **concept DOI** (which always resolves to the newest version) in prose and the
   **version DOI** in the reference list.
2. **The installation instructions disagree between files.** `paper/protocol.md` and
   `paper/checklist.md` advertise `pip install trivialbaselines` while
   `trivialbaselines/README.md` documents `git clone && pip install .`. Only one can be true.
   Fix before submission; the draft already flags this.
3. **The screen's sealed files are not under version control.** `paper/screen_batch_*.json`,
   `paper/screen/access_recovery.json` and everything in `paper/screen/analysis/` are untracked
   working-tree files. A statement that they are "in the public repository" is not yet true.
   Commit them, and commit the v1.2 protocol and codebook, before the availability statement is
   filed. See `paper/registration.md` §2.
4. **`manuscript/` must be excluded from any archive.** It contains an earlier draft whose
   numbers no code in the repository has ever produced; `manuscript/DO_NOT_SUBMIT.md` says so.
   Exclude the directory from the Zenodo deposit explicitly rather than trusting a default.
5. **The unreachable count in the draft is stale.** `paper/DRAFT.md` → *Data and code
   availability* says "the 20 unreachable records are listed by PMID"; after the v1.3 access
   recovery the number is **16**, and the list is in `paper/prisma_flow.md`. The statement
   above uses 16. Reconcile the draft, and note that `paper/screen/analysis/prisma_flow.py`
   must be added to the availability section's list of released scripts.

---

## 7. Acknowledgements

> **Acknowledgements.** We thank [name] for statistical review of the interval methods and the
> verdict rule; the review was independent and [name] is not an author. We thank [name] for
> [clinical reading / comments on an earlier draft]. [Any other non-authorship contribution.]

Anyone who contributed but does not meet all four ICMJE criteria goes here — and **must give
written permission to be named**, because being acknowledged can be read as endorsing the
conclusions. Collect that permission before submission.

---

## 8. Pre-submission checklist for this file

- [ ] Author list finalised, order agreed and recorded, ORCID for every author
- [ ] Radiologist co-author recruited, or the clinical claim removed from the paper
- [ ] Biostatistician review obtained; their status as reviewer or author decided on ICMJE
      criterion 1 and reflected consistently in §3 and §7
- [ ] Every author has personally confirmed all four ICMJE criteria (§2)
- [ ] CRediT table complete; every role that was performed is attributed (§3)
- [ ] The four screeners are named in the Methods, with any AI or automated assistance
      described at the step where it was used (§3.2)
- [ ] Competing-interests declaration collected from every author individually (§4)
- [ ] The benchmark- and preprint-conflict checks in §4.4 have actually been run
- [ ] Funding statement chosen, with the funder's role stated if there is a funder (§5)
- [ ] Zenodo deposit made, DOI minted, every placeholder replaced (§6.3)
- [ ] Sealed screen files committed so the availability statement is true (§6.3)
- [ ] Everyone named in the Acknowledgements has given written permission (§7)
- [ ] Registration wording taken verbatim from `paper/registration.md` §5, and none of the
      forbidden phrases in `paper/registration.md` §5.4 appears anywhere in the manuscript

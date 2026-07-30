# Access recovery over the four reserve blocks

Companion to `paper/screen/access_recovery_reserve.json`. Second access pass over the **35
records coded `unreachable_paywalled`** in the sealed reserve blocks
`paper/screen_reserve_R{1,2,3,4}.json`. Those blocks had had only one access pass; the four
main batches already had their second pass recorded in `paper/screen/access_recovery.json`.

**The sealed reserve files were not modified.** This is an analysis-time overlay, in the same
form as `access_recovery.json`.

---

## 1. Headline

| | as sealed | after this pass |
|---|---|---|
| unreachable | 35 | **34** |
| eligible-looking denominator | 111 | 111 |
| unreachable rate | 31.5% | **30.6%** |

**One record of thirty-five recovered.** The pre-registered §7 threshold is 15%; at 30.6% it
**still binds**, so the bounding interval remains the headline rather than the point estimate.

### Per block

| block | unreachable as sealed | recovered | still unreachable | denominator | rate before | rate after |
|---|---|---|---|---|---|---|
| R1 | 9 | **1** | 8 | 26 | 34.6% | **30.8%** |
| R2 | 5 | 0 | 5 | 22 | 22.7% | 22.7% |
| R3 | 9 | 0 | 9 | 28 | 32.1% | 32.1% |
| R4 | 12 | 0 | 12 | 35 | 34.3% | 34.3% |
| **total** | **35** | **1** | **34** | **111** | **31.5%** | **30.6%** |

---

## 2. No zero-image baseline was found

**Zero positives in this pass.** The count across the project remains exactly zero.

The one fully recovered paper (38324428) reports no zero-image baseline of any kind: all six
sub-flags FALSE, evidenced by the complete 14-term search. The 13-page supplement of 40644803
was also fully 14-term searched — a positive in a supplement would still count — and likewise
contains none.

Every term hit was read in context and adjudicated individually rather than dismissed in bulk.
In 38324428 all 14 hits are metaphorical (*"may serve as a baseline for general target
localization problems in DBS"*), algebraic (*"Ignoring the trivial case W_B = 0"*), or
descriptive of a result distribution (*"The majority (> 80%) is below 4 mm"*). The two
`constant` hits in the 40644803 supplement are the intercept row of a logistic-regression
table, not a constant-class predictor.

---

## 3. Which endpoints worked

The brief expected the **PMC OA Web Service** to be the single most likely win. It cannot help
here at all: **not one of the 35 records is in PMC.** NCBI's ID converter returns *"Identifier
not found in PMC"* for every record and Europe PMC returns `pmcid=null, inEPMC=N` for every
record. Europe PMC `fullTextXML` returned HTTP 404 for all 35 on both routes.

What actually worked was a two-step chain that no single service could complete:

1. **Crossref `query.bibliographic` title search — the discovery endpoint.** It was the only
   source that revealed the existence of the TechRxiv preprint DOI
   `10.36227/techrxiv.19750774.v1` (CC BY-NC-SA 4.0) for record 38324428. **Unpaywall,
   OpenAlex and Europe PMC each independently reported that record as closed with zero OA
   locations.**
2. **Internet Archive Wayback Machine, raw (`id_`) mode against an archived 302 — the delivery
   endpoint.** The live TechRxiv site returns a Cloudflare interstitial and HTTP 403, and the
   figshare object it used to redirect to has since been **deleted** (404 at both
   `api.figshare.com` and `ndownloader.figshare.com`), so Semantic Scholar's OA link is dead at
   source. The Wayback copy is now the only reachable copy of a document the authors themselves
   posted under an open licence.

**figshare API** gave a partial result: the ACS Supporting Information for 40644803 under
CC BY-NC 4.0 — the supplement only, not the article body.

Returning nothing: Unpaywall (33/35 `closed`), OpenAlex (agreed on all 35), Europe PMC
supplementaryFiles and PPR, arXiv (no title match), OpenAIRE (zero fulltext URLs), CORE (one
metadata record whose download 404s).

**BASE was unusable from this environment** — it returns *"Access denied for IP address … and
user agent …"* and requires a registered IP. Recorded as an environment limitation, not as
evidence of absence.

**HKUST's institutional repository was not crawled**: its `robots.txt` disallows `/ir/search/`
for non-whitelisted agents. Record 36403310 stays unreachable on that basis.

### Two false starts worth recording

Both were caught and corrected before any conclusion was drawn from them:

- **CORE** was first queried with `title:"…"`, which CORE does not parse as a filter; it
  returned the entire **88,586,129-record corpus** as an apparent hit. Re-run querying by DOI.
- **arXiv** silently returned **HTTP 429 for 15 of the 35** queries, which look identical to
  "no match". Re-run with rate limiting; all 35 then returned HTTP 200.

---

## 4. Two traps that would inflate a recovery count

**33741850 — metadata says OA; the file is not the paper.** Unpaywall and OpenAlex both report
`is_oa=true, oa_status=green, version=submittedVersion, cc-by-nc-nd` at the Fujita Health
University repository. A pass that trusted the metadata would score this **recovered**. It is
not. The deposited file is **one page**, and is the author's **doctoral dissertation abstract in
Japanese** (`論文内容の要旨`) plus the thesis examination committee's report — not the
*Nuclear Medicine Communications* article. It cannot support the mandatory 14-term search, so
the record stays unreachable and `trivial_baseline` stays `not_assessable`.

**40644803 — the figshare match is the supplement.** DOI `10.1021/acsnano.5c02822.s001`. The
supplement was obtained and searched, but the body was not, so the negative cannot be certified
and the record stays unreachable.

---

## 5. Demonstrably open access, refused by this environment

Two further records are **open access by publisher-declared licence** and unreachable only
because this execution environment is refused. As in the previous pass, they are counted
unreachable because no full text was read, and the cause is disclosed rather than charged to
the literature.

- **35181263** — Semantic Scholar reports a GREEN open-access accepted manuscript at
  ScienceDirect (`/article/am/pii/S221253452200003X`); Elsevier returns HTTP 403 to every
  client here, and there is no Wayback snapshot.
- **38723886** — Unpaywall and OpenAlex both report `hybrid`, `publishedVersion`,
  `cc-by-nc-nd`, publisher-hosted; ScienceDirect serves this environment a 2,664-byte
  interstitial, and there is no Wayback snapshot.

Recovering both elsewhere would give 32/111 = 28.8%, still far above 15%.

---

## 6. The recovered record

**38324428** — Weng L, Zhu Z, Wu H, Zhu J. *Reduced-Reference Learning for Target Localization
in Deep Brain Stimulation.* IEEE Trans Med Imaging 2024. doi:10.1109/TMI.2024.3363425

Read as the **TechRxiv v1 preprint** (CC BY-NC-SA 4.0), 12 pages, 50,090 characters — **not the
version of record**. Flagged for adjudication on that ground, and reserved for the
version-of-record sensitivity analysis exactly as 36200353 is in `access_recovery.json`.

**Eligibility resolved in favour of inclusion.** S1 had recorded this as genuinely undecidable
from the abstract: the abstract frames the task as regression (*"we consider target
localization as a non-linear regression problem"*) and reports only millimetre errors, pointing
to `E-SEG`, while also naming a classification network. The full text settles it. I2 is met — a
supervised classifier is fitted over slices under a binary cross-entropy loss. I3 is met — the
input is 2D slices of a 3T volumetric MRI acquisition. I4 is met — precision and recall are
defined and reported numerically, and precision is PPV while recall is sensitivity, both named
in I4. `E-SEG` does not apply: its own text is qualified *"with NO categorical class decision
evaluated"*, a class decision **is** evaluated here, and E-SEG directs that a paper doing both
be included with only the classification arm coded. Coded accordingly.

| field | value |
|---|---|
| evaluation unit | slice |
| headline unit | `na_only_one_unit_reported` |
| split unit | **patient_subject** — *"a training set (127 subjects) and a validation set (32 subjects, about 20%)"* |
| trivial_baseline | all six **FALSE**, evidenced |
| positional distribution reported | **no** |
| dataset | Human Connectome Project, WU-Minn HCP 1200 Subjects Release (public) |
| modality / region | MRI / brain |
| n patients (test) | 159 (32) |
| headline metric | precision@1 = 0.98 (classification arm) |
| test set | internal held out, averaged over five rounds |
| uncertainty | none |
| code availability | not stated |

**Scope limitation on the negative.** The preprint's text refers to an appendix (*"determined by
a DTI-based tractography process, as described in the appendix"*), but **no appendix is present
in this 12-page v1 document** — it ends with the reference list. The 14-term negative is
therefore evidenced **for the complete main text only**, and the IEEE version-of-record
supplement was not obtained. This is the second ground for `flag_for_adjudication`.

Notably for this project's own construction, the paper ranks slices by classifier score but
**never reports the distribution of target-bearing slice indices** — `slice index` and
`position` both return zero hits. Only single illustrative slice numbers appear in figure
captions (*"(a) subject 48, slice 108"*). This was checked specifically rather than incidentally.

---

## 7. Sources refused

Sci-Hub, LibGen and every other unauthorised source were **not fetched at any point, not even to
test availability**. No record is reported as reachable because such a source holds it. Cloudflare
and bot-detection challenges were **not circumvented**; where a site refused this environment that
is recorded as an environment limitation. `robots.txt` was respected.

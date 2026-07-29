# Pre-registered protocol: how often does the published literature check whether a 3D imaging benchmark is solvable without pixels?

**Status: FROZEN, 2026-07-29.** The sampling frame has been executed, the sample drawn, and
every endpoint fixed. No record in the analysis sample has been read by the protocol author.
Amendments go in the changelog (§12) and in `paper/screen_frame.json`; nothing is edited
silently.

**Companion files**

| file | what it is |
|---|---|
| `paper/screen_frame.json` | the extraction form and the codebook, one decision rule per ambiguous case |
| `paper/screen_sample.json` | the 100 sampled papers, four batches plus the overlap set, plus the reserve |
| `paper/screen/frame_pmids.txt` | the frozen frame, 9,979 PMIDs |
| `paper/screen/permutation.txt` | the seeded permutation of the frame |
| `paper/screen/reproduce_frame.py` | `--verify` re-derives both SHA-256 digests offline; `--refetch` re-runs the query live |
| `paper/screen/build_sample.py` | rebuilds `screen_sample.json` from the permutation; `--check` diffs against the committed file |

---

## 1. The question, and why the frame is the whole argument

We have shown that on one benchmark a zero-image positional baseline reaches 0.851 slice-level
AUROC against a published 0.861, and that 98% of the margin over chance needs no pixels
(`paper/protocol.md`). That is a *demonstration*. It says the failure mode exists. It says
nothing about how often it goes unchecked.

Turning "here is a failure mode" into "here is its prevalence" requires a defensible
denominator. A convenience sample of papers we happened to find while writing the audit is
worthless, and a reviewer will say so in one sentence, because the papers we found are exactly
the papers where we suspected the problem. Everything below exists to make the denominator
survive that sentence.

**Primary endpoint (P1).** Among included papers, the proportion reporting at least one
**zero-image baseline** — constant/prevalence, positional, acquisition-metadata, or
permuted-label — with a *measured value* on the same metric.

This is the family our checklist requires at items B1–B3. A clinical-variables-only nomogram is
deliberately **not** in the primary: it is a different comparison and it does not test whether
the benchmark is solvable without pixels. It is captured in secondary endpoint S1 so the two
can be read apart.

---

## 2. The frame

**Database.** PubMed, via the NCBI E-utilities `esearch.fcgi` endpoint. Chosen because it is
the only source of the three we tried that can be queried reproducibly from a script with no
API key and no rate-limit lottery. What we tried and what happened, on 2026-07-29, is recorded
honestly in §10 — the arXiv API returned an empty body from this environment and the Semantic
Scholar API returned HTTP 429.

**Exact query string.** Copy-pasteable; also frozen as a constant in `reproduce_frame.py`.

```
("deep learning"[tiab] OR "convolutional neural network"[tiab] OR "convolutional neural
networks"[tiab] OR CNN[tiab] OR "neural network"[tiab] OR "neural networks"[tiab]) AND
(MRI[tiab] OR "magnetic resonance imaging"[tiab] OR "computed tomography"[tiab] OR CT[tiab]
OR PET[tiab] OR "optical coherence tomography"[tiab]) AND (classification[tiab] OR
classifier[tiab] OR classify[tiab] OR "computer-aided diagnosis"[tiab] OR diagnosis[tiab] OR
detection[tiab]) AND (AUC[tiab] OR AUROC[tiab] OR "area under the curve"[tiab] OR "area under
the receiver"[tiab] OR accuracy[tiab] OR sensitivity[tiab] OR specificity[tiab] OR "F1"[tiab])
AND english[la] AND 2019/01/01:2026/12/31[dp] NOT (review[pt] OR "systematic review"[pt] OR
meta-analysis[pt] OR editorial[pt] OR comment[pt] OR "case reports"[pt])
```

**Date run.** 2026-07-29 (UTC).
**Hit count.** **9,979.** All 9,979 PMIDs were retrieved and frozen.
**Frame SHA-256.** `d611def0785f3a5e7b7489364959f1d3471b61651f98a3ed049252654264374b`

### 2.1 Why the query is built this way

Four design choices carry the defensibility, and each one has a failure it prevents.

**The query contains no term that selects on the outcome.** No "slice", no "slice-level", no
"baseline", no "patient-level", no "leakage", no "shortcut". A frame built from any of those
would guarantee the answer. The query selects on *population* — deep learning, a volumetric
modality, a classification task, a performance metric — and nothing else. Everything about how
a paper evaluates is invisible to the query and is decided only at screening.

**All topic terms are `[tiab]`, none are MeSH.** A `[Mesh]` term such as
`"Imaging, Three-Dimensional"[Mesh]` would have raised precision considerably. It was rejected:
MeSH indexing lags by months to years, so a MeSH-gated frame systematically depletes 2025–2026,
and recency is a variable we want to analyse (exploratory subgroup: 2019–2022 vs 2023–2026).
Title/abstract terms are available the moment the record exists.

**Publication types are filtered only negatively.** We tested both forms. Requiring
`journal article[pt]` and merely excluding `review[pt] OR editorial[pt] OR …` gave nearly
identical counts — 1,737 vs 1,738 in 2024, 2,322 vs 2,326 in 2025 — so the positive requirement
bought nothing, and a positive publication-type requirement can only drop records that have not
yet been typed, which again skews recent. The negative-only filter cannot drop an untyped
record, and any review that slips through is caught by exclusion code `E-TYPE` at screening.

**The frame is deliberately broader than the eligible population.** Segmentation papers,
2D-imaging papers and radiomics papers all match the query and are all excluded at screening.
The pilot (§4.2) suggests roughly 30–50% of records will be eligible. Narrowing the query to
raise precision — for instance `NOT segmentation[ti]` — was considered and **rejected**: it
would silently drop multi-task papers that do both segmentation and classification but are
titled around segmentation, which is not a random subgroup. We pay for the frame's honesty in
screening labour, not in bias.

### 2.2 Frame drift, and why the frozen list governs

PubMed keeps indexing retrospectively, so re-running the query in six months will not return
exactly 9,979. This is handled by freezing rather than by hoping: the full PMID list is
committed to `paper/screen/frame_pmids.txt` and **sampling is defined on the frozen list, not
on a live query.** A third party runs `reproduce_frame.py --verify` and gets the same 100 papers
from the repository with no network access at all. `--refetch` re-runs the live query and
reports added/dropped counts; that drift will be reported in the paper as a frame-stability
check, not absorbed.

The canonical ordering before permutation is **de-duplicated, ascending numeric PMID**, because
PubMed's own return order is not stable between calls.

---

## 3. Sampling

**Seed: 20260729** — the UTC date the frame was retrieved. Fixed before the permutation was
drawn, chosen for being non-arbitrary and unpickable, and never changed. Recording the reason
matters: a seed with no stated origin invites the suspicion that several were tried.

**Draw.** `random.Random(20260729).shuffle(frame)`, CPython's Mersenne Twister, whose stream is
stable across versions. The resulting permutation is *also* committed
(`paper/screen/permutation.txt`, SHA-256
`dad12a30b77d1213ac5e8ced89cf3a6620977b5734b5076641bb8adb2db74a1a`) so reproduction is
verifiable by file comparison even if some RNG detail ever changed. Sampling fraction
100/9,979 = 1.002%.

**Allocation, by permutation position:**

| positions | n | role |
|---|---|---|
| 1–15 | 15 | **overlap set** — coded independently by all four screeners |
| 16–37 | 22 | batch A |
| 38–58 | 21 | batch B |
| 59–79 | 21 | batch C |
| 80–100 | 21 | batch D |
| **1–100** | **100** | **analysis sample** |
| 101–110 | 10 | **pilot set — permanently excluded from analysis**, already read by the protocol author (§4.2) |
| 111–400 | 290 | pre-specified reserve, in permutation order, pre-assigned round-robin to A/B/C/D |

Each screener codes 15 overlap + ~21 unique = 36–37 records. The overlap set is positions 1–15
of a random permutation, so it is itself a random subsample of the frame, not a hand-picked
"interesting" set.

### 3.1 Sample size, and the pre-specified extension rule

The primary endpoint is expected to be near zero, so precision is entirely a question of the
upper confidence bound on a zero count:

| included n | estimate if 0 papers report a zero-image baseline | 95% Wilson |
|---|---|---|
| 35 | 0.0% | [0.0%, 9.9%] |
| 60 | 0.0% | [0.0%, 6.0%] |
| **75** | 0.0% | **[0.0%, 4.9%]** |
| 100 | 0.0% | [0.0%, 3.7%] |

"Fewer than 1 in 20" is the claim worth making, so the target is **75 included papers**. The
pilot implies 100 records will yield only ~30–50, so the extension will very probably trigger.
It is therefore fixed here, before any outcome is seen:

> **Extension rule.** Screen positions 1–100. If the number of included papers is below 75,
> continue into the reserve **in permutation order**, in blocks of 50 records, until either 75
> included papers are reached or position 400 is exhausted, whichever comes first. Every record
> in a started block is screened and reported; blocks are never truncated part-way once begun.

This is inverse sampling from a random permutation. It is unbiased for the prevalence, it
requires no judgement once running, and the stopping point is a deterministic function of the
data, not of anyone's preference. If position 400 is exhausted before 75 included papers, we
report whatever n we have with its honest interval and say the target was not met.

**Excluded papers are never replaced by a hand-chosen substitute.** Replacement outside the
permutation order is the classic way a random sample quietly becomes a convenience sample.

---

## 4. Inclusion and exclusion

The full criteria, all ten exclusion codes and all nine ambiguous-case rules live in
`paper/screen_frame.json` under `eligibility`. They are decidable from an abstract plus a
Methods skim. In summary, a paper is **included** when it (I1) is primary research from
2019–2026, (I2) fits a supervised classifier assigning a categorical label, (I3) whose input is
a spatially resolved image from a **volumetric** acquisition — CT, CBCT, MRI, PET, SPECT,
volumetric OCT/OCTA, 3D US, DBT — and (I4) reports a numeric classification metric.

### 4.1 The two rules screeners will get wrong if they are not warned

**A metric is not a task.** Segmentation papers routinely report sensitivity, specificity and
AUC. Three of the ten pilot papers did. Exclusion requires checking what was *evaluated*, not
which metric appeared. Likewise a paper titled "CNN-based glioma **detection** in MRI"
(PMID 39031408) evaluates segmentation with Dice and nothing else — **code the evaluated task,
never the title's verb.**

**Under-described papers must not be excluded for being under-described.** If a paper never says
whether the model saw slices or a volume, it is included with `input_representation='unclear'`.
Dropping the vague ones would systematically remove the least rigorous papers, which biases
every endpoint in the direction that flatters the literature.

### 4.2 The pilot — run before the criteria were fixed

Ten records at permutation positions 101–110 were read (abstracts only) on 2026-07-29 to test
whether the draft criteria were decidable. They are **permanently excluded from the analysis
sample**, because the protocol author has now seen them.

| pos | PMID | provisional | what it tested |
|---|---|---|---|
| 101 | 41732778 | ambiguous | OCTA "images" — volume or en-face projection? Not decidable from the abstract |
| 102 | 39693092 | include | OCT B-scans + en-face + 3D volume arm, AUC reported |
| 103 | 35052273 | ambiguous | CT **and** X-ray in one paper; is a mixed-modality paper eligible? |
| 104 | 39389801 | exclude | YOLOv4+U-Net localisation/segmentation — but reports sensitivity and specificity |
| 105 | 34136106 | include | 3D MRI, 185 patients, AUROC; also does segmentation (multi-task) |
| 106 | 39520662 | exclude | CBCT segmentation, Dice/Hausdorff only |
| 107 | 34924987 | exclude | ABIDE functional-connectivity **matrix** input — no image reaches the model |
| 108 | 41897586 | exclude | U-Net segmentation of OCT foci — reports AUC 0.8411 |
| 109 | 39031408 | exclude | titled "detection", evaluates segmentation with Dice |
| 110 | 38866884 | include | SPECT MPI, 3D reconstructions, AUC 0.91, n=5,443 |

**Six amendments were adopted as a result, all logged in the codebook changelog:**

- **A1 → new code `E-PROJ`.** A volumetric acquisition collapsed to a 2D projection (en-face,
  MIP, scout/localiser) has no slice axis and is excluded. Prompted by 101.
- **A2.** Mixed 2D/3D papers are included if a volumetric arm is reported separately; if results
  are pooled irreducibly across 2D and 3D, include and flag `headline_value_scope`. From 103.
- **A3.** Multi-task segmentation+classification papers are **included**, coding the
  classification arm only and ignoring Dice/IoU entirely. From 105.
- **A4.** Code the evaluated task, not the title's verb. From 109.
- **A5 → new code `E-DERIV`.** Non-spatial derived inputs — connectivity matrices, radiomics
  feature vectors, volumetry tables — are excluded and counted separately, since they are
  outside the failure mode but inside the query. From 107.
- **A6.** The modality enumeration was extended to **SPECT** and **CBCT**, which the draft had
  missed. From 110 and 106.

Provisional yield 3 clear includes, 2 ambiguous, 5 clear excludes — the 30–50% inclusion
estimate that drives the extension rule in §3.1.

---

## 5. The extraction form

Field-by-field in `paper/screen_frame.json` under `fields`; every field has a codebook entry
with a decision rule for the ambiguous case. Three properties of the form are worth stating here
because they are what make the resulting numbers auditable.

**Evidence precedes code.** Fields marked `requires_quote` cannot be submitted without a
verbatim quote and its location (section, page, table or figure). This follows the project rule
that every claim about a paper must quote it, and it is the only defence against coding drift
outside the overlap set.

**The negative must be evidenced too.** Before `trivial_baseline` may be coded all-false, the
screener must full-text search for every one of: *baseline, chance, random, majority, prevalence,
constant, trivial, metadata, clinical-only, clinical model, position, location, slice index,
permut* — including the supplement — and record that they did so in `searches_run`. An
unevidenced negative on the primary endpoint is not accepted.

**"Unclear" is an answer.** Every categorical field has an `unclear` or `not_stated` level, and
those levels are reported as their own category, never imputed and never merged. `split_unit`
carries the sharpest instance: the very common *"the dataset was randomly split 80/20"* with no
unit named is coded `random_unit_not_stated`, and must **not** be upgraded to patient-level
because the word "patients" appears elsewhere in the paper. That distinction is a finding.

The fields the task requires map as follows: evaluation unit → `evaluation_unit_reported`;
which is the headline → `headline_unit`; split unit → `split_unit` (+
`split_disjointness_verified`); **trivial/non-imaging baseline → `trivial_baseline`, the primary
field, six independent sub-flags**; positional distribution → `positional_distribution_reported`;
plus `dataset_name`, `modality`, `organ_or_region`, `n_patients`, `n_patients_test`,
`headline_metric`, `headline_value`; and `fulltext_reachable` + `oa_status`.

---

## 6. Agreement

Fifteen papers (permutation positions 1–15) are coded independently by all four screeners, each
submitted as a sealed file, timestamped in git, before any screener sees another's codes and
before any overlap paper is discussed.

**Fleiss' kappa is the primary statistic, not Cohen's.** Cohen's kappa is defined for two
raters; with four, Fleiss' is the correct generalisation. The six **pairwise Cohen's kappas** are
reported as a matrix alongside it, so the requested statistic is present and the correct one is
primary.

**The kappa paradox is guarded against in advance.** P1 is expected to be extremely skewed —
possibly no paper in the overlap set reports a zero-image baseline. Under that skew kappa
collapses toward 0 even at 100% agreement, which would look like a catastrophic reliability
failure and be entirely an artefact. **Raw percent agreement and Gwet's AC1 are therefore
pre-specified here, before any coding**, and reported in the same table as kappa for every
agreement field. Fixing this now is the difference between a standard robustness measure and a
post-hoc rescue.

Agreement is reported for `final_inclusion`, the **P1 flag**, `evaluation_unit_reported`,
`headline_unit`, `split_unit` and `positional_distribution_reported`. Intervals: bootstrap
percentile 95% over 2,000 resamples of the 15 papers, seed 20260729.

**Threshold and remedy.** If Fleiss' kappa on the P1 flag falls below 0.60 — or, where the
paradox guard applies, if raw agreement falls below 90% — a documented adjudication round is
held, the codebook is amended in the changelog, and **every already-coded paper is re-coded**.
Both pre- and post-reconciliation statistics are reported.

**Outside the overlap set**, where disagreement is invisible by construction, a **20% random
subsample of each screener's batch** (seed 20260729, drawn within batch) is re-coded
independently by the next screener in the cycle A→B→C→D→A, and the disagreement rate on the P1
flag is reported. Any record marked `screener_confidence='low'` or `flag_for_adjudication=true`
is adjudicated by a second screener regardless of batch.

---

## 7. Paywalls, and how unreachable papers are counted

Of the 100 sampled papers, 60 are in PubMed Central, 3 carry a Creative Commons licence in
Crossref, 31 have only a closed or text-mining licence, and 6 have no licence metadata. So
**37 will need work to reach, and some will not be reachable at all.** How those 37 are handled
decides whether the estimate is honest.

**The access ladder.** Try each rung in order, stop at the first that works, record which rung
in `fulltext_reachable`:

1. PubMed Central, or the publisher's own open-access HTML/PDF.
2. The publisher's site directly.
3. Institutional subscription access, where a screener legitimately holds it — record which
   screener.
4. An author's institutional repository, an accepted manuscript, or a preprint version
   (arXiv/medRxiv/bioRxiv/HAL). Record `fulltext_version_used`; a preprint may differ from the
   version of record, and papers coded from anything other than the version of record are
   reported separately and in a sensitivity analysis.
5. Interlibrary loan, or a direct request to the corresponding author. Record the date
   requested; if nothing arrives within **21 days**, code unreachable.

**Not permitted: Sci-Hub or any other unauthorised source.** Stated explicitly so it is not left
to anyone's discretion.

**Counting.** A paper whose abstract is consistent with inclusion but whose full text cannot be
obtained is coded `unreachable_eligibility_unresolved`. It is **not excluded**. It appears in the
flow diagram and in endpoint S6, and:

- the **primary** analysis is complete-case, over included and reachable papers;
- **two bounding analyses are reported alongside it, unconditionally** — a lower bound with every
  unreachable paper coded as *not* reporting a zero-image baseline, and an upper bound with every
  one coded as reporting one;
- if unreachable papers exceed **15%** of the eligible set, **the bounding interval replaces the
  complete-case estimate as the headline number.**

**The direction of the bias is stated in the paper.** Paywalled articles skew toward higher-tier
clinical journals, which are precisely the venues most likely to require a comparator arm. So
silently dropping unreachable papers would push P1 *downward* — in the direction that flatters
our own thesis. That is exactly why the bounding analyses are unconditional rather than
contingent on the missingness looking bad.

---

## 8. Pre-specified analysis

Fixed before any analysis-sample record was read. Full statements in
`paper/screen_frame.json` under `endpoints`.

**Primary.** P1, proportion of included papers reporting ≥1 zero-image baseline (constant /
positional / acquisition-metadata / permuted-label) with a measured value on the same metric.
**Wilson score 95% interval, two-sided.** Wilson is chosen because the estimate is expected near
0, where the Wald interval is degenerate (zero width at k=0) and Clopper–Pearson is needlessly
conservative. Every proportion in the paper uses Wilson, including the secondaries, so no
interval method is ever chosen after seeing a result.

**Secondary,** each with a Wilson interval: S1 any non-imaging baseline including
clinical/demographic-only; S2 proportion whose headline unit is the slice; S3 among papers
reporting any slice-level metric, the proportion also reporting patient-level; S4 proportion
explicitly stating a subject-level split; S5 proportion reporting or discussing the positional
distribution of labels; S6 proportion unreachable; S7 the full cross-tabulation of headline unit
against the P1 flag; S8 proportion reporting a subject-clustered interval; S9 proportion
reporting n positive *patients* and not only n positive slices.

**Exploratory subgroups**, labelled exploratory in every table, **no multiplicity correction, no
tests**: year 2019–2022 vs 2023–2026; modality; public vs private dataset; clinical-radiology
vs engineering/computing venue (classified from the journal's scope statement *before*
unblinding); evaluation unit.

**No significance test is pre-specified for any endpoint.** This is an estimation study. If a
reviewer asks for one it will be reported and labelled post-hoc.

**Reported regardless of which way it comes out.** If P1 turns out to be substantial — if a
respectable fraction of the literature already reports a zero-image baseline — that is the
result, it is the headline, and the paper's contribution narrows to the formalisation, the
statistics and the patient-level contrast. Two of our seven audited datasets already failed to
match (LUNA16, PI-CAI) and are reported as prominently as the five that did; the same rule
applies here.

---

## 9. Flow reporting

A PRISMA-style flow diagram is reported: frame size 9,979 → records screened (100, or more under
the extension rule) → excluded at stage 1 with counts per exclusion code → assessed at full text
→ unreachable → included. Counts per exclusion code are given individually, not as a single
"excluded" total, so a reader can see what the frame's imprecision consisted of. `E-DERIV`
(radiomics/connectivity inputs) is reported separately because those papers are inside the query
and outside the failure mode.

---

## 10. Limitations, stated before the result

**The frame is PubMed-indexed, English-language, journal-published work.** A large share of
methodological medical-imaging work appears at MICCAI, IPMI, CVPR, MIDL and on arXiv and is
never PubMed-indexed. That literature is **not represented**, and the prevalence we report is
the prevalence in the peer-reviewed clinical/biomedical literature — which is the population the
target venue cares about, but it is not "the field".

**We could not pre-register a second frame, and are not pretending otherwise.** On 2026-07-29,
from this environment, the arXiv API (`export.arxiv.org/api/query`) returned an empty body for
every request including the documented example query, and the Semantic Scholar Graph API
returned HTTP 429 without an API key. A query we cannot execute is a query we cannot freeze, so
**no arXiv or Semantic Scholar frame is pre-registered here.** If a CS-venue frame is added
later it must go through this same procedure — query frozen, hit count recorded, full ID list
committed with a digest, seed fixed, criteria piloted — in a dated amendment, before any of its
records are read, and it will be reported as a **separate** prevalence, never pooled with this
one.

**The query is broad and the exclusion rate will be high.** That is a deliberate trade of labour
against bias (§2.1) and it costs precision: the extension rule (§3.1) is the mitigation, and if
it does not reach 75 included papers the interval will simply be wider and said to be.

**OA status in `screen_sample.json` is an automated hint, not a finding.** It is derived from
PMC presence and Crossref licence metadata, both of which understate open access — some genuinely
free articles carry no CC licence in Crossref. Actual reachability is determined by a screener
following the §7 ladder, and it is `fulltext_reachable`, not `oa_status`, that enters any
analysis.

**Single-database screening has no grey literature and no citation chasing.** Deliberate: both
would reintroduce the convenience-sample problem the frame exists to eliminate.

---

## 11. Reproducing this

```bash
# offline: re-derive the frame and permutation digests from the committed files
python paper/screen/reproduce_frame.py --verify

# online: re-run the frozen query today and report drift against the frozen frame
python paper/screen/reproduce_frame.py --refetch
```

`--verify` output on 2026-07-29:

```
[OK ] frame: d611def0785f3a5e7b7489364959f1d3471b61651f98a3ed049252654264374b
[OK ] permutation-on-disk: dad12a30b77d1213ac5e8ced89cf3a6620977b5734b5076641bb8adb2db74a1a
[OK ] permutation-recomputed: dad12a30b77d1213ac5e8ced89cf3a6620977b5734b5076641bb8adb2db74a1a
[OK ] frame size: 9979 (expected 9979)
```

The timestamp of the git commit that adds these five files is the pre-registration timestamp.
Deposit on OSF, with that commit hash recorded in the deposit, is an outstanding action item and
should happen before screening begins.

---

## 12. Changelog

| date | version | event |
|---|---|---|
| 2026-07-29 | 0.1 | Draft criteria and endpoints written from the research question, before any record was read. |
| 2026-07-29 | 0.2 | Query designed and tested for population coverage only. `[Mesh]` gating and positive publication-type filtering tested and rejected (§2.1). Narrowing on `NOT segmentation[ti]` considered and rejected. |
| 2026-07-29 | 0.3 | Frame executed: 9,979 hits, all PMIDs frozen. Seed 20260729 fixed; permutation drawn and frozen. |
| 2026-07-29 | 0.4 | Pilot of 10 records (positions 101–110). Six amendments A1–A6 adopted; codes `E-PROJ` and `E-DERIV` created; modality list extended to SPECT and CBCT. Pilot records permanently excluded from analysis. |
| 2026-07-29 | **1.0** | **FROZEN.** Extraction form, agreement plan, paywall handling, endpoints and analysis fixed. No analysis-sample record read. |
| 2026-07-29 | 1.1 | **Metadata correction, no change to the sample.** The first build read identifiers with `iter("ArticleId")`, which also walks each record's embedded reference list, so 229/400 DOIs and 276/400 PMCIDs were taken from a cited paper instead of the article — PMID 40335658, a 2025 *Eur Radiol* paper, had been given a 1985 *JBJS* DOI, and `oa_status` was wrong wherever the PMCID was. Fixed to scoped lookups (`PubmedData/ArticleIdList` direct child, `ELocationID` fallback) in `build_sample.py`, which now carries a seeded 25-DOI Crossref title-match regression test: 25/25 agree. §7 open-access counts were restated (60 PMC / 3 CC / 31 closed / 6 unknown; previously 65/3/27/5). **Which PMIDs are sampled and which batch each falls in were never affected** — allocation is determined solely by `permutation.txt`, whose digest is unchanged. |

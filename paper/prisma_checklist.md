# PRISMA 2020 checklist, adapted for a random-sample meta-research screen

Scope: the **prevalence screen** component of the manuscript only. The benchmark audit is a
different kind of study and PRISMA does not govern it; where an item is answered by audit
material that is said explicitly.

Reference: Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated
guideline for reporting systematic reviews. *BMJ* 2021;372:n71. doi:10.1136/bmj.n71.

## Status vocabulary, and the rule for using it

| status | meaning |
|---|---|
| **addressed** | done, and the location column says where a reader finds it |
| **adapted** | the item applies but not in its literal form; the adaptation and its rationale are stated in the notes |
| **not applicable** | the item presupposes something this study does not have; the rationale is stated |
| **not done** | the item applies and is not satisfied. No excuse in the status column |

**The rule.** An item is marked *addressed* only if the material exists **in the manuscript or
in a released artefact a reader can open**, not if the data merely exist on disk in a form
nobody has assembled. Several items below are marked *not done* on exactly that ground, with
the assembling work named. Nothing is marked *not applicable* to avoid work; every such mark
carries a reason a methodologist can argue with.

## Why an adapted checklist rather than the standard one

PRISMA 2020 was written for systematic reviews of studies that estimate the effects of
interventions or the accuracy of tests. This screen estimates **how often a class of published
paper reports a particular kind of comparator** — the unit of observation is a sentence in a
Methods or Results section, quoted verbatim, not an effect estimate. Three consequences run
through the table below and are stated once here:

1. **Sampling replaces census.** A seeded random sample of 100 records is drawn from a frozen
   9,979-record frame instead of screening every record. This adds a box to the flow diagram
   and makes the frame digest, the seed and the permutation part of the reporting requirement.
   Items 7, 8 and 16a are adapted accordingly.
2. **Risk of bias and certainty of evidence do not transfer.** There is no effect estimate whose
   internal validity could be graded, and GRADE has no defined application to "does this paper
   report a null baseline". Items 11, 15, 18 and 22 are marked *not applicable*, each with a
   rationale, and the analogous constructs that **do** exist — mandatory verbatim quotes,
   mandatory pre-negative full-text searches, screener confidence, the unreachability bound —
   are named so the item is not simply dropped.
3. **Reporting bias becomes access bias.** Publication bias in the usual sense is not the threat.
   The threat is that unreachable papers are not a random subset: paywalled articles skew toward
   higher-tier clinical journals, which are the venues most likely to require a comparator arm.
   Item 14 is adapted to that, and the direction of the bias — *against* the paper's own thesis
   — is pre-stated in the protocol.

This adaptation is declared here rather than performed silently. Where an item was dropped, it
was dropped in writing.

---

## The checklist

Section references: `D` = `paper/DRAFT.md`; `P` = `paper/screen_protocol.md`;
`F` = `paper/screen_frame.json`.

### Title

| # | item | status | where | notes |
|---|---|---|---|---|
| 1 | Identify the report as a systematic review | **adapted**, action required | `D` title | The title says "a pre-registered screen of how often the check is reported". Two fixes are needed: it does not use the words "systematic review" (correct — it is a random-sample prevalence screen and calling it a systematic review would overclaim), and it **must not say "pre-registered"** unqualified, because no registry deposit was made. See `paper/registration.md` §5.4. Proposed: "a protocol-frozen random-sample screen". |

### Abstract

| # | item | status | where | notes |
|---|---|---|---|---|
| 2 | Abstract per the PRISMA 2020 for Abstracts checklist | **not done** | `D` Abstract | The abstract has never been checked against the 12-item abstracts checklist. It currently reports the frame, the sample and the design but not the number of included studies, not the flow, and not the registration status; and it uses "pre-registered". npj Digital Medicine will expect the abstracts checklist for a review-type component. **Action:** run the 12-item checklist against the abstract and correct the registration wording. |

### Introduction

| # | item | status | where | notes |
|---|---|---|---|---|
| 3 | Rationale for the review in the context of existing knowledge | **addressed** | `D` §1; `P` §1 | The rationale is explicit: the audit demonstrates a failure mode; the screen exists to say how often it goes unchecked, and the protocol argues at length why a convenience sample would be worthless. Prior art (Badgeley 2019; Yan 2018; the 2020 Kaggle notebook; Ong Ly 2024) is credited in `D` §4.5. |
| 4 | Explicit statement of objectives / questions | **addressed** | `P` §1, §8; `F` → `endpoints` | Primary P1 and secondaries S1–S9 are each defined in one sentence with the interval method fixed. |

### Methods

| # | item | status | where | notes |
|---|---|---|---|---|
| 5 | Eligibility criteria and how studies were grouped | **addressed** | `P` §4; `F` → `eligibility` | Four inclusion criteria I1–I4 and ten exclusion codes, each decidable from an abstract plus a Methods skim, plus nine named ambiguous-case rules and, after v1.2, fourteen more (D1–D14). Grouping for the syntheses is by evaluation unit and by the P1 flag. |
| 6 | Information sources, with the date each was last searched | **addressed, with a stated limitation** | `P` §2, §10; `paper/screen/frame_meta.json` | **One** database: PubMed via NCBI `esearch.fcgi`, run 2026-07-29 06:42:52 UTC, 9,979 hits, full PMID list frozen and digested. No registers, no grey literature, no citation chasing, no contact with authors. `P` §10 states before the result that MICCAI/IPMI/CVPR/MIDL/arXiv work is **not represented** and that the estimate is a prevalence in the peer-reviewed biomedical literature, not "the field". Single-database searching is a real deviation from best practice and is defended, not hidden: it is the trade made to keep the frame executable from a script and therefore reproducible. |
| 7 | Full search strategy for every source | **addressed** | `P` §2; `paper/screen/reproduce_frame.py`; `frame_meta.json` | The complete Boolean string is given verbatim in three places and is a frozen constant in the released script. `--verify` re-derives the frame and permutation digests offline; `--refetch` re-runs the query live and reports drift against the frozen frame rather than absorbing it. |
| 8 | Selection process: how many reviewers, independence, tools | **adapted; addressed in part, with a named deviation** | `P` §3, §6; `paper/prisma_flow.md` | Four screeners. The 15-paper overlap set (permutation positions 1–15) was coded **independently by all four**, each submitting a sealed file with a written independence statement. The remaining **85 records were coded by a single screener each** — a deliberate design choice, declared in the protocol, with the 20% within-batch duplicate re-code as the mitigation. **That mitigation was not executed**, nor was second-screener adjudication of the 49 single-screened records flagged low-confidence or `flag_for_adjudication`. No automation tool was used at any selection stage. The sampling step and its digests are reported as part of selection, which is the adaptation. |
| 9 | Data collection process: reviewers, independence, tools | **addressed** | `P` §5; `F` → `fields`, `reading_effort` | One frozen extraction form. Fields marked `requires_quote` cannot be submitted without a verbatim quote and its location. **A negative on the primary flag must itself be evidenced**: before `trivial_baseline` may be coded all-false the screener must full-text search all fourteen of *baseline, chance, random, majority, prevalence, constant, trivial, metadata, clinical-only, clinical model, position, location, slice index, permut*, including the supplement, and record having done so in `searches_run`. No automation, no author contact. |
| 10a | Outcomes: list and define all, and state whether all compatible results were sought | **addressed** | `P` §8; `F` → `endpoints`; `headline_selection_rule` | P1 and S1–S9 defined in advance with Wilson intervals fixed for all of them, so no interval method can be chosen after seeing a result. Where a paper reports several headline numbers, `headline_selection_rule` names the rule used to pick one (abstract sentence first) and `headline_metric_qualifier` records what was passed over. |
| 10b | Other variables: list and define, with assumptions about missing data | **addressed** | `P` §5; `F` → `fields` | Dataset, modality, organ, n patients, n slices, metric, value, test-set type, input representation, code availability, and more. Every categorical field carries an `unclear` / `not_stated` level, and **those levels are reported as their own category, never imputed and never merged**. `split_unit` carries the sharpest instance: "randomly split 80/20" with no unit named is coded `random_unit_not_stated` and must not be upgraded because the word "patients" appears elsewhere. |
| 11 | Study risk-of-bias assessment | **not applicable** (adapted) | `F` → `fields`; `P` §5 | No risk-of-bias tool was applied and none is appropriate: the observation is *what a paper says about its own evaluation*, quoted verbatim, not an effect estimate whose internal validity could bias a pooled result. A miscoded paper is a measurement error, addressed by agreement statistics, not a biased study. The nearest analogues are reported and should be read as this item's substitute: `screener_confidence`, `flag_for_adjudication`, `split_disjointness_verified`, and the requirement that every code carry its quote. |
| 12 | Effect measures | **addressed** | `P` §8 | Proportions. Wilson score 95% two-sided, chosen in advance because the primary estimate was expected near zero where Wald is degenerate at k = 0 and Clopper–Pearson is needlessly conservative. No effect measure in the clinical sense exists. No significance test is pre-specified for any endpoint; this is an estimation study. |
| 13a | Which studies were eligible for each synthesis | **addressed** | `F` → `endpoints._denominator_default`; `paper/prisma_flow.md` | The default denominator is included **and** full text obtained. A paper whose full text was never read cannot support an evidenced negative on P1, so it is excluded from the complete-case denominator by rule and carried in the bounding analyses instead. |
| 13b | Data preparation before synthesis | **addressed, and a codebook defect reported rather than hidden** | `paper/screen/analysis/pool_and_agree.py` docstring; `analysis_out.json` → `codebook_audit` | The six `trivial_baseline` sub-flags were declared boolean with no level for "could not be assessed", so four screeners independently invented four conventions (`false`, `null`, `"unclear"`, `false`). The analysis reads the flags **three-valued** and reports the defect; coercing `"unclear"` to truthy drives Fleiss' κ to −0.18, and that number is reported too, because it is what the pre-registered field would give without the correction. Overlap records are pooled by majority with ties broken **against the paper's own thesis**. |
| 13c | Tabulation and visual display methods | **addressed** | `D` Tables 5–6; `paper/prisma_flow.md`; `paper/figures/prisma_flow.svg` | Every count in the flow document and the figure is generated by `paper/screen/analysis/prisma_flow.py` from the named inputs and asserted against the published endpoint files. |
| 13d | Synthesis method | **addressed** | `P` §7, §8 | No meta-analysis: proportions with Wilson intervals. Two bounding analyses over the unreachable records are reported **unconditionally**, and past a pre-registered 15% censoring threshold the bounding interval replaces the point estimate as the headline. Observed censoring is 29.6%, so the bound is the headline. |
| 13e | Heterogeneity exploration (subgroup analysis, meta-regression) | **not done** | `P` §8 | Five exploratory subgroups were pre-specified — year band 2019–2022 vs 2023–2026, modality, public vs private dataset, clinical-radiology vs engineering venue, evaluation unit — and **only the last was computed** (as S7, headline unit × P1). The other four have not been run. Since P1 is zero in every included paper, every subgroup of it is also zero and the analyses would be uninformative, but that is a reason to state the fact, not to leave the item unmarked. **Action:** either run the four remaining subgroups on the secondary endpoints or state in the manuscript that they were pre-specified and not run. Adding subgroups that were *not* pre-specified is not an option. |
| 13f | Sensitivity analyses | **addressed in part** | `P` §7; `paper/screen/analysis/recompute_with_recovery.py` | The unconditional bounding analyses over the unreachable set are the principal pre-registered sensitivity analysis, and the as-sealed / post-recovery contrast is printed side by side. **One pre-registered sensitivity analysis is outstanding:** `P` §7 requires that papers coded from anything other than the version of record be reported separately and in a sensitivity analysis. PMID 36200353 was coded from an Authorea preprint. With a single such record the analysis is a one-line leave-one-out, and it has not been run. |
| 14 | Reporting bias assessment | **adapted; addressed** | `P` §7 | Publication bias in the funnel-plot sense has no meaning here. The analogous threat is **access bias**, and the protocol states its direction *before* the result: paywalled articles skew toward higher-tier clinical journals, precisely the venues most likely to require a comparator arm, so silently dropping unreachable papers would push the primary estimate downward — in the direction that flatters the paper's own thesis. That is why the bounding analyses are unconditional rather than contingent on the missingness looking bad. |
| 15 | Certainty assessment | **not applicable** (adapted) | `D` §5.1; `paper/prisma_flow.md` | GRADE assesses certainty in a body of evidence about an effect. There is no effect. Nothing is graded and nothing is described as high or low certainty. What is reported instead, and should be read as this item's substitute: the upper Wilson bound on a zero count, the censoring bound that replaces the point estimate, the κ that must be quoted in the same sentence as S4, and an explicit statement that absence in a sample is not absence in a literature. |

### Results

| # | item | status | where | notes |
|---|---|---|---|---|
| 16a | Study selection: numbers screened, assessed and included, with a flow diagram | **addressed** | `paper/prisma_flow.md`; `paper/figures/prisma_flow.svg`; `paper/prisma_flow.json` | Flow at all three protocol versions (v1.0 as sealed, v1.2 adjudicated, v1.3 post-recovery), so the effect of each amendment is visible line by line. The generating script cross-checks its own totals against `analysis_out.json` and `recovery_out.json` and fails rather than emit a figure that disagrees with the published endpoints. |
| 16b | Studies that might appear eligible but were excluded, with reasons | **addressed in part** | `paper/screen_batch_{A,B,C,D}.json`; `paper/prisma_flow.md` | Every one of the 46 excluded records carries its exclusion code and, where required, a verbatim `exclusion_quote`, in the released sealed files. Exclusion counts are given per code and per stage rather than as one "excluded" total, as `P` §9 requires, so a reader can see what the frame's imprecision consisted of. **Missing:** no citation-level table of the excluded records appears in the manuscript or its supplement; a reader currently has to open a JSON file. **Action:** emit a supplementary table of PMID, citation, stage and code. |
| 17 | Characteristics of each included study | **not done** | data exist in `paper/screen_batch_*.json` | The fields are collected — dataset, modality, organ, n patients, n slices, metric, value, test-set type, input representation, code availability — and marginal distributions over the 35-record complete case are in `analysis_out.json` → `distributions_complete_case`. **No per-study characteristics table has been assembled**, and the marginals have not been regenerated on the 38-record post-recovery denominator. **Action:** one script over the batch files plus the recovery overlay. |
| 18 | Risk of bias in each included study | **not applicable** | — | See item 11. |
| 19 | Results of each individual study | **not done** | data exist in `paper/screen_batch_*.json` | Each record carries its headline metric, value, scope and the quote that supports it. As with item 17, no per-study results table has been assembled for the manuscript. Same action, same script. |
| 20a | For each synthesis, characteristics and risk of bias of contributing studies | **addressed in part** | `analysis_out.json` → `distributions_complete_case`, `S7_headline_unit_by_P1` | Contributing-study characteristics are available as marginals and as the S7 cross-tabulation; the risk-of-bias half is not applicable per item 11. Blocked on the same missing table as items 17 and 19, and on regeneration at n = 38. |
| 20b | Results of each synthesis, with precision and direction | **addressed; the manuscript is stale** | `D` Table 5; `analysis_out.json` → `endpoints`; `recovery_out.json` → `after` | All endpoints are reported with Wilson intervals, with complete-case and both bounds side by side. **`D` Table 5 and `D` §5.1 still carry the v1.0 as-sealed numbers** (35 included, 20 unreachable, 36.4% censoring, and a claim that the agreement remedy is outstanding when it has since been executed as v1.2). The current primary values are 38 included, 16 unreachable, 29.6% censoring, P1 = 0/38 with headline bound [0.0%, 29.6%]. **Action:** bring Table 5, §3.10, §5.1 and the abstract onto the post-recovery numbers, or state explicitly which version each reports. |
| 20c | Results of heterogeneity investigations | **not done** | — | See item 13e. Four of five pre-specified exploratory subgroups were not run. |
| 20d | Results of sensitivity analyses | **addressed in part** | `recovery_out.json`; `adjudication_out.json` | The bounding analyses, the as-sealed / post-recovery contrast, and the pre- versus post-amendment endpoint contrast are all reported. The version-of-record sensitivity analysis required by `P` §7 is outstanding (item 13f). |
| 21 | Reporting biases affecting the synthesis | **adapted; addressed** | `P` §7; `D` §5.1; `paper/screen/access_recovery.json` | Censoring is 29.6% of the eligible set, about twice the threshold at which the protocol converts the headline to a bound. The direction of the resulting bias is stated and runs against the paper's own thesis. Three still-unreachable records are **demonstrably open access** and unreachable only because this execution environment is refused by Wiley, RSNA and Elsevier; they are counted as unreachable because no full text was read, and the cause is disclosed rather than charged to the literature. |
| 22 | Certainty of evidence for each outcome | **not applicable** | — | See item 15. |

### Discussion

| # | item | status | where | notes |
|---|---|---|---|---|
| 23a | General interpretation in the context of other evidence | **addressed** | `D` §4, §4.5 | Prior art is credited rather than competed with: Badgeley 2019 for the metadata arm, Yan 2018 Table 1 and the 2020 Kaggle notebook for the positional construction, Ong Ly 2024 as the closest competitor. The claimed contribution is narrowed accordingly. |
| 23b | Limitations of the evidence included | **addressed** | `D` §5.1; `P` §10 | Frame limited to PubMed-indexed English-language journal work; the CS-venue literature is stated to be unrepresented, before the result. "Absence in a sample is not absence in a literature" is stated and the manuscript is asserted to contain no sentence beginning "no published paper". |
| 23c | Limitations of the review processes used | **addressed in part** | `D` §5.1; `paper/prisma_flow.md`; `paper/registration.md` | `D` §5.1 covers the agreement failure, the censoring, the undersized sample, the unreliable `split_unit` field and the absence of screener blinding. It does **not** yet cover: single screening of 85 of 100 records with the mitigation unexecuted; the unexecuted access-ladder rung 5; the unexecuted second-screener adjudication of 49 flagged records; and that the post-amendment reliability figure is a counterfactual re-encoding rather than an independent re-rating. It also still states that the agreement remedy is outstanding, which v1.2 superseded. **Action:** fold the deviation list from `paper/prisma_flow.md` into §5.1 and correct the stale sentence. |
| 23d | Implications for practice, policy and future research | **addressed** | `D` §4.4, §6 | Concrete: what benchmark publishers should release, and the seven-rule reporting protocol with its one-page checklist. |

### Other information

| # | item | status | where | notes |
|---|---|---|---|---|
| 24a | Registration name, registry and number, or a statement that the review was not registered | **not done in the manuscript; the honest statement is drafted** | `paper/registration.md` §3, §5.2 | **The review was not registered.** No OSF, PROSPERO or other registry deposit was made before screening; the protocol named an OSF deposit as an action item to be completed first and it was not done. The protocol, codebook, frame, permutation and sample were committed to a public repository before any analysis-sample record was coded (commit `a64d202`, 2026-07-29 07:12 UTC), which is a weaker and different guarantee. Wording that states this without implying prospective registration is drafted in `paper/registration.md` §5.2. **Action:** paste it into the manuscript, and make any deposit retrospectively and label it so. |
| 24b | Where the protocol can be accessed, or a statement that one was not prepared | **addressed** | `paper/screen_protocol.md`; `paper/screen_frame.json` | Both released in full with their changelogs. The version frozen before screening is recoverable as `git show a64d202:paper/screen_protocol.md`. |
| 24c | Amendments to information provided at registration or in the protocol | **addressed** | `P` §12; `paper/registration.md` §4 | Three amendments — v1.1 metadata correction with no effect on the sample; v1.2 codebook amendment executing the protocol's own pre-specified agreement remedy; v1.3 second pass of the access ladder — each dated, each with the trigger, the endpoints affected and the **direction** of the change. The v1.2 amendment moved one secondary endpoint in the direction that makes the literature look better, and that is stated. |
| 25 | Sources of financial and non-financial support, and the role of funders | **not done** | template in `paper/authorship.md` | No funding statement has been written. npj Digital Medicine requires one even when the answer is "none". |
| 26 | Competing interests | **not done** | template in `paper/authorship.md` | No competing-interests statement has been written, and none has been collected from the co-authors. Each author must declare individually. |
| 27 | Availability of data, code and other materials | **addressed, with one gap** | `D` "Data and code availability" | The frozen protocol and codebook, the frame with its digest, the seeded permutation, the four sealed screener files with every quote and search string, the analysis scripts and their outputs, and the `trivialbaselines` tool are all released, and the 16 unreachable PMIDs are listed so a reader with better access can finish the screen. **Gap:** the Zenodo deposit has metadata prepared (`trivialbaselines/.zenodo.json`, `CITATION.cff`) but **no DOI has been minted**, so every "Zenodo DOI" reference in the manuscript is currently a placeholder. Two smaller gaps are flagged in the draft itself: the install instructions disagree between files, and the retrieved full texts are not redistributable (URLs plus verbatim quotes are the audit trail instead). |

---

## Summary

| status | items |
|---|---|
| **addressed** | 3, 4, 5, 6, 7, 9, 10a, 10b, 12, 13a, 13b, 13c, 13d, 14, 16a, 20b (stale), 21, 23a, 23b, 23d, 24b, 24c, 27 (one gap) |
| **adapted** (and addressed under the adaptation) | 1 (action required), 8, 14, 16a, 21 |
| **addressed in part** | 8, 13f, 16b, 20a, 20d, 23c |
| **not applicable**, with rationale | 11, 15, 18, 22 |
| **not done** | 2, 13e, 17, 19, 20c, 24a, 25, 26 |

Eight items are **not done**. Four of them — 2, 24a, 25, 26 — are writing, and can be closed in
an afternoon using `paper/registration.md` §5 and `paper/authorship.md`. Three — 17, 19, 20a —
are a single script over data already collected. One — 13e / 20c — is a decision about whether
to run four pre-specified subgroups or to say in print that they were pre-specified and not run.
None of the eight requires new screening.

The items that would need new labour, and that no amount of writing can close, are recorded in
`paper/prisma_flow.md` as protocol deviations rather than as checklist items: the unexecuted
sample extension, the unexecuted access-ladder rung 5, the unexecuted duplicate re-coding, and
the absent independent post-amendment reliability estimate.

# Pre-submission enquiry — npj Digital Medicine

**Status: DRAFT, READY TO SEND. NOT SENT.** A human sends this.

**Route.** Send through npj Digital Medicine's own pre-submission enquiry channel — the editorial
office address or web form published on the journal's "Contact"/"For Authors" page at the time of
sending. **Verify that address on the journal's site; do not guess it, and do not address a named
editor unless the site names one for this journal.**

**Length.** The two paragraphs are **447 words** (238 + 209) and paste in as-is. If the journal's
form imposes a hard limit below that, cut **in this order, and only in this order**: (1) the
`(n = 9)` and the PEst clause "it targets the unit of evaluation rather than acquisition bias";
(2) the release list, down to "we release the auditor and the screen's materials"; (3) the
LUNA16 / PI-CAI clause. **Never** cut the censoring sentence, the bound `[0.0%, 36.4%]`, the
interim-n clause, the agreement-failure clause, or the preprint clause — each of those is the
letter's insurance against an editor discovering the same fact later, and removing one to save
twenty words converts an honest letter into a misleading one.

**Fill in before sending:** `[YOUR NAME, AFFILIATION, EMAIL]` (a named corresponding author — on
the current author list, Sathvik Loke, Illinois Mathematics and Science Academy,
`trivialbaselines/CITATION.cff`), and the date.

**Governing plan:** `paper/PAPER_PLAN.md` §3.2 — lead with the screen, name Ong Ly et al. in the
first sentence, never state 0% without the bound, never describe the screen as complete.

---

## Metadata block (precedes the two paragraphs)

**Article type:** Article (meta-research / methods)
**Working title:** *What a slice-level benchmark certifies without the pixels: a label-file audit
of seven public benchmarks, and a pre-registered screen of how often the check is reported*
**Corresponding author:** [YOUR NAME, AFFILIATION, EMAIL]
**Status:** Complete draft, in internal revision (re-anchored 2026-07-29); prevalence-screen
extension in progress. Full abstract available on request.

---

## The enquiry

> Dear Editors,
>
> Ong Ly et al. (npj Digit Med 2024;7:124) measured how much shortcut learning of hidden
> acquisition bias inflates reported medical-AI performance; we have measured how often anyone
> checks. In a pre-registered screen of the volumetric-imaging classification literature — 100
> papers drawn by seeded permutation from a frozen 9,979-record PubMed frame, coded independently
> by four screeners against a frozen extraction form — **not one paper we could read reported a
> measured zero-image baseline of any kind, and not one reported the positional distribution of
> its labels.** We report that as a bound rather than a point estimate: 20 of the 55 eligible
> papers (36.4% [24.9%, 49.6%]) could not be obtained in full text, exceeding our pre-specified
> 15% censoring threshold, so the protocol makes **[0.0%, 36.4%]** the headline and the
> complete-case reading (0 of 35, 0.0% [0.0%, 9.9%]) a companion never quoted alone. One form of
> the result is censoring-free: across all 145 coded records, including every excluded and
> unreachable paper, none carries a positive code on any zero-image sub-flag. Two caveats, from us
> rather than from a referee: the screen is interim (n = 35 against a pre-registered target of 75,
> extension running), and inter-rater agreement on the primary flag failed its own pre-specified
> floor — for a diagnosed reason we report in full, the frozen form having had no "could not be
> assessed" level; on the six overlap papers where the code is defined, all four screeners agreed.
>
> The screen accompanies an audit that supplies the check it finds missing. From a published label
> file alone — subject identifier, slice index, label, split — we fit pixel-blind null models and
> read them at both the slice and the patient level. On RSNA 2019 Intracranial Haemorrhage
> (752,802 slices, 18,938 patients, official metric per-slice), one positional score vector reads
> **0.737 [0.735, 0.740] at the slice level and 0.453 [0.445, 0.461] at the patient level**;
> nothing changes but the unit of evaluation, and no published number of anyone's is involved.
> Against peer-reviewed comparators the pixel-blind model reaches a median 0.469 of the published
> margin over chance (n = 9); LUNA16 and PI-CAI do not fire at all and are reported at equal
> prominence; the only rows matching a published number outright have a preprint comparator and
> are labelled a worked example, not evidence. Three things distinguish this from PEst: it needs
> **no pixel access**, so a third party can audit benchmarks they will never hold; it targets the
> **unit of evaluation** rather than acquisition bias; and we release the auditor (MIT,
> `numpy`/`pandas` only) with the screen's protocol, frozen frame, sealed screener files and
> analysis script. Would this be of interest, and if so as an Article or a Brief Communication?
>
> Yours sincerely,
> **[YOUR NAME, AFFILIATION, EMAIL]**

---

## Appendix — every number in the enquiry, and where it came from

| Claim | Value | Source on disk |
|---|---|---|
| Ong Ly et al. citation | npj Digit Med 2024;7:124, doi 10.1038/s41746-024-01118-4 (PMID 38744921) | retrieved from PubMed, 2026-07-29; the "up to 20%" and PEst description are the paper's own abstract |
| Frame | 9,979 PMIDs, PubMed, query and SHA-256 frozen | `paper/screen_protocol.md` §2; `paper/screen/frame_pmids.txt` |
| Sample | 100, seeded permutation (seed 20260729), 4 independent screeners, 15-paper overlap | `paper/screen_protocol.md` §§3–4; `paper/screen_batch_{A,B,C,D}.json` |
| Primary endpoint, complete case | 0/35, 0.0% [0.0%, 9.9%] | `paper/screen/analysis/analysis_out.json`, `endpoints.P1_complete_case` |
| **Headline bound** | [0.0%, 36.4%] | same, `P1_lower_bound` / `P1_upper_bound`; the "HEADLINE" note is in the file |
| Unreachable rate | 20/55, 36.4% [24.9%, 49.6%] | same, `S6_unreachable` |
| Positional distribution never reported | 0/35, 0.0% [0.0%, 9.9%] | same, `S5_positional_distribution` |
| 145 coded records, no positive sub-flag | — | `paper/DRAFT.md` §3.10; `analysis_out.json`, `per_record` |
| Interim n, target 75, extension triggered | — | `analysis_out.json`, `flow.protocol_target_included` / `extension_rule_triggered` |
| Agreement failure and its diagnosis | raw 65.6%, Fleiss' κ −0.015 [−0.164, 0.120] vs a 90% / 0.60 floor; unanimous on the 6 defined overlap papers | `analysis_out.json`, `agreement`; `paper/DRAFT.md` §3.10, §5.1 |
| RSNA ICH, both units | 0.737 [0.735, 0.740] slice; 0.453 [0.445, 0.461] patient; 752,802 slices; 18,938 patients | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`, label `any` |
| Median trivial fraction vs peer-reviewed comparators | 0.469, IQR 0.437–0.490, n = 9 | `paper/trivial_fraction_distribution.{json,md}`; `paper/DRAFT.md` Abstract, §3.1 |
| LUNA16 / PI-CAI nulls | LUNA16 CPM 0.0020 and sensitivity 0.0006 at 1 FP/scan, *below* a random-score reference CPM of 0.0027 (trivial fraction −0.002); PI-CAI positional exactly 0.500 | LUNA16: `paper/audit_results.md` §3.6 (`pipeline/audit_prep/luna16_cpm.py`), row 10 of the distribution table. PI-CAI: `pipeline_out/trivial_baselines/picai_case_level.md`, `positional_20bin` row |
| Preprint-anchored rows | the six matched rows all rest on arXiv:2407.06165, no journal reference as of 2026-07-29 | `paper/audit_results.md` §2.3; `paper/DRAFT.md` §3.2, §5 |

### Things this letter deliberately does not say

- **It does not claim the screen is finished.** It is n = 35 against a target of 75 and says so.
- **It does not quote 0% as a prevalence.** The protocol makes the bound the headline once
  censoring exceeds 15%, and the letter reports the bound first.
- **It does not hide the agreement failure or the preprint dependency.** An editor who found
  either one later would discount everything else in the letter; both are named here in the same
  sentences that make the positive claim.
- **It does not claim novelty for the positional construction.** The claim is the formalisation,
  the statistics, the patient-level contrast, the prevalence measurement and the tool — nothing
  more. If the editor asks what is new, that is the answer, and it is the same answer the
  manuscript gives (`paper/DRAFT.md` §4.5, §5).

**One blocker before this is sent.** `paper/DRAFT.md` credits Yan et al. (CVPR 2018) and Badgeley
et al. (npj Digit Med 2019) as prior art, but **does not yet mention OsciiArt's 2020 Kaggle
notebook "Baseline with no image" (RSNA-STR PE)** — the same construction four years earlier,
recorded in `paper/FINDINGS.md` prior-art table and in `paper/audit_results.md`. Nothing in this
enquiry depends on it, but the manuscript must cite it before submission, and it should be cited
before the editor is invited to read the manuscript. `paper/COLLABORATORS.md` §5.1 already flags
that the prior-art search needs redoing against Google Scholar, MICCAI/MIDL/ML4H and the RSNA ICH
Kaggle write-ups; that job is upstream of this letter.

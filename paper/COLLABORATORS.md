# What this paper needs from humans

Written 2026-07-29, to accompany `paper/PAPER_PLAN.md` and `paper/DRAFT.md`.

Four things in this paper cannot be produced from the repository, no matter how much compute
is thrown at it. Each is below with the exact ask, why it is load-bearing, what happens if it
is skipped, and — where an email is the right instrument — the actual text.

> **These emails are drafts. Nothing here has been sent, and nothing here should be sent by
> anything other than a human who has read it.** Every one of them commits the sender to a
> position in public, and two of them are addressed to people whose work this paper
> criticises.

Priority order, by turnaround time rather than importance:

| # | ask | who | lead time | blocks |
|---|---|---|---|---|
| 4 | courtesy notice to Rempe et al. | the audited authors | 3–6 weeks | nothing formally, but should precede submission |
| 1 | senior radiologist co-author | clinical collaborator | 2–4 weeks | §3.3 of the reviewer defence; the clinical paragraph in §3 below |
| 2 | biostatistics review | statistician, ideally not a co-author | 2–3 weeks | the MATCHED rule and the trivial fraction's interval |
| 3 | one-paragraph clinical confirmation | the same radiologist | 1 week, after they have read the draft | the anatomy objection |

---

## 1. Senior radiologist co-author

### What we need, item by item

Six specific things. Not "please review the manuscript" — that produces a nod and no content.

**1.1 The anatomy question, answered on the record.**
A reviewer will say: *prostate cancer really does concentrate in the mid-gland; your
"positional null" is just anatomy.* We need a co-author who can write the paragraph in §3 of
this document and stand behind it. Our answer is that this is exactly the point — an
evaluation protocol that credits anatomy to a model is the problem — but that answer is only
credible from someone who reads prostate MRI.

**1.2 Whether slice-level AUROC is ever the clinically relevant metric.**
Specifically: are there real clinical tasks where the slice is the decision unit? Lesion
localisation for biopsy targeting is a candidate. If yes, we must carve the scope explicitly
and say "where the clinical question really is localisation, report it as localisation *and*
report the patient-level number, because readers will otherwise assume it". If no, the
critique is broader than currently written and should say so.

**1.3 A read on the Duke breast slice task.**
The data owners define positives as slices inside the tumour bounding box and negatives as
slices at least five away, with everything between discarded, and every patient in the cohort
has cancer. We describe the resulting 0.823 positional baseline as a tautology and the task as
within-patient localisation rather than diagnosis. Is that the right clinical description? Is
the five-slice exclusion band clinically motivated or arbitrary?

**1.4 A read on the fastMRI+ label caveat.**
The maintainers describe their annotations as an indication of where a pathology could be
present rather than adjudicated ground truth. Does that change how much weight the 0.873
meniscus-tear row can carry?

**1.5 Wording review with one specific instruction.**
Every conclusion in this paper is about an *evaluation protocol*, never about a model's
internals. The failure mode is a sentence drifting into "the model learned nothing". We want
a co-author who will hunt for that sentence specifically. `paper/PAPER_PLAN.md` §2 is the
table of forbidden claims; ask them to read that first and then the draft.

**1.6 Whether the checklist is usable.**
`paper/checklist.md` is one page and is meant to be used by reviewers under time pressure. Ask
them to run it against one real manuscript they are currently reviewing or have recently
reviewed, and report which items were unanswerable from the manuscript, which were ambiguous,
and how long it took. That is a two-hour task and it converts the checklist from an assertion
into a tested instrument. **If they will do only one thing on this list, ask for this one.**

### What happens if we skip it

The anatomy objection (§3.3 of `PAPER_PLAN.md`) lands without an authoritative answer, and the
paper reads as a methods critique written by people who do not read the images. In a
radiology-facing journal that is close to fatal.

### Draft email

> **Subject:** Co-author invitation — a reporting problem in slice-level medical imaging
> benchmarks, and a one-page checklist that needs a clinician's eye
>
> Dear [Name],
>
> I am writing to ask whether you would consider joining a methods paper as a clinical
> co-author. I will be concrete about what the paper claims and what I need from you, because
> the second is a bounded amount of work and I would rather you judge it on that basis than on
> the abstract.
>
> **The finding.** Many 3D medical imaging benchmarks attach labels to individual slices and
> report a slice-level AUROC. We fitted a model that sees no pixels at all — only where a slice
> sits in its stack, taken from the benchmark's own published label CSV — on six public
> benchmarks. On fastMRI Prostate, using the authors' own label file and their own
> patient-disjoint train/test split, that pixel-blind model reaches a slice-level AUROC of
> 0.854 (95% CI 0.812–0.891) against a published headline of 0.861. Read at the patient level,
> the identical scores give 0.506. On two of the six benchmarks the same model fails outright,
> and we report that at the same prominence.
>
> **What the paper does not claim.** It does not claim that any published model learned
> nothing. We could not reproduce the pipeline of the paper closest to this result, and we say
> so in the abstract. The claim is about an evaluation protocol: that a published slice-level
> number is reachable by a model with no access to the images.
>
> **What I would ask of you.** Six things, in about a day of work spread over two or three
> weeks:
>
> 1. A one-paragraph statement, in your own words and under your own name, on whether
>    clinically significant prostate lesions are expected to concentrate at particular
>    positions along the gland — I have drafted a version for you to correct or reject rather
>    than a blank page, and it is attached.
> 2. A judgement on whether slice-level AUROC is ever the clinically relevant metric, and if so
>    for which tasks.
> 3. A read on two dataset-specific questions: how to describe the Duke Breast Cancer MRI
>    slice-level task, in which every patient has cancer; and how much weight to place on
>    fastMRI+ annotations, whose maintainers describe them as indicative rather than
>    adjudicated.
> 4. A wording pass with one specific instruction: find any sentence that slides from "this
>    evaluation protocol certifies a number a pixel-blind model reaches" into "the model
>    learned nothing", and flag it. I have written the list of claims the evidence does not
>    support and will send it with the draft.
> 5. **The one I would most like.** Take the attached one-page checklist and run it against a
>    manuscript you are currently reviewing or have recently reviewed. Tell me which items were
>    unanswerable from the manuscript, which were ambiguous, and how long it took. That
>    converts a page of assertions into a tested instrument, and it is roughly two hours.
> 6. Normal co-author duties: approval of the final text and of the claims made in your name.
>
> I would rather have a co-author who tells me the framing is wrong than one who signs it. If
> after reading the draft you think the clinical argument does not hold, that is a useful
> outcome and I would want to hear it before submission rather than from a reviewer.
>
> Target venues are Radiology: Artificial Intelligence or npj Digital Medicine. The draft, the
> checklist and a document listing every claim the evidence does *not* support are attached.
>
> With thanks,
> [Name]

---

## 2. Biostatistician review

Ideally an **independent reviewer rather than a co-author**, so the acknowledgement can read
"reviewed by X, who is not an author". If they find substantive problems and fix them, offer
authorship then.

### The five questions, in order of how much damage a wrong answer does

**2.1 The MATCHED rule.**
We declare a row MATCHED when the upper bound of the baseline's subject-clustered 95%
percentile interval reaches or exceeds the published point estimate. That is a descriptive
decision rule and we say no *p*-value is claimed from it. Questions: is that defensible as
stated? Would an equivalence framing (two one-sided tests against a pre-specified margin) be
stronger? Would either change any of the twelve verdicts in Table 2? Is there a rule that is
symmetric in a way ours is not?

**2.2 The trivial fraction's interval.**
`(baseline − 0.5) / (published − 0.5)`, with the interval obtained by propagating the
baseline's bootstrap distribution and treating the published number as a constant. We state
in the paper that this is too narrow as a statement about the ratio. Question: when the
publication reports a half-width but does not name the resampling unit — Rempe et al. report
86.1 ± 1.8 and their own text says the interval is an unclustered bootstrap over 1,000
iterations — is there a defensible way to propagate both sources, or is refusing to combine
them the right call? If refusing is right, we would like a sentence we can quote.

**2.3 The position-stratified AUROC.**
`stratified_auc` computes the Mann–Whitney statistic within bins of relative slice position and
pools, so only same-position positive/negative pairs contribute. Questions: is it a consistent
estimator of anything nameable? What is its null distribution? Should it carry a variance
estimate or a bootstrap interval, and if so clustered on what? Is the choice of stratum count
(we report 5 and 10, and the harness reports how many strata were actually populated) a
researcher degree of freedom that needs pre-specifying?

**2.4 Multiplicity.**
Fifteen rows over six benchmarks, eight dataset-arms in the slice-versus-patient table, and a
per-column metadata screen that is explicitly labelled as a maximum over statistics with no
correction. Questions: does the paper need a family-wise or false-discovery correction
anywhere? Our position is that the pre-specified baselines are the headline and the per-column
screen is exploratory and labelled so — is that sufficient, or does the reviewer need to see a
correction?

**2.5 The DeepLesion partition resampling.**
We rebuild Yan et al.'s 25/25/50 patient-disjoint partition 200 times and report mean 0.5571,
sd 0.0131, range [0.5243, 0.5778]. That range is a spread over partitions, not a confidence
interval for a population parameter, and it is currently presented in the same visual slot as
the bootstrap intervals elsewhere. Question: how should it be labelled so it is not read as a
CI, and is there a better summary?

**Two things to hand them alongside the questions**, because they make the review cheap:

- `python pipeline/s04_stats.py --self-test`, block [6] — the bootstrap coverage simulation
  (clustered 91.5% coverage / width 0.370; naive 46.5% / width 0.117; ratio 3.18) on data with
  a closed-form true AUC of 0.6880.
- `trivial-baselines --self-test` — the permutation-null behaviour, including the result that
  an out-of-fold metadata baseline on a subject-level label sits at 0.424 rather than 0.500 on
  a synthetic label that is by construction invisible to metadata. We would like that argument
  checked; it is the basis for judging every metadata baseline against its own null rather
  than against 0.5.

### Draft email

> **Subject:** Statistical review request — five specific questions, ~2 hours, on a
> medical-imaging benchmark audit
>
> Dear [Name],
>
> May I ask you to review the statistics in a methods paper before we submit it? I am asking
> for a review rather than co-authorship in the first instance, because I would like to be able
> to write "reviewed by X, who is not an author" — and because I think the questions are
> bounded enough to be answerable in an afternoon. If it turns into real work, authorship is
> yours.
>
> **The setting.** Benchmarks that label 3D scans slice by slice and report a pooled
> slice-level AUROC. We fit models that see no pixels — position within the stack, acquisition
> metadata — on the benchmarks' own published label tables, and ask how much of the published
> number they reach.
>
> **Five questions.**
>
> 1. We call a comparison MATCHED when the upper bound of a subject-clustered 95% percentile
>    bootstrap interval reaches the published point estimate. Is that defensible as a
>    descriptive rule? Would a TOST-style equivalence framing be better, and would it change
>    any verdict?
> 2. Our headline ratio is (baseline − 0.5)/(published − 0.5). We propagate the baseline's
>    uncertainty and treat the published number as a constant, and we say in the paper that the
>    resulting interval is too narrow as a statement about the ratio. When the source paper
>    reports a half-width but does not state the resampling unit, is there a defensible way to
>    combine the two sources, or is refusing to combine them correct? If refusing is correct, I
>    would like a sentence I can quote.
> 3. We propose a remedy metric: the Mann–Whitney statistic computed within strata of relative
>    slice position, so only same-position pairs contribute. Is it a consistent estimator of
>    something nameable? What is its null distribution? Should it carry an interval, clustered
>    on what?
> 4. Fifteen comparisons across six benchmarks, plus a per-column metadata screen that we label
>    as an uncorrected maximum. Does anything here need a multiplicity correction, or is
>    pre-specifying the headline baselines sufficient?
> 5. One result is a mean and spread over 200 re-drawn data partitions rather than a bootstrap
>    interval. How should it be labelled so no reader takes it for a CI?
>
> Two runnable artefacts that will make this cheaper: a bootstrap-coverage simulation with a
> closed-form true AUC (subject-clustered interval covers 91.5% at nominal 95%; the naive
> slice-level interval covers 46.5% and is 3.18× narrower), and a self-test demonstrating that
> an out-of-fold metadata baseline on a subject-level label sits at 0.424 rather than 0.500 by
> construction. Both run in under a minute with numpy and pandas.
>
> The draft and the code are attached. I am specifically interested in the places where you
> think we have been too generous to ourselves.
>
> With thanks,
> [Name]

---

## 3. The one-paragraph clinical confirmation

This is the single most reusable thing a clinical co-author can give us, and it should be
requested as a **correction to a draft** rather than as a blank page. Send them the paragraph
below and ask them to rewrite, qualify or reject it.

### Draft paragraph, for a radiologist to correct or reject

> *"Clinically significant prostate cancer is not uniformly distributed along the
> superior–inferior axis of the gland. The peripheral zone, where the majority of clinically
> significant tumours arise, is most extensive at the mid-gland and base, and apical and basal
> extremes of an axial acquisition contain proportionally less prostatic tissue and more
> periprostatic structure. Axial T2-weighted and diffusion acquisitions are also typically
> planned to centre coverage on the gland. It is therefore expected, on anatomical and
> protocol grounds together, that slices annotated as containing a PI-RADS ≥ 3 lesion will be
> concentrated away from the first and last slices of an axial prostate stack, and this should
> not be read as an artefact of any particular dataset. The consequence for evaluation is the
> point at issue: a classifier evaluated by pooling slices can be rewarded for reproducing this
> anatomical prior without any patient-specific discrimination, and a slice-level figure of
> merit cannot distinguish the two."*

### The exact instructions to send with it

1. **Correct it rather than approve it.** If any clause is wrong, overstated, or true only for
   certain acquisition protocols, strike it. We would rather ship three sentences that are
   right than six that are defensible.
2. **Tell us whether the zonal claim needs a citation and which one.** We would rather cite the
   standard anatomical source than assert it.
3. **Tell us whether the same argument holds for the other organs in the audit** — breast DCE,
   knee (meniscal tears), lung (nodules), and whole-body CT lesion type. We expect the answer
   to differ by organ, and the differences are informative: DeepLesion's positional model is
   high at both the slice *and* patient level precisely because its labels are anatomical
   regions, and it would strengthen the paper to have a clinician say so.
4. **Tell us if this makes the finding less interesting.** If a radiologist's honest reaction is
   "everybody knows lesions are mid-gland, so of course slice position predicts the label", we
   need to hear it now. Our answer is that this is exactly why the slice-level metric is the
   wrong one — but if the honest reaction is that the whole result is obvious, the paper should
   be reframed around the remedy and the checklist rather than around the audit, and that is a
   decision to make before submission and not after review.

### What happens if we skip it

Reviewer objection §3.3 in `PAPER_PLAN.md` — "your null is anatomy" — is answered by the
authors rather than by a clinician, and it is the objection most likely to come from the
reviewer whose opinion the editor weights most heavily.

---

## 4. Courtesy notice to Rempe et al.

### Why to send it

Six of the twelve scored rows in this paper concern one preprint, arXiv:2407.06165. Offering
its authors sight of the critique before submission is (a) the right thing to do, (b) the
single cheapest way to neutralise a reviewer objection — "the authors were shown this and their
response is in the supplement" is unanswerable — and (c) our best chance of finding out that we
have misread their Table II before a journal does.

**Send it at least six weeks before submission**, and record the date sent and any reply in the
manuscript. If there is no reply, say in the paper that they were offered sight and did not
respond by the stated date. Do not characterise silence as agreement.

### Rules for this email, which matter more than its wording

- **No claim about their model.** We could not reproduce their pipeline. The email must say so
  before it says anything else.
- **Offer the numbers, not just the conclusion.** Attach the label-file audit card and the
  waterfall so they can check the arithmetic.
- **Give a real deadline and honour it.**
- **Do not offer co-authorship.** It would compromise the critique and put them in an awkward
  position. Offer instead to include a response verbatim in the supplement.
- **Send it from a named human**, not from a shared address.

### Draft email

> **Subject:** Courtesy notice before submission — a zero-image baseline on the fastMRI
> Prostate label file used in arXiv:2407.06165
>
> Dear Dr Rempe and colleagues,
>
> I am writing to give you sight of a result about the fastMRI Prostate benchmark before we
> submit it, because your paper is the source of the published numbers we compare against and I
> think you should see it first and have the opportunity to correct us.
>
> **What we did.** We fitted a model that uses no pixels, no k-space and no phase: a 20-bin
> estimate of P(PI-RADS > 2 | relative slice position), fitted on the 218 training patients in
> the `data_split` column of the public `t2_slice_level_labels.csv`, and scored on the 46 test
> patients in the same column. Nothing else enters it.
>
> **What we found.** That model reaches a slice-level AUROC of **0.854 (95% CI 0.812–0.891,
> subject-clustered)** on the T2 label file and **0.851 (0.816–0.887)** on the DWI file,
> against the 86.1% ± 1.8 reported in your Table II. A version that uses no training data at
> all — simply the negative distance from the middle of the stack — reaches 0.841 on the DWI
> file. Read at the patient level, the identical scores give 0.506 and 0.424 respectively.
> Holding relative slice position fixed, the same scores give 0.546 and 0.539.
>
> **What we are not claiming, stated as plainly as I can.** We are **not** claiming that your
> model learned nothing, and the manuscript says so in the abstract, the results and the
> limitations. We implemented your protocol on our own prostate cache and reached 0.616 against
> your reported 0.809 for the magnitude+phase arm, so we do not reproduce your pipeline and we
> have no standing to make any claim about it. Our claim is about the *evaluation protocol*:
> that a slice-level AUROC on this benchmark certifies a number that a pixel-blind model also
> reaches, and that a patient-level number would not have.
>
> Your split is patient-disjoint and we say so; this is not a data-leakage finding in the sense
> of Yagis et al. It is what remains after the split is done correctly.
>
> **What I would like from you.**
>
> 1. Please tell me if we have misread your Table II, your evaluation unit, or your split. I
>    have attached the audit card, which lists every number, the exact command, the SHA-256 of
>    the label file, and the test-set composition (1,399 slices from 46 subjects, 68 positive
>    slices, 20 positive patients).
> 2. If you have a patient-level number for any arm, we would very much like to report it. Its
>    absence is a point we make, and we would rather make a smaller point accurately.
> 3. If you would like to write a response, I will include it verbatim in the supplement,
>    unedited, at whatever length you consider necessary. I am not offering co-authorship,
>    because that would compromise both the critique and your position.
>
> I intend to submit after **[DATE, at least six weeks from sending]**. If I have not heard from
> you by then I will note in the manuscript that you were offered sight of these results on
> [DATE SENT] and record that no response was received, without characterising that silence.
>
> I should add that the reason this benchmark is in the paper at all is that you published the
> label file with a subject identifier, a slice index and a split column, which is what made a
> third-party check possible without the images. Most benchmarks do not, and the paper says
> that too.
>
> With respect and thanks,
> [Name and affiliation]
>
> *Attached: the audit card `fastmri_prostate_t2_published.md`, the DWI card, the waterfall
> figure, and the current draft.*

---

## 5. Two things nobody else can do for us

Recorded here so they are not mistaken for collaborator work. These are our own jobs and they
block submission.

**5.1 Redo the prior-art search properly.** The search recorded in `paper/audit_targets.md`
§3.4 was a handful of web queries, and it missed Yan et al. 2018 — a published position-only
baseline on a benchmark that was already on our own target list. Before submission, search
Google Scholar, the MICCAI / MIDL / ML4H / IPMI proceedings, and the RSNA 2019 ICH Kaggle
solution write-ups and forum threads. The competition community very likely knew that slice
position is predictive; a public forum post predating us would not sink the paper, but
discovering it in review would.

**5.2 Decide the venue and stop.** `paper/PAPER_PLAN.md` §3 recommends a pre-submission enquiry
to npj Digital Medicine naming Ong Ly et al. 2024 explicitly, and going to Radiology:
Artificial Intelligence immediately if the reply is cold. That enquiry is two paragraphs and
one week, and it is worth more than any amount of further polishing.

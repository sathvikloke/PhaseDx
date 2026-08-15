# Approach email — Dr Christopher (Risto) Filippi

**To:** risto.filippi@sickkids.ca
**From:** your own named address, not a shared one
**Attach:** `3_manuscript_anonymized.pdf` only. Not the title page, not the cover letter.

Follows the rules already recorded in `collaborator_candidates.md`: lead with the
slice-to-patient collapse, do not open with a critique of any individual paper, do not
open with the identity of the audited authors, and never write the sentence "this model
learned nothing." The ICH numbers are used rather than the prostate ones because he is
the ICH person.

---

**Subject:** A pixel-blind baseline reaches 0.74 per slice and 0.46 per patient on the RSNA 2019 ICH benchmark — would you consider co-authoring?

Dear Dr Filippi,

On the RSNA 2019 Intracranial Hemorrhage benchmark, a model that reads no pixels — only
each slice's relative position in its stack — reaches AUC 0.738 per slice and 0.455 per
patient from the identical score vector. A constant predictor reads exactly 0.500 at
both units in all 24 holdouts. The gap is not a modelling result; it is a statement
about which unit a benchmark is scored on.

I am writing to you because *Radiology: AI* invited you to write the editorial on trust
and uncertainty in ICH detection AI, which makes you one of the few people who can say
whether the reading I have put on this is the clinically correct one.

**What is already done.** The analysis is finished and the code is public. Every value
sits on a patient-disjoint frozen holdout with a single fit; the flagship is reproduced
by a second implementation sharing no code with the first; and we obtained the official
RSNA label file and reconciled our slice ordering source against it row for row, with
zero disagreements across 4.5 million label comparisons. The same locked protocol,
applied unchanged to a prostate MRI benchmark we had not previously read, reproduced the
divergence but not its mechanism.

**What I am asking.** Co-authorship, on the ICMJE criteria, for three specific
contributions I cannot make:

1. Whether the Discussion is right that the per-patient number is the one aligned with a
   per-patient decision — or whether that understates the clinical value of slice-level
   localisation in hemorrhage.
2. Whether a positional prior on head CT is better described as a confound or simply as
   anatomy, and where the paper should draw that line.
3. Critical revision of the clinical framing generally. It is currently written by people
   who do not read scans.

**What you should know before deciding.** My co-authors and I are secondary-school
students. There is no funding, no institutional sponsor and no conflict of interest on
any side. The manuscript is 24 pages and is attached; the target is *Radiology:
Artificial Intelligence*, and it is not yet submitted, so your name would go on before
submission rather than being added later.

If the answer is no, a single paragraph on question 2 would still be worth a great deal,
and I would acknowledge it rather than list you as an author. If you would rather not
spend time on it at all, please just say so and I will not follow up.

With thanks for your time,

Sathvik Loke
Illinois Mathematics and Science Academy
sloke@imsa.edu

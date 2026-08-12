# Co-author invitation — draft

Send to: a radiologist or clinical imaging researcher.
Attach: `main.pdf` and `figures.pdf` only. Not the title page or cover letter.

---

**Subject:** A model that reads no pixels scores 0.737 on RSNA ICH — asking you to co-author the paper

---

Dear Dr. [Name],

On the RSNA 2019 Intracranial Hemorrhage benchmark — 752,802 slices, 18,938 patients — a
predictor that sees no image data at all, only where a slice sits in the stack, reaches
0.737 AUC at the slice level. Read per patient, the same scores give 0.453.

The paper is attached. I am writing to ask whether you would join it as a co-author.

**What it finds.** Across four public benchmarks, a pixel-blind model reaches a median
0.469 of the published margin over chance — about half — when scored on the same metric
and the same evaluation unit the original papers used. On one benchmark it reaches none of
it, and that null is reported at the same prominence. The sub-chance patient-level reading
turned out to be a stack-depth confound of the mean operator rather than a reversed signal,
and the paper says so and shows the arithmetic.

**Why I am writing to you specifically.** The work is a methods audit, and its weakest
point is clinical. It asserts that a per-patient number is the one corresponding to the
decision a radiologist actually makes, and it reports that none of 30 sampled cleared
510(k) summaries uses a slice-level metric — which cuts against its own motivation. I would
rather have those judgments made by someone who reads studies than defend them from the
outside.

**What co-authorship would mean.** Under ICMJE this is real work, not a name on a masthead:
critically revising the manuscript, approving the final version, and being accountable for
it. I am expecting you to want changes. A reviewer who changes nothing has not reviewed it.

**Where it stands.** Formatted for *Radiology: Artificial Intelligence* as Original
Research — 2,995 words, three figures, four tables, CLAIM checklist complete. It has been
through two rounds of adversarial review that produced a major revision and then a minor
one. All code and outputs are public, and every number in the Results is a deterministic
output of released code run on a published label file.

**One thing you should know before you read it.** All four current authors are
secondary-school students. I am at the Illinois Mathematics and Science Academy. I am
telling you now rather than letting you find out after you have spent time on it. The paper
has to stand on its own, which is why I am sending the paper and not a description of it.

If it is not for you, a one-line no is genuinely useful and I will not follow up. If you
know someone better placed, I would be grateful for the name.

Thank you for your time.

Sathvik Loke
Illinois Mathematics and Science Academy
sloke@imsa.edu · +1 331-707-3013

---

## Notes on sending this

**Who to send it to.** Someone who has published on evaluation methodology, shortcut
learning, or benchmark quality in imaging AI — not simply a famous radiologist. The
citations in the paper are a good starting list. People who have written about this problem
already believe it exists.

**One at a time.** Simultaneous invitations to several people is how you end up with an
authorship dispute or two declines that talk to each other.

**What not to do.** Do not offer authorship in exchange for nothing. Do not send a
description instead of the paper. Do not oversell — the paper's own Discussion is careful
about what it cannot show, and an email that is less careful than the paper reads badly
against it.

**If they say yes**, you need from them: full name with middle initial, degrees,
affiliation with street address, ORCID iD, and a decision on author order. Their ICMJE
disclosure form is submitted separately by them.

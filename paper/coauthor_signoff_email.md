# Co-author sign-off email — send before submitting

This is an **approval request**, not an invitation. Under ICMJE criteria every author
must approve the specific version that gets submitted and be accountable for it, so
this needs an actual reply from each of the three, not silence.

Attach: `3_manuscript_anonymized.pdf`, `2_full_title_page_NOT_anonymized.pdf`,
`1_cover_letter.pdf`. The other files are for the portal, not for review.

---

**To:** Ethan, Neeraj, Aditya
**Subject:** Sign-off needed before I submit to Radiology: AI — the paper changed a lot

Hi all,

I'm ready to submit to *Radiology: Artificial Intelligence*, but I can't do it until
each of you has read the attached version and replied to say you approve it. That's
an ICMJE requirement, not a formality — every author has to approve the exact version
submitted and be accountable for it.

Please don't just skim and say yes. The paper has changed substantially since you
last saw it, and some of the changes are ones you'd want to know about:

**What changed**

- **The title.** It's now *An Audit of Six Public Imaging Datasets*, not seven.
- **Two arms were withdrawn.** The two fastMRI+ knee arms are gone. Their prepared
  label tables were lost and only 155 of the 199 volumes remain locally, so they
  can't be reproduced on their original cohort. Rather than keep them on the old
  estimator, we cut them.
- **Two arms were rebuilt.** Duke Breast and LUNA16 were regenerated from their
  public sources and verified byte-identical against the checksums the original runs
  recorded, then re-scored. No number in the paper now rests on the superseded
  pooled estimator.
- **The headline number moved.** The primary estimate is now the mean over 24
  holdouts (0.7381 per slice, 0.4551 per patient) rather than the single frozen
  holdout, because that one draw sat at the edge of its own family.
- **A new benchmark.** We read PI-CAI's 1,295 lesion delineation volumes under the
  same locked protocol. The slice-versus-patient divergence reproduced on a
  completely different organ and modality — but the stack-depth mechanism did not,
  and the paper says so.
- **The provenance is now verified.** We obtained the official RSNA label file and
  reconciled our mirror against it row for row: zero disagreements across 4.5 million
  label comparisons.

**What I need each of you to check**

- Your name, affiliation and ORCID on the title page are correct.
- The conflict-of-interest statement — "no conflicts, no funding, no industry
  support" — is true for you.
- **Ethan specifically:** the cover letter and Table 3 state that you wrote the second
  implementation, working separately and sharing no code with mine, and that its
  results agree with the primary one to 0.0003 AUC per slice. Please confirm that
  description is accurate before I send it.

**What I need you to do**

1. Read the manuscript and reply saying you approve this version for submission.
2. After I submit, RSNA will email you a copyright form and an ICMJE conflict-of-
   interest form. Both need completing or the submission stalls.
3. If you don't have an ORCID yet, register one at orcid.org — it's free and takes
   two minutes. You'll need it if the paper is accepted.

If anything in there looks wrong to you, say so now rather than after it's in. I'd
much rather fix it than defend it.

Thanks,
Sathvik

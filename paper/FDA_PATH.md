# FDA regulatory path: NOT VIABLE

**Verdict: NOT VIABLE.** Not because 510(k) summaries are too opaque to audit — they are
easy to audit — but because the audit comes back negative. FDA's evidence base for
radiology AI does not carry this flaw. Every volumetric device in a random sample of 30
AI radiology 510(k)s reports performance **per scan, per case, or per subject**. None
reports a slice-level or per-image-within-volume metric. The special controls for
computer-assisted triage and CADe/CADx already mandate the evaluation unit this paper
argues for, so there is no regulatory consequence to demonstrate.

---

## What was actually checked

Frame: openFDA `/device/510k.json`, product codes QIH, QFM, QAS, MYN, POK, QDQ (the AI
radiology codes: triage/notification, CADe, CADx, image analyzers, automated processing).
**n = 535 clearances; 529 (98.9%) publish a Summary.** Random sample of 30 (seed 20260729,
decisions 2020–2026) pulled from `accessdata.fda.gov/cdrh_docs/pdf{YY}/{K}.pdf`. All 30
downloaded; all 30 had a machine-readable text layer; no OCR needed. Sample list:
`/Users/sathvikloke/Downloads/PhaseDx/paper/fda_scoping/sample_frame.json`.

| Coded field | Result (n=30) |
|---|---|
| Reports a **slice-level or per-image-within-volume** metric | **0 / 30** |
| Volumetric (CT/MR) devices reporting per-scan / per-case / per-subject | **all of them** |
| Reports any diagnostic accuracy metric (Sn/Sp/AUROC) | 16 / 30 |
| Reports unique **patient** count for the evaluation set | 9 / 30 |
| Reports the metric at more than one analysis unit | 1 / 30 (MammoScreen K240301) |
| Asserts train/test separation in some form | ~15 / 30 |
| Asserts it at the **patient/subject** level | 3 / 30 |

The decisive sentence is in qER (K200921, head CT triage), which describes its own
architecture as a post-processing module that "combines slice-level outputs to a
scan-level triage result," then reports only scan-level Sn/Sp/AUC over 1,320 head CT
scans. Rapid LVO (K221248): 217 scans. Aidoc BriefCase (K222329): 499 cases. Optellum
(K202300): 300 subjects. Every other occurrence of the word "slice" in all 30 documents
is *slice thickness*, reported as a subgroup covariate. The aggregation step the paper
recommends is not missing from the regulatory record — it is the regulatory requirement.

## The blocker, stated plainly

Three things have to be true for the regulatory paper to exist, and none of them is:

1. **The flaw has to be present.** It isn't. 21 CFR 892.2080 / 892.2070 / 892.2090
   special controls force case-level or lesion-level endpoints, and manufacturers comply.
   Where per-image reporting does survive it is 2D modalities with clustered units —
   Velmeni (K252953) computes CIs over 1,791 teeth from 128 patients; Overjet (K253930)
   reports image-level AUC over 1,888 images with no patient count. That is a clustering /
   multiplicity problem, not the slice-position shortcut, and two dental devices is an
   anecdote.
2. **It has to be demonstrable, not just assertable.** No cleared device's test set is
   public. The trivial baseline can never be run against any of them. The strongest
   achievable claim is "the summary does not let me check" — a non-reporting claim, not a
   finding.
3. **That non-reporting claim has to be new.** It is not. *JAMA Netw Open* 2025,
   "Generalizability of FDA-Approved AI-Enabled Medical Devices," coded the full census of
   **903** devices on exactly these fields (clinical performance study reported: 55.9%;
   explicitly none: 24.1%; discriminatory metrics: 200). Also: Wu et al., *Nat Med* 2021;
   Muehlematter et al., *Lancet Digit Health* 2023 (predicate networks — LDH has already
   published its FDA-AI audit); *Clin Radiol* 2023 (151 imaging devices); *JAMA Netw Open*
   2026 (missing clinical evidence → recall hazard, i.e. the consequence claim, already
   made, with an outcome attached). And *Radiology: AI* 2025, "Distinguishing between Rigor
   and Transparency in FDA Marketing Authorization," is a pre-written rebuttal: absence
   from the public summary is not absence from the submission. An n=30 or even n=500
   redo lands underneath all of it.

No consolation study is offered. The "patient counts are missing from 70% of summaries"
observation is real and is narrower than what has been published, but it is a
Correspondence at best, it does not connect to the slice-position finding, and it is not
worth weeks.

## Straight answer on IF 17+

**No. IF 17+ is not reachable for this project on any path currently visible.** The
regulatory bridge was the only route above npj Digital Medicine, and it is now
empirically closed rather than merely hard — FDA already requires the fix. What remains
is a well-executed methods-and-audit paper whose core construction is prior art
(OsciiArt 2020, Yan CVPR 2018, Badgeley 2019, Ong Ly 2024), with no prospective data, no
patient outcomes, and no intervention. **npj Digital Medicine (~IF 12–15) is the ceiling,
and it is a good ceiling** — *Radiology*, *Radiology: AI*, and *eBioMedicine* are the
realistic alternates. The only thing that would move this into Lancet DH / Nature Medicine
territory is a prospective demonstration that a deployed system's real-world performance
degrades in the way the baseline predicts, which requires clinical data access and a
site partnership, not more analysis. Stop pricing the ceiling and go write the npj paper.

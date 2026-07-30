# Courtesy notice to Rempe et al. — ready to send

**Status: DRAFT, READY TO SEND. NOT SENT. Nothing in this file has been transmitted to anyone.**
A human sends this. Do not automate it, do not send it from a shared address, and do not send it
until the six items in §2 are resolved.

Governing rules: `paper/COLLABORATORS.md` §4. This file implements them; where the two differ,
§4 governs on procedure and this file governs on the numbers, which have been re-read from the
artefacts named in the appendix.

---

## 1. Addressee and where the address came from

| | |
|---|---|
| **To** | `moritz.rempe@uk-essen.de` |
| **Named as** | "Corresponding author: moritz.rempe@uk-essen.de" |
| **Printed on** | Page 1, footnote, of the PDF of arXiv:2407.06165**v2** ("Tumor likelihood estimation on MRI prostate data by utilizing k-Space information"), fetched from `https://arxiv.org/pdf/2407.06165v2` on **2026-07-29** |
| **Corroborated by** | The arXiv abstract page `https://arxiv.org/abs/2407.06165`, submission history line, retrieved 2026-07-29: `From: Moritz Rempe [view email]` — the submitter of both v1 (4 Jun 2024) and v2 (14 Apr 2025) |
| **Affiliation** | Institute for AI in Medicine (IKIM), University Hospital Essen, Girardetstraße 2, 45131 Essen, Germany (affiliation 1 on the title page) |

**No address was guessed, inferred from an institutional pattern, or reconstructed from another
paper.** The arXiv "view email" link (`/show-email/a39903e9/2407.06165`) is behind a bot check
and was **not** used; it was not needed, because the address is printed in the paper itself.

**If you want to copy the other authors** (Hörst, Seibold, Hadaschik, Schlimbach, Egger,
Kröninger, Breuer, Blaimer, Kleesiek — full list on the arXiv title page): **their addresses are
not printed in the preprint and must not be guessed.** Look each one up in the IKIM / University
Hospital Essen / Fraunhofer IIS staff directories and record where you got it, the same way this
row does. Sending to the printed corresponding author alone discharges the obligation; a wrong
guess at a senior author's address does not.

## 2. Fill in before sending

1. `[SEND DATE]` — the date the mail actually leaves.
2. `[REPLY-BY DATE]` — **at least six weeks after `[SEND DATE]`** (`COLLABORATORS.md` §4). Once
   written it is binding: do not submit before it, and do not extend it silently.
3. `[YOUR NAME, AFFILIATION, EMAIL]` — a named human, not a shared address. On the current
   author list that is Sathvik Loke, Illinois Mathematics and Science Academy
   (`trivialbaselines/CITATION.cff`).
4. Attachments — confirm all five are attached and that the draft is the current one.
5. **One consistency blocker.** The email says we record the positional construction as prior art,
   naming Yan et al. (CVPR 2018) and a 2020 Kaggle notebook (OsciiArt, "Baseline with no image",
   RSNA-STR PE). Yan et al. is in `paper/DRAFT.md` §4.5/§5; **OsciiArt currently is not** — it
   lives only in `paper/FINDINGS.md` and `paper/audit_results.md`. Either cite it in the draft
   before attaching the draft, or cut that half-sentence. Do not send a letter that credits prior
   art the attached manuscript omits.
6. **Optional but recommended.** The email points at §3.2, §4.1 and §5 for "this is not a claim
   about their model", because those are the three places `paper/DRAFT.md` currently carries it.
   The **abstract does not**. Putting one clause of it in the abstract would make the letter and
   the paper agree at the level a reader meets first; if that edit is made, update the section
   list in the email to match.

**Record in the manuscript**: the send date, whether a reply arrived, and the reply's substance.
If no reply arrives, state that they were offered sight on `[SEND DATE]` and that no response had
been received by `[REPLY-BY DATE]`. **Do not characterise silence as agreement, concession, or
anything else.**

---

## 3. The email

> **Subject:** Courtesy notice before submission — a zero-image baseline on the fastMRI Prostate
> label files used in arXiv:2407.06165
>
> Dear Dr Rempe, and colleagues,
>
> I am writing to give you sight of a result about the **fastMRI Prostate evaluation protocol**
> before we submit it, and to ask you to correct us if we have got it wrong. Let me put the
> essential distinction in the first paragraph, because everything else depends on it: **our
> finding is about the evaluation protocol, not about your model.** We are not claiming, and the
> manuscript nowhere claims, that your network learned nothing or that your result is wrong. What
> we measured is that a model which never sees a pixel, a phase or a k-space sample — one that
> uses only the slice index in the benchmark's published label file — reaches a slice-level AUROC
> close to the numbers in your Table II under your own protocol, on the same patient-disjoint
> split your paper and your repository use — verified patient for patient against your own
> `dwi_2D_*.csv` files. That is a statement about what a slice-pooled AUROC on this benchmark
> certifies. It is not a statement about your implementation, and I have a specific reason, given
> below, why we have no standing to make one.
>
> **What we did.** We fitted a 20-bin estimate of P(PI-RADS > 2 | relative slice position within
> the volume). It uses one column of the label file and nothing else — no image, no phase, no
> k-space, no acquisition metadata. We fitted it on the **218 training patients** of the
> predefined 70/15/15 fastMRI Prostate split — the split your §III-C states you used — excluded
> the 48 validation patients, and scored it on the **46 test patients**.
>
> Because "the same split" is a claim that is easy to make and easy to get wrong, we checked it
> rather than assumed it. The label file we ran on, `dwi_slice_level_labels.csv` from
> `github.com/cai2r/fastMRI_prostate` (SHA-256 `e22a354132cce884…`), is **content-identical** to
> the copy you ship at `src/datasets/dwi_slice_level_labels.csv` — same 9,490 rows, same seven
> columns, byte-identical apart from a trailing newline. Your own split files
> `src/datasets/dwi_2D_{train,val,test}.csv` resolve to 218 / 48 / 46 patients and
> 6,637 / 1,458 / 1,395 slices, and **their patient sets are identical to the ones we used**. So
> the arm we fitted and scored is yours, patient for patient, and it is patient-disjoint — this
> is **not** a data-leakage finding in the sense of Yagis et al. It is what remains after the
> split has been done correctly. The DWI test arm is 1,395 slices from 46 patients, with 83
> positive slices arising in 27 positive patients. We ran the identical procedure on the
> T2 label file from the same dataset repository (SHA-256 `d248d41c9915c3fe…`; 1,399 test slices,
> 68 positive slices, 20 positive patients), which your repository does not ship — see question 2
> below.
>
> **What we found.**
>
> | | slice-level AUROC |
> |---|---|
> | your Table II gold standard, image + k-space | 0.861 ± 0.018 |
> | your Table II, PCA ×2, magnitude | 0.813 ± 0.022 |
> | your Table II, PCA ×2, magnitude + phase | 0.809 ± 0.021 |
> | **zero-image positional baseline, DWI label file** | **0.851 [0.816, 0.887]** |
> | **zero-image positional baseline, T2 label file** | **0.854 [0.812, 0.891]** |
>
> Your values are transcribed from your Table II, not recomputed. Ours are 95% percentile
> bootstrap intervals clustered on subject; I note that the resampling unit behind your ± is not
> stated, so the two intervals are not strictly comparable and we do not combine them.
>
> Three details matter more to us than the point estimates.
>
> 1. **It does not depend on fitting anything.** Sweeping the bin count gives 0.834 (5 bins),
>    0.842 (10), 0.851 (20), 0.841 (50) on DWI, and 0.835 / 0.848 / 0.854 / 0.856 on T2. A score
>    that uses **no training data at all** — simply `−|relative position − 0.5|`, "how close is
>    this slice to the middle of the stack" — reaches 0.841 on DWI and 0.825 on T2.
> 2. **The same score vector disagrees with itself between reading units.** Read at the patient
>    level, with nothing changed but the unit of evaluation, those same scores give **0.424
>    [0.298, 0.547]** (DWI) and **0.506 [0.381, 0.632]** (T2).
> 3. **The effect is removable, and we say what the remedy is.** Holding relative slice position
>    fixed — a stratified Mann-Whitney over position bins, so that only same-position slice pairs
>    contribute — the zero-image scores fall to 0.539 (DWI) and 0.546 (T2), i.e. to chance, which
>    is the correct behaviour for a model whose only input is position.
>
> **What we are not claiming, stated as plainly as I can.** We attempted to reproduce your
> pipeline on our own prostate cache and **we could not.** Our implementation of your protocol
> reaches **0.574 [0.516, 0.629]** for the magnitude arm against your reported 0.813, and
> **0.616 [0.559, 0.672]** for magnitude + phase against your reported 0.809. Since we do not
> reproduce your pipeline, we have no standing to say anything about your model or your code, and
> the manuscript says so in three places — the results (§3.2, "What this licenses"), the
> discussion (§4.1) and the limitations (§5, first entry). The gap is very likely ours: our
> prostate cohort is far smaller than your 312 subjects, and we ported behaviour from your
> repository (`TIO-IKIM/Tumor-likelihood-estimation-on-MRI-prostate-data-by-utilizing-k-Space-information`
> — your undersampling in `src/datasets/dwi_dataset.py`, among other things) and may simply have
> read it wrong. We report the failure as our failure, and if you can see immediately what we got
> wrong I would be glad to know.
>
> I should also say where this sits in the paper, so that nothing in the manuscript surprises
> you. These rows are presented as a **worked example, explicitly labelled as a comparison
> against a preprint**, and not as the paper's headline. The paper's principal result is on RSNA
> 2019 Intracranial Haemorrhage and involves no published number of anyone's: the same pixel-free
> score vector reads 0.737 at the slice level and 0.453 at the patient level on 18,938 patients.
> Two of the seven benchmarks we audited — LUNA16 and PI-CAI — do **not** show the effect at all,
> and we report that at the same prominence. We also record that the construction is not ours:
> Yan et al. (CVPR 2018, Table 1) published a location-only baseline in DeepLesion's own defining
> paper, and a 2020 Kaggle notebook did something similar for RSNA-STR PE.
>
> **What I would like from you.**
>
> 1. **Please tell me if we have misread anything** — your Table II, your evaluation unit, or
>    your split. I have attached the audit cards, which list every number, the exact command, the
>    SHA-256 of each label file and the full test-set composition, so the arithmetic can be
>    checked line by line.
> 2. **One question we would rather have from you than infer:** which label file Table II's rows
>    correspond to. Your §III-C says "we only work on the DWI data", your repository ships only
>    the DWI label and split files, and the DWI file has 9,490 rows — all of which says DWI. But
>    your abstract and §III-C both give "9508 slices", and 9,508 is the exact row count of
>    `t2_slice_level_labels.csv`. We read that figure as describing the dataset rather than your
>    working set, which would make DWI the arm and leave nothing to resolve; we would like to be
>    told rather than to have reasoned it out. We report both arms so that no conclusion of ours
>    depends on the answer, and the two differ by 0.003 AUROC in any case.
> 3. **If you hold a patient-level number for any arm, we would very much like to report it.**
>    The absence of one is a point we make; we would rather make a smaller point accurately.
> 4. **If you would like to write a response, I will include it verbatim in the supplement**,
>    unedited, at whatever length you consider necessary. I am not offering co-authorship,
>    because that would compromise both the critique and your position.
>
> I intend to submit after **[REPLY-BY DATE]**. If I have not heard from you by then, I will note
> in the manuscript that you were offered sight of these results on **[SEND DATE]** and that no
> response was received by that date — recorded as a fact and nothing more, with no
> interpretation placed on it. If you need longer, say so and I will wait; I would much rather be
> corrected before submission than after.
>
> Finally, and I mean this: this check was possible only because of two decisions neither party
> was obliged to make. The fastMRI Prostate team published a label file carrying a subject
> identifier, a slice index, a label and a split column — most of the benchmarks we looked at do
> not publish enough to be checked this way at all, and the paper says so. And you published your
> code and your three split CSVs, which is the only reason we can state that the split we fitted
> on is genuinely yours rather than one that merely shares its name. The finding is a consequence
> of that transparency, not a penalty for it, and the paper makes the point in its own
> recommendations (§4.4): a subject identifier, a slice index and a split column, published with
> the labels, are what make a benchmark checkable by people who will never be granted the pixels.
>
> With respect and with thanks,
>
> **[YOUR NAME, AFFILIATION, EMAIL]**
>
> *Attached:*
> 1. `fastmri_prostate_t2_published.md` — the T2 audit card (every number, command, SHA-256)
> 2. `fastmri_prostate_dwi_published.md` — the DWI audit card
> 3. The waterfall figure (our reproduction attempt, rung by rung)
> 4. The current manuscript draft
> 5. The audit tool, `trivialbaselines` v1.0 (MIT; numpy and pandas only), so that every number
>    above can be regenerated from the label file in one command

---

## 4. Appendix — every number in the email, and the file it was read from

Provided so the sender can verify the letter before it goes, and so that a reply disputing a
number can be checked against the same artefact.

| Claim in the email | Value | Source on disk |
|---|---|---|
| T2 label file SHA-256 | `d248d41c9915c3fe10d7d6ecbf38d30c3c28eef0cdc3a216a4e74978fe84e414` | `pipeline_out/trivial_baselines/fastmri_prostate_t2_published.json` (`labels_sha256`, 16-char prefix); full digest recomputed from the cached CSV |
| DWI label file SHA-256 | `e22a354132cce884a3e7c8e762cf039f5269a4eb49dba650453c3537a3c20ecf` | same, DWI payload |
| Source repository, licence | `github.com/cai2r/fastMRI_prostate`, files `fastmri_prostate/data/{t2,dwi}_slice_level_labels.csv`; MIT (repo), no DUA for the CSVs | `paper/audit_results.md` §3.1 and Table 1 (line 969); repository tree listed via the GitHub API, 2026-07-29 |
| **The label file is the *dataset's*, not Rempe et al.'s** — but their vendored copy is content-identical | `src/datasets/dwi_slice_level_labels.csv` in `TIO-IKIM/Tumor-likelihood-estimation-on-MRI-prostate-data-by-utilizing-k-Space-information`: same 9,490 rows, same 7 columns, `DataFrame.equals` True, bytes equal after stripping one trailing newline (796,851 vs 796,852 B) | downloaded from `raw.githubusercontent.com` and diffed against the cached cai2r copy, 2026-07-29. **The T2 file is not in their repository.** |
| **The split is theirs, verified patient for patient** | their `src/datasets/dwi_2D_{train,val,test}.csv` → 218 / 48 / 46 patients, 6,637 / 1,458 / 1,395 slices; patient sets **identical** to the `data_split` column we used | same download; set comparison on `fastmri_pt_id`. Independently, `pipeline/s12_rempe.py` §1.2 records train/val/test split agreement 1.0000 against our own cache |
| Split: 218 train / 48 validation / 46 test **patients** | — | recomputed from the cached label CSVs, `groupby('data_split')['fastmri_pt_id'].nunique()`; slice counts 6,647 / 1,462 / 1,399 (T2) and 6,637 / 1,458 / 1,395 (DWI) match `split_column_report.values_seen` in both payloads |
| Their §III-C: "we only work on the DWI data"; "the authors of the FastMRI Prostate dataset give a predefined 70% - 15% - 15% datasplit … which was also used in this work" | verbatim | text extracted from the arXiv:2407.06165v2 PDF, 2026-07-29; also recorded at `pipeline/s12_rempe.py:33-34, 57-60` |
| Their public code | `github.com/TIO-IKIM/tumor-prediction-on-undersampled-MRI-kSpace`, which 301-redirects to `…/Tumor-likelihood-estimation-on-MRI-prostate-data-by-utilizing-k-Space-information` | page 2 of their PDF ("…licly available at https://github.com/TIO-IKIM/"); redirect resolved 2026-07-29. **Use the redirect target in the email if you name a path.** |
| T2 test composition | 1,399 slices, 46 subjects, 68 positive slices, 20 positive patients | T2 card, "Test set" line |
| DWI test composition | 1,395 slices, 46 subjects, 83 positive slices, 27 positive patients | DWI card, "Test set" line |
| Zero-image positional, T2 | slice 0.854 [0.812, 0.891]; patient 0.506 [0.381, 0.632] | T2 card, `positional_20bin` row |
| Zero-image positional, DWI | slice 0.851 [0.816, 0.887]; patient 0.424 [0.298, 0.547] | DWI card, `positional_20bin` row |
| Bin sweep | DWI 0.834 / 0.842 / 0.851 / 0.841 and T2 0.835 / 0.848 / 0.854 / 0.856 at 5 / 10 / 20 / 50 bins | both cards, "Bin sensitivity" |
| No-fit centrality | 0.825 (T2), 0.841 (DWI) | both cards, "no-fit centrality" column |
| Position-stratified (remedy) | 0.546 (T2, 5 strata), 0.539 (DWI, 6 strata) | `pipeline_out/rempe/positional_baseline{,_dwi_labels}.json`, `slice_auc_position_stratified`; tabulated in `paper/DRAFT.md` §3.8 |
| Their published values 0.861 ± 1.8, 0.813 ± 2.2, 0.809 ± 2.1 | transcribed, not recomputed | `pipeline/s12_rempe.py:379-390` (`REPORTED`), which also records their level as "slice-level AUROC, unclustered bootstrap (1000 iterations)" |
| Our reproduction: 0.574 [0.516, 0.629] magnitude | vs their 0.813 | `pipeline_out/s12_arm_mag.log`, rung W2 |
| Our reproduction: 0.616 [0.559, 0.672] magnitude + phase | vs their 0.809 | `pipeline_out/s12_waterfall_magphase.log`, rung W2 |
| RSNA ICH, 18,938 patients: 0.737 slice → 0.453 patient | slice [0.735, 0.740]; patient [0.445, 0.461] | `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`, label `any` |
| LUNA16 and PI-CAI do not fire | LUNA16 0.534 slice / 0.581 patient; PI-CAI positional exactly 0.500 | `pipeline_out/trivial_baselines/luna16_fp_reduction_candidates.md`, `picai_case_level.md`; `paper/FINDINGS.md` §3.3 |
| Which arm Table II refers to | 9,508 rows (T2) vs 9,490 (DWI); their abstract's "9508 slices" | both payloads, `n_rows`; `paper/audit_results.md` §3.1. **⚠ See the note below — the letter does not repeat that section's conclusion.** |
| Prior art acknowledged | Yan et al., CVPR 2018, Table 1 (59.7% location-only vs their 90.5%) | `paper/DRAFT.md` §4.5, §5 |
| No journal reference for arXiv:2407.06165 | none, as of 2026-07-29 | arXiv abstract page, re-queried 2026-07-29; `paper/audit_results.md` §2.3 |

### ⚠ One internal record this letter does not follow, and why

`paper/audit_results.md` §3.1 and the "Note for revision" in `paper/DRAFT.md` §3.2 conclude that
**T2 is the correct arm**, on the sole ground that their abstract's "9508 slices" is the exact row
count of the T2 file. Checking the preprint's full text while drafting this letter weakened that
inference in two ways, both verified above:

1. Their §III-C says, verbatim, *"we only work on the DWI data"*; and the "9508 slices" figure
   appears in §III-C in a sentence describing **the dataset** ("this dataset contains 312 male
   patients with a total of 9508 slices"), not their working set. The abstract most likely carries
   the dataset-level figure.
2. Their repository ships **only** the DWI label file and the DWI split CSVs. There is no T2 arm
   in their released code.

The letter therefore asks the question instead of asserting an answer, and states the evidence in
the direction it actually points. **This does not change any number in the paper** — both arms are
reported and they differ by 0.003 AUROC — but `audit_results.md` §3.1's recommendation to reverse
`audit_targets.json` should be revisited before submission, since the original `audit_targets.json`
recommendation (DWI) now looks correct. Flagged here rather than edited into those files, because
they are another task's artefacts.

**Numbers deliberately kept out of the email.** The trivial fractions (0.981 [0.865, 1.084] on
T2, 0.973 [0.876, 1.073] on DWI against the 0.861 headline; above 1 against the PCA and R = 16
arms) are in the manuscript and in the attached cards, and they are the correct way to state the
result in a paper. In a letter they read as a scoreboard, and the ratio's interval propagates
uncertainty in our baseline only, which makes it too narrow as a statement about the ratio. The
email gives the two AUROCs side by side instead and lets the recipient form the ratio. Nothing is
hidden: both cards print the fraction on their "Headline" line.

**Consistency check.** Every claim in the email appears in `paper/DRAFT.md` in the same direction
and with the same value (§3.2, §3.3, §3.8, §4.5, §5). If a reader of this letter went on to read
the manuscript, there is nothing there that would come as a surprise — that was the drafting
constraint.

# ⚠️ DO NOT SUBMIT ANYTHING IN THIS FOLDER

Every `.docx` in this directory reports numbers that **no code in this repository has ever
produced**, and several of them are not reachable from the data on the drive at all.

## What is wrong with it

| Claim in the draft | What the data actually shows |
|---|---|
| Phase-only AUC 0.85 (prostate), 0.97 (breast) | Pooled out-of-fold, patient-level: prostate_t2 **0.380**, breast **0.631**; every CI covers or sits below 0.5 |
| 121 prostate DWI examinations | **50** DWI files exist on the drive; 121 is DWI + T2 combined, which are different sequences |
| Bootstrap CIs, DeLong tests, 2 seeds, patient-level splits | None of this existed in the repo when the draft was written — no seed loop, no CI code, no DeLong anywhere |
| Figure 5 example images | Generated with **transposed prostate axes** — it shows slice 0 of diffusion volume 30, not a mid-gland slice — and treats radial breast k-space as Cartesian |

The breast cohort counts (140 acquisitions, 108 malignant) *are* consistent with the
`breast_updated` subset. Almost nothing else is.

## Why the numbers were never reproducible

The original `utils/data_utils.py` assumed 4-D k-space. Nothing on the drive is 4-D except
brain and knee. Prostate DWI is additionally 2x undersampled with the zeros stored, so a
plain inverse FFT — which is what that code did — produces an image with two copies of the
pelvis folded on top of each other. Every prostate result predating 2026-07-27 was trained
on aliased anatomy.

## What to use instead

- `pipeline_out/report/RESULTS.md` — the generated report, every number traced to JSON
- `paper/DRAFT.md` — the current manuscript draft
- `paper/PAPER_PLAN.md` — framing, target venue, limitations

Kept only as a record of what was originally attempted.

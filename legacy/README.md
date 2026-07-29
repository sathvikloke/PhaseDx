# Legacy code — superseded, and wrong in ways worth recording

Nothing here is used by the current pipeline. It is kept because it explains why the
project's early results were wrong, and because deleting the evidence of a mistake makes the
mistake easier to repeat.

**Do not run any of it.**

## What each file got wrong

### `utils/data_utils.py`

Three defects, any one of which invalidates a result:

1. **It assumes 4-D k-space** — `(slices, coils, H, W)`. Nothing in the fastMRI prostate or
   breast releases is 4-D. Prostate DWI is `(50 diffusion, 30 slices, 20 coils, 200, 150)`;
   prostate T2 is `(3 averages, 30, 20, 640, 451)`; breast is
   `(real/imag, 288 spokes, 640, 16 coils, partitions)`. Every index was wrong.
2. **It ignores that prostate DWI is 2× undersampled with the zeros stored.** 75 of the 150
   phase-encode lines are exactly zero and the ISMRMRD header declares
   `accelerationFactor = 2` with the autocalibration lines shipped separately as
   `calibration_data`. A plain inverse FFT of that array produces an image with **two copies
   of the pelvis folded on top of each other**. Every prostate result the project produced
   before 2026-07-27 was trained on aliased anatomy. GRAPPA is required first.
3. **It min–max normalises phase.** Phase is circular on [−π, π]; rescaling by per-slice
   extrema destroys the units and makes a pixel's value depend on the most extreme pixel
   elsewhere in the slice. The replacement keeps radians and feeds sin/cos, which is also
   continuous across the wrap.

It also applied `ifftshift` on both sides of the inverse FFT, which is only equivalent to a
centred transform for even-length axes.

### `make_real_figure5.py`

Reads the prostate diffusion axes **transposed**: it treats axis 0 as slices when axis 0 is
the 50 diffusion volumes, so `k[sidx, 0]` selects diffusion volume ~30, slice 0 — an edge
slice that is usually near-empty — rather than a mid-gland slice. It also treats the breast
radial stack-of-stars acquisition as Cartesian and zero-fills it, which is not a
reconstruction. Figure 5 of the superseded manuscript came from this script.

### `train.py`, `evaluate.py`, `run_experiment.py`, `models/models.py`

Structurally fine but built on the reader above, and they split at file level rather than
subject level, evaluate at slice level only, and have no confidence intervals, no
significance test, and no multi-seed support.

## What replaced it

`pipeline/` — see the top-level `README.md`. Every reconstruction in the new pipeline is
validated against the vendor reference shipped inside the same HDF5 file (`reconstruction_rss`,
`trace_b50`, or `temptv`), which is the check that would have caught all of the above.

# Hostile-reviewer audit of `paper/tex/`

Date: 2026-07-30. Auditor: automated pass over `main.tex`, `supplement.tex`,
`cover_letter.tex`, `refs.bib`, `figures/`, checked against the JSON/CSV/markdown
artefacts in `pipeline_out/` and `paper/`.

Verdict on the two standing rules: **no null was rounded up** anywhere in either
document, and I found **no sentence claiming a model learned nothing**. The
prior-art position is stated in the introduction in bold, not buried. The three
defects that would have done real damage were (1) a broken bibliography, (2) a
figure caption that overstated the intervals, and (3) two sentences that turned a
censored screen into a universal absence. All three are fixed.

---

## 0. Compile status

A TeX toolchain **is** present (`tectonic` 0.17.0, no `pdflatex`/`bibtex` binary,
but tectonic runs BibTeX internally). All three documents were built in a clean
sandbox from the files as they now stand.

| document | undefined cite/ref | overfull boxes | BibTeX errors |
|---|---|---|---|
| `main.tex` | **0** | **0** | 0 |
| `supplement.tex` | 0 | 12 (worst 37 pt) | n/a (no bib) |
| `cover_letter.tex` | 0 | 0 | n/a |

`main.bbl` resolves 38 references. Before the fixes, `main.tex` produced **48
"Citation undefined" warnings and emitted no reference list at all**.

Overleaf-breaker checklist: no missing figure files (all five
`figures/fig[1-5]*.pdf` present and referenced), no unbalanced braces, no
unescaped specials, no exotic packages (`geometry amsmath graphicx array booktabs
siunitx caption natbib hyperref` in main; `longtable ragged2e microtype amssymb
lmodern` added in the supplement — all stock TeX Live). `\resizebox` is used in
five tables and `graphicx` is loaded, so that is fine.

The 12 remaining overfull boxes in the supplement are all unbreakable `\texttt{}`
identifiers inside narrow table columns (e.g.
`positional_distribution_reported`, `screener_confidence = low`). Worst is 37 pt
(≈0.5 in). Cosmetic; not fixed because fixing them means renaming released field
names in prose.

### `\todo` markers deliberately left in (8)

These are genuine missing inputs, not stubs, and `\todo` is a locally defined
`\newcommand` that compiles. They must be filled before submission:
affiliations for Ethan Johnson and Aditya Raut; funding statement; COI
disclosures; IRB/ethics statement; acknowledgements; repository URL and archival
DOI; final word count (commented out). `refs.bib` also flags that
`trivialbaselines2026` has **no Zenodo DOI minted yet**.

---

## 1. Discrepancies found and fixed

### 1.1 CITATIONS — the manuscript had no working bibliography (fixed)

**C-1 (critical). Wrong bib filename.** The end of `main.tex` read
`\IfFileExists{references.bib}{...\bibliography{references}}{placeholder}`. The
bibliography in the folder is `refs.bib`. On Overleaf the `\IfFileExists` branch
would have been false, the reference list replaced by an apologetic placeholder
paragraph, and **every `\citep` rendered as `[?]`**. Replaced with a plain
`\bibliographystyle{unsrtnat}` / `\bibliography{refs}`. The silent-fallback
construction is also actively harmful — it converts a hard failure into a
plausible-looking PDF — so it is gone rather than repointed.

**C-2 (critical). 32 of 35 citation keys did not exist in `refs.bib`.** The keys
in `main.tex` were short forms; the bib entries use suffixed forms. Only
`holm1979`, `wilson1927` and `yan2018deeplesion` resolved. Remapped throughout:
`badgeley2019→badgeley2019hip`, `burduja2020→burduja2020ich`,
`colak2021→colak2021rsnape`, `degrave2021→degrave2021covid`,
`ehteshamibejnordi2017→bejnordi2017camelyon16`,
`flanders2020→flanders2020rsnaich`, `fleiss1971→fleiss1971kappa`,
`geirhos2020→geirhos2020shortcut`, `guo2019→guo2019wsi`, `islam2021→islam2021pe`,
`jarkman2022→jarkman2022generalization`, `kapoor2023→kapoor2023leakage`,
`maierhein2018→maierhein2018rankings`, `ongly2024→ongly2024shortcut`,
`osciiart2020→osciiart2020baseline`, `page2021→page2021prisma`,
`rempe2024→rempe2024kspace`, `roberts2021→roberts2021pitfalls`,
`ruan2021→ruan2021wsi`, `saha2024→saha2024picai`, `setio2017→setio2017luna16`,
`tampu2022→tampu2022leakage`, `varoquaux2022→varoquaux2022failures`,
`wen2020→wen2020cnn`, `yagis2021→yagis2021leakage`, `zech2018→zech2018pneumonia`,
`zhao2022→zhao2022fastmriplus`. The header comment block listing the keys was
rewritten to match.

**C-3. Three cited works had no bib entry under any key.** `chen2025`,
`gwet2008`, `zbontar2018`. Actions:
* `chen2025` → new entry `chen2025camcsa`, transcribed from
  `paper/published_inversions_round2.json` (`chen_camcsa_11methods`): Chen Y, Wu
  H, Huang Z, Zhang Z, *Sci Rep* 2025;15, PMCID PMC12657889 — the eleven-method
  CAMELYON16 negative control at τ_a = 0.927. The entry's `note` says the volume
  and DOI were **not** re-verified against a live publisher record in this pass;
  they must be before submission.
* `gwet2008` → new entry `gwet2008ac1`: Gwet KL, *Br J Math Stat Psychol*
  2008;61(1):29–48, doi 10.1348/000711006X126600.
* `zbontar2018` → see C-4.

**C-4. Wrong source cited for the fastMRI Prostate benchmark.** `main.tex` cited
`zbontar2018` for "fastMRI Prostate (T2 and DWI)". Žbontar et al. 2018 is the
fastMRI **knee/brain** dataset paper; it does not contain the prostate release.
Repointed to `tibrewala2023fastmriprostate` (arXiv:2304.09254), which was already
in `refs.bib` and unused.

**C-5. Under-cited dataset.** `saha2018` was cited for "Duke Breast Cancer MRI".
`refs.bib` distinguishes `saha2018radiogenomics` (the *Br J Cancer* 922-subject
study) from `saha2021dukebreastmri` (the TCIA collection, doi
10.7937/TCIA.e3sv-re93). The benchmark being audited is the TCIA release, so both
are now cited, dataset first.

**C-6. Wrong-source citation for a leaderboard measurement.** The claim
"CAMELYON16's organisers scored the same 32 submissions at two units and
published both boards (τ = 0.754, 61 of 496 pairs discordant)" was cited to
`ehteshamibejnordi2017` (the *JAMA* article). Per `paper/rank_inversion.md` §5.1
and the `camelyon16leaderboard` note in `refs.bib`, τ = 0.754 and 61/496 are **our
measurement on the two Grand Challenge leaderboard pages**, not numbers the JAMA
paper prints. Now cited to `camelyon16leaderboard,bejnordi2017camelyon16` and
reworded to "on which we measure".

**C-7. Prior-art attribution conflated two Yan papers.** `yan2018cvpr` (the CVPR
paper carrying the "Baseline: Location feature" row at 59.7% vs 90.5%) is
`yan2018deeplesiongraphs` in the bib — "Deep Lesion Graphs in the Wild", CVPR
2018. `yan2018deeplesion` is the *J Med Imaging* dataset release. The Limitations
sentence "DeepLesion's own defining paper published a location-only baseline in
2018" points the CVPR key at a description that belongs to the dataset paper.
The key is now correct; **the wording is still loose and is listed as unresolved
(U-6)**.

**C-8. Software not cited.** Added `\citep{trivialbaselines2026}` at the
`trivialbaselines` v1.0 mention in Methods.

Bib entries now referenced: 36 keys, all resolving. 15 verified entries remain
unused (`chilamkurthy2018headct`, `delong1988`, `hill2024shortcutting`,
`kaggle2019rsnaich`, `kaggle2020rsnastrpe`, `lin2023rsnacspine`, `lin2024shortcut`,
`muehlematter2021approval`, `muehlematter2023predicate`, `ngo2022adjacentslices`,
`oakdenrayner2020hidden`, `rudie2024ratic`, `bricaud2026adni`,
`guneri2026pancreatitis`, `wu2021fdaeval`). Not an error — but `wu2021fdaeval`
and the two `muehlematter` entries support the supplement's uncited "903-device
census" sentence (see U-4).

### 1.2 NUMBERS

**N-1. Wilson upper bound wrong by one digit (2 places).** "Forty of 91 papers
(44.0%; 95% CI: 34.2%, **54.3%**)" in Results §3.3 and the same in
Table `tab:screen`. `paper/FINDINGS.md` line 354 gives `[34.2%, 54.2%]`, and I
recomputed Wilson for k = 40, n = 91: `[0.34209, 0.54191]`. Corrected to **54.2%**
in both places.

**N-2. Two different values for the same block's unreachable rate, in adjacent
sentences.** Results §3.3 gave the four block rates as "35.6%, 34.6%, 22.7% and
32.1%" (correct — `pooled_final.json/flow_by_block/*/S6_unreachable`), then said
"adding 150 papers to the original 100 moved the pooled figure from **36.4%** to
32.6%". 36.4% is the **superseded** v1.0/v1.2 coding of those same 100 papers
(`paper/prisma_flow.json`, whose own header declares itself SUPERSEDED and says
"Do not cite any number from this file"). The authoritative figure for the main
block is 21/59 = 35.6%. Corrected to **35.6%**, which also makes the two
sentences agree.

**N-3. Sign contradiction between text and table on densenet121.** Results §3.4
said "for densenet121 the slice-level difference is **+0.004**"; Table `tab:rank`
Panel B says `d_slice = −0.004`. Both are correct under opposite conventions —
`rankinversion.json` records the pairwise test as phase-minus-magnitude
(`slice_diff = +0.004447`, CI `[−0.0409, +0.0459]`, p = 0.816) while Panel B's
`d` is magnitude-minus-phase. The text stated neither. Rewritten to name the
direction and to say explicitly that it is the same number as Panel B's −0.004.
Same fix in Panel B's footnote.

**N-4. Misleading emphasis in Table `tab:rank` Panel B.** Holm *p* of 0.012
(densenet121) and 0.006 (resnet18) were bold, but the identical 0.006 (resnet50)
and 0.024 (vit_b_16) were not — implying two of the four Holm-supported
architectures were not supported. `rankinversion.json` marks all four
`supported: true` and the text says "individually Holm-supported for four". All
four Holm *p* values are now bold, and the footnote says what the bold means and
that `d` is magnitude − phase throughout the panel.

**N-5. Two cohort sizes in the paper for the same cohorts, never reconciled.**
Methods gave "brain n = 454 … knee n = 96"; Table `tab:rank` gives 136 brain and
29 knee subjects; the coil-prediction result is "on 136 independent test
subjects". Both sets are real (454/199 volumes in `recon_fidelity/*.json`;
136/29 subjects in `rankinversion.json`) but a reviewer reads 454 as the cohort
the rank analysis ran on. Methods now says "n = 454 available volumes" and adds
one sentence stating that the rank-inversion arm runs on the 136 brain and 29
knee subjects for which every method's per-slice predictions exist, and that
Table 6 reports those.

**N-6. "The six RSNA ICH subtype rows" (2 places).** One of the six official
labels is *any haemorrhage*, which is not a subtype. Changed to "the six RSNA ICH
label rows" in Results §3.2 and in Table `tab:fraction`'s footnote.

**N-7. Two full-text scan counts that read as a contradiction.** Methods: "2,934
open-access full texts"; Results: "of 2,642 full texts scanned for a fine-unit
and a coarse-unit column". `paper/rank_inversion.md` lines 324 and 391 confirm
these are different stages (total Europe PMC harvest vs. the same-metric column
scanner). Methods now states the relationship in one clause.

### 1.3 OVERCLAIMING

**O-1. Figure 1 caption asserted more than the intervals support.** It said
"every patient-level interval sits at or below chance". Two of the six do not:
epidural `[0.4613, 0.5244]` and intraventricular `[0.4870, 0.5077]` cross 0.5
(`rsna_ich_unit_collapse.json`). Rewritten: "every patient-level point estimate
lies below chance, and four of the six intervals lie wholly below it (the
epidural and intraventricular intervals cross 0.5)."

**O-2. Abstract Conclusion turned a censored screen into a universal absence.**
"in a pre-registered sample of the literature this check was **never reported**"
— 44 of 135 eligible papers could not be read, and the protocol makes the bound
the headline. Changed to "no paper that could be read reported this check". The
abstract's Results paragraph already carried the bound and was left alone.

**O-3. Same defect in the Summary Statement.** "yet no paper in a pre-registered
random sample of this literature reported such a check" → "yet no paper that
could be obtained … in a sample where a third of eligible papers could not be
obtained".

**O-4. The "censoring-free" statement was stronger than the logic allows.** The
paragraph claimed the 0-of-345-records result was "stronger" than the
complete-case estimate and that "the censoring does not touch it". An unreachable
record is coded `not_assessable` on every sub-flag and therefore **cannot** carry
a positive code, so the 44 unreachable papers contribute scope but no
information. Rewritten to keep the finding (it genuinely is broader than 0/91,
because it also covers papers read and then excluded) while stating that it is
**not** censoring-proof and that only the bounding interval addresses the
unreachable records. This is the one change that makes a result weaker; it is
also the one a reviewer was certain to make for us.

**O-5. "Inversions are real" overstated the published-inversion paragraph.** The
bold lead-in said "In the published literature, inversions are real and change
reported winners", which asserts the unit as cause — exactly what Maier-Hein et
al. cap and what the same paragraph then walks back. Rewritten to "reported
orderings do differ between units, and in no located case has the difference been
shown to exceed sampling noise — including ours", which is what §5 of
`paper/rank_inversion.md` and the section header in `refs.bib` actually say.

**O-6. A "zero-image" result resting on a window/level column.** Results §3.2
opened "One further **zero-image** result …" for the DeepLesion lung row, whose
predictor is the DICOM window/level header — a field that records a choice made
about the anatomy being viewed, and therefore in tension with the Methods'
column-discipline rule excluding image-derived columns. Reframed: "One further
result", plus a clause giving the actual values (−1500,500 vs −175,275),
identifying it as a reconstruction-protocol field, and labelling the row an
acquisition-metadata rather than a positional result.

### 1.4 COSMETIC (supplement)

Four unbreakable-line overflows of 92–144 pt were reduced by adding
`\allowbreak` in two file paths, shrinking the SHA-256 verbatim block to
`\scriptsize`, and un-mathing two κ set lists. Overfull count 18 → 12, worst
144 pt → 37 pt. No content changed.

---

## 2. Verified correct — no change made

Everything below was checked cell-by-cell against the named artefact and matches.

**RSNA ICH flagship** (`pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`):
all 30 cells of Table `tab:ich` (prevalences, both AUROCs, all four interval
bounds per row, all six gaps 0.205–0.307); max-aggregated range 0.486–0.505;
permutation nulls 0.502 slice / 0.523 patient; the "0.070 below" arithmetic;
constant predictor 0.492/0.501; 752,802 / 21,744 / 18,938. The naive-interval
ratio "1.5–2.0× too narrow" — I recomputed all six width ratios: 1.506, 1.679,
1.760, 1.907, 1.984, 1.990.

**Four routes** (Table `tab:routes`): released tool 0.7376 [0.7352, 0.7399] /
0.4561 [0.4478, 0.4640] from `rsna_ich_any_slice_full.json`; independent
implementation 0.7374/0.4533; agreement to 0.003 at both units (slice 0.0002,
patient 0.0028).

**Bin sweeps and robustness** (`rsna_ich_any_slice_full.json`): slice 0.716 /
0.733 / 0.738 / 0.745; patient 0.437 / 0.445 / 0.456 / 0.632; no-fit centrality
0.735; apparent-train 0.7379 vs held-out 0.7376; volume-size patient 0.591;
combined position+metadata 0.718 against its own null of 0.718.

**Trivial fraction** (`paper/trivial_fraction_distribution.md`): all five rows of
Table `tab:fraction` (n, min, Q1, median, Q3, max) reproduce exactly; the nine
peer-reviewed strongest-per-arm values re-sort to median 0.469, IQR
[0.437, 0.490]; "eight of nine between 0.395 and 0.613"; the six ICH values
against the stronger and weaker Burduja systems (0.395–0.613 and 0.410–0.615);
verdict counts 6/17/1 and **0 MATCHED once the preprint is removed**; 24 rows,
seven benchmarks, eight label files.

**Two non-firing arms**: LUNA16 CPM 0.0020, sensitivity 0.0006 at 1 FP/scan
against a random reference of 0.0027, fraction −0.002; PI-CAI positional exactly
0.500 at every bin setting on the official split, metadata 0.692 [0.626, 0.755]
against 0.91 and 0.86.

**Table `tab:units`**: all nine rows verified against their individual
`trivial_baselines/*.json` payloads, including the LUNA16 slice CI (0.513, 0.558
— note `paper/FINDINGS.md` prints 0.514; the JSON says 0.51347, so the manuscript
is right) and the Duke breast "undefined (922/922 positive)".

**Rempe rows**: T2 0.854 [0.812, 0.891] and DWI 0.851 [0.816, 0.887] against a
published 0.861, fractions 0.981 [0.865, 1.084] and 0.973 [0.876, 1.073]; T2 bin
sweep 0.835/0.848/0.854/0.854/0.856 over 5/10/20/30/50; no-fit 0.825 (T2), 0.841
(DWI); reimplementation 0.616 vs 0.809 and 0.574 vs 0.813; stratified
0.854→0.546 (5 strata), 0.851→0.539 (6 strata), 0.574→0.467, 0.616→0.562.

**Prevalence screen** (`paper/screen/analysis/pooled_final.json` — the
authoritative file; `paper/prisma_flow.json` is self-declared SUPERSEDED and was
correctly *not* used): every number in Table `tab:screen` and Figure 3's caption.
250 / 79 / 171 / 127 / 36 / 135 / 44 / 91 / 115; E-SEG 39, E-DERIV 32, remainder
44; S6 32.6% [25.3, 40.9]; P1 0/91 [0, 4.1] and 0/79 [0, 4.6]; bound
[0.0%, 32.6%] with outer envelope 40.9%; S1 5/91, S2 17/91, S3 6/19, S4 29/91,
S5 1/91, S8 2/91; 85 report one unit; 48 no uncertainty; split-unit distribution
29/22/17; S7 all cells P1 = False; post-hoc R4 300/114/56, 32.9% [26.3, 40.3],
0/114 [0, 3.3]; block rates 35.6/34.6/22.7/32.1; censoring-free 345 records over
300 papers, 0 positive; subgroups 36.0% [26.7, 46.6] vs 11.5% [4.0, 29.0] and
37.6% [28.5, 47.8] vs 21.4% [11.7, 35.9]; 23 unclassified. Wilson upper for 0/75
= 4.87% → the "4.9%" justifying the n = 75 target is right.

**Agreement** (`paper/screen/analysis/adjudication_out.json`): pre-reconciliation
raw 65.6% [50.0, 80.0], Fleiss −0.015 [−0.164, 0.120], AC1 0.479 [0.119, 0.740],
6/15 unanimous; amended encoding raw 95.6% [86.7, 100.0], Fleiss 0.932
[0.777, 1.000], AC1 0.934 [0.800, 1.000]; split_unit 0.637 [0.430, 0.824];
core-six restriction n = 6, 100%; naive-truthy κ −0.176. The "same four sealed
files re-encoded, not an independent re-rating" caveat appears in the abstract's
sibling paragraph, the Results, the Limitations, Table 5's footnote and
supplement §S2 — five times, correctly, every time.

**Rank inversion** (`pipeline_out/rankinversion.json`): 447 pairs
(78+3+210+153+3), 204 sign-discordant candidates, 0 survivors; every
`d_between`, `δ vs slice`, `δ vs patient`, split-half τ and verdict in Table
`tab:rank`; prostate T2 ρ = −0.311 [−0.356, 0.635], 121/210 flipping, 59% of
pairs, split-half τ_agg −0.42 [−0.644, −0.121], all 21 patient AUROCs in
[0.381, 0.539]; brain top-3 overlap 1 of 3; all six interaction rows, the mean
I = +0.028 [0.015, 0.041] p = 0.001, 6/6 positive, 4 Holm-supported, 2 sign
flips; aggregation gains +0.045 vs +0.017 and, at matched slice AUROC
0.912–0.924, +0.032/+0.023/+0.034 vs +0.005/+0.004/+0.006 (I recomputed these
from the per-method table).

**The 32% top-1 figure is correct** — see U-2 below; this is the one place the
brief and the artefact disagree and the manuscript follows the artefact.

**Case study** (`pipeline_out/report/RESULTS.md`): all nine patient-level AUROCs
and all eighteen interval bounds; permutation null range 0.548–0.645 over 20
distinct replicates not containing 0.500; background-only 0.604 [0.528, 0.673]
vs 0.629, 0.595 vs 0.586 (control above headline), 0.549 vs 0.587; coil
prediction 0.921 [0.870, 0.966] on 136 subjects and 0.979 [0.953, 0.996] within
site; envelope 0.476; 16 confound results clearing chance; 102 training / 456
control runs; release batch 0.743 vs 0.633 and η² 0.108 vs 0.033.

**Reconstruction** (`pipeline_out/recon_fidelity/*.json`): r = 1.0000 brain and
knee, 0.9982 T2, 0.9835 DWI low-b-averaged (`per_file_lowb_averaged`), 0.9772
breast; both supplement tables including the per-slice/per-file split (breast
140 files / 2,240 slices in one table vs 2,224 / 139 in the other is correct —
16 slices and 1 file are `n_missing`); the anatomy-support null shifts and
margins.

**Simulation**: true AUC 0.6880, 200 datasets of 20 × 15, coverage 91.5% vs
46.5%, widths 0.370 vs 0.117, factor 3.18; metadata-on-synthetic 0.424.

**Frozen frame** (`paper/screen/frame_meta.json`): the supplement's verbatim
PubMed query is byte-identical to the frozen `query` field; 9,979 records;
SHA-256 `d611def0…`; run 2026-07-29 06:42:52 UTC; seed 20260729. The registration
timeline's commits `a64d202`, `b589efa`, `f484685` match `git log`.

**FDA scoping**: every claim the *main text* makes is corroborated by
`paper/FDA_PATH.md` — 0/30 slice-level metrics, all volumetric devices per
scan/case/subject, 1/30 at more than one analysis unit — and the required
disclaimer ("we make no claim that regulatory summaries hide the evaluation
unit") is present in both documents. See U-5 for the supplement's tally.

**Other spot checks that hold**: RSNA-STR PE run-length ratios 0.974 and 1.001;
fastMRI+ coverage 199 of 1,173; CAMELYON16 32 submissions / τ 0.754 / 61 of 496
(496 = C(32,2)); Ruan τ_a 0.603; Chen τ_a 0.927; Yan 59.7% vs 90.5% and our
55.7%; OsciiArt 0.33 vs 0.44 and the 2020-10-10 date; Yagis 30–55% and ~96% on
random labels; Badgeley 0.78 / 0.91 / 0.52 / 1.00; Maier-Hein τ 0.74–0.85;
trivial-fraction extremes table (all ten constructed rows); "every row has a
published margin of at least 0.21" (the tightest is Rempe's R = 16 arm at 0.214).

---

## 3. Unresolved — needs a human decision

**U-1 — RESOLVED 2026-07-30, and this audit had it backwards.** The paper's own
section IIIC settles it: *"The dataset comes with k-Space data of both T2 and
DWI. To show the feasibility of our approach, we only work on the DWI data."*
**DWI is the correct arm.** The 9,508-slice figure is Rempe et al.'s description
of the *dataset*, given one paragraph before they state which arm they use, and
it does not match the diffusion label file they describe using (9,490 rows) —
a discrepancy in their reporting, not evidence about their arm. Three further
DWI-only details in the same section agree: matrix 100 × 100 / FOV 200 mm, the
b=50/b=1000/b=0 averages, and the GRAPPA comparison (needed because fastMRI
prostate DWI ships 2× undersampled). Consequences applied: protocol rule 4 in
Table 7 now quotes **0.851**, not 0.854; the Results paragraph leads with the
diffusion arm, names it as the authors' own, and states the slice-count
discrepancy as the reason both arms are shown. `paper/audit_results.md` §3.1 has
been rewritten — it was the source of this error. `audit_targets.json` and
`pipeline/s12_rempe.py` were right all along and are unchanged.

The superseded reasoning is kept below for the record.

~~**U-1 (important). The task brief and `paper/audit_results.md` disagree about
which fastMRI Prostate arm is Rempe et al.'s.**~~ The brief I was given states
"Rempe et al. work on DIFFUSION, so DWI is correct". `paper/audit_results.md`
§3.1 states the opposite and gives evidence: Rempe et al.'s abstract reports
"312 subject and a total of 9508 slices", and 9,508 is the exact row count of
`t2_slice_level_labels.csv` — I confirmed the DWI file
(`pipeline_out/rempe_dwi_slice_level_labels.csv`) has 9,490 rows. That section
concludes "**T2 is the correct arm**", says the persisted artefact
`pipeline_out/rempe/positional_baseline.json` is already right, and says the
`anchor_correction` block in `paper/audit_targets.json` (which asserts DWI)
"should be reversed before submission".

I did **not** change the manuscript, because it does not depend on the answer:
both arms are reported side by side everywhere, protocol rule 4 quotes the T2
number (0.854, consistent with `audit_results.md`) and rule 5 quotes the DWI
number (0.851). **Someone must read Rempe et al.'s abstract and settle this**,
then fix whichever of `audit_targets.json` or the brief is wrong. If DWI turns
out to be correct, protocol rule 4 in Table 7 must change from 0.854 to 0.851.

**U-2. The brief's "top-1 reproduces in 43.1%" is a different statistic from the
manuscript's 32%.** `rankinversion.json` for brain has
`stability.top1_stability_slice = 0.3185` (the slice-level top-1 method
reproduces under subject resampling — which is what the manuscript's sentence
claims) and `agreement.top1_same_rate = 0.431` (the two *units* pick the same
top-1). `paper/rank_inversion.md` line 236 and its §3.3 table both print **32%**.
The manuscript is correct as written; no change made. Flagging so nobody
"corrects" it to 43.1% later.

**U-3. The brief's prevalence numbers do not match the artefact.** The brief says
"P1 complete-case 0/111; Wilson upper ~3.3%; unreachable ~31%; bound ~[0%, 31%]".
`pooled_final.json` gives the pre-registered result as 0/91, upper 4.1%,
unreachable 32.6%, bound [0.0%, 32.6%], and the post-hoc extension as 0/114 with
upper 3.3% and bound [0.0%, 32.9%]. The brief appears to blend the two
denominators. The manuscript follows the artefact and reports the bound as the
headline in the abstract, the Results, Table 5 and the Conclusion. No change.

Same pattern, smaller: the brief says the air-only control comes "within 0.018 of
the real headline" (artefact: 0.025 / −0.009 / 0.037), the flagship patient AUROC
is 0.4561 (that is the *released tool*; the manuscript headlines 0.4533 from the
independent reimplementation — see U-7), and breast reconstruction r is 0.97
(artefact 0.9772). The manuscript matches the artefacts in all three.

**U-4. The supplement cites nothing.** `supplement.tex` deliberately loads no
`.bib`, but it makes attributable factual claims — "a full census of 903 devices
coded on closely related fields has already been published", "21 CFR 892.2080 /
892.2070 / 892.2090". `refs.bib` already holds `wu2021fdaeval`,
`muehlematter2021approval` and `muehlematter2023predicate`. Either add a short
plain-text reference list to §S5 or drop the census sentence. Not changed —
adding a bibliography to a file that advertises itself as self-contained is an
editorial decision.

**U-5. The FDA tally cannot be verified, and two internal documents disagree.**
Supplement Table (§S5.3) says "States patient-level train/test disjointness:
**1/30**". `paper/FDA_PATH.md`'s table says "Asserts it at the patient/subject
level: **3/30**". The released artefact
`paper/fda_scoping/sample_frame.json` contains only the sample list
(`frame_source`, `frame_n`, `frame_with_summary`, `sample_seed`, `sample_n`,
`pdf_url_pattern`, `sample`) — **there is no coding sheet**, so none of the ten
tallies in that table is checkable by a reader or by me. Two actions needed:
reconcile 1/30 against 3/30, and release the per-device coding sheet or delete
the table. I did not adjudicate this because doing so requires downloading and
reading 30 FDA PDFs. The main text does not use either number.

**U-6. "DeepLesion's own defining paper published a location-only baseline in
2018".** The Limitations sentence now cites `yan2018deeplesiongraphs` (CVPR
2018, "Deep Lesion Graphs in the Wild"), which is where Table 1's "Baseline:
Location feature" row lives — but the *dataset*-defining paper is
`yan2018deeplesion` (*J Med Imaging* 2018). The Introduction handles this
correctly ("in the paper that defines the DeepLesion lesion-type task"); the
Limitations sentence does not. One-word fix, but it changes an attribution, so
an author should make it.

**U-7. Which RSNA ICH patient-level number is the headline.** The abstract, Key
Points, Results, Conclusion, Table 1 and Table 3 all use **0.453 [0.445, 0.461]**
— the independent reimplementation (`rsna_ich_unit_collapse.json`). Table 2 lists
the released tool at **0.4561 [0.4478, 0.4640]**. Both are real, Table 2 makes
the choice visible, and the "agree to 0.003" sentence covers the gap, so the
manuscript is internally consistent. But my brief designates 0.7376/0.4561 as
the flagship pair, i.e. the released tool. Deliberate choice or drift? If the
released tool is meant to be the headline, six locations change and the "0.070
below the null" arithmetic becomes 0.067.

**U-8. Minor, not fixed.** The abstract says "a rank-inversion analysis on 21
method configurations across five cohorts"; per-cohort counts are 13, 3, 21, 18,
3. 21 is the full configuration space (7 architectures × 3 input conditions, as
Limitations says) and Table 6 gives the per-cohort numbers, so it is defensible —
but a reviewer may read it as 21 per cohort.

**U-9. Not verified.** `refs.bib`'s header asserts every entry was checked
against a live record on 2026-07-30. I did not re-verify any DOI, PMID, volume,
page range or author list against a live source — no network lookups were made in
this pass. The one entry I know is unverified is `chen2025camcsa`, which I created
from a repository JSON and flagged in its own `note`.

---

## 4. Files changed

* `paper/tex/main.tex` — 32 citation-key remaps, 3 wrong-source citation
  repoints, bibliography block replaced, header comment rewritten, N-1 to N-7 and
  O-1 to O-6 applied.
* `paper/tex/refs.bib` — appended section 9 with `chen2025camcsa` and
  `gwet2008ac1`.
* `paper/tex/supplement.tex` — four line-breaking fixes only. **No number, claim
  or table cell in the supplement was changed.**
* `paper/tex/cover_letter.tex` — unchanged.
* `paper/tex/AUDIT.md` — this file.

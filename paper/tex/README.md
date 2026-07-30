# Overleaf packaging

Everything in this folder is written for a **stock Overleaf project on pdfLaTeX**.
No custom document class, no shell-escape, no exotic packages, no fonts to install.
If a file here fails to build on Overleaf, that is a bug in the file, not a missing
dependency — read "If a package is missing" at the bottom before working around it.

---

## What is in this folder

| file | what it is | compiles alone? |
|---|---|---|
| `main.tex` | the manuscript, 6 figures + 7 tables | needs `refs.bib` and `figures/` |
| `supplement.tex` | supplementary material, sections S1–S8 | needs `figures/`; no `.bib` |
| `cover_letter.tex` | cover letter to *Radiology: Artificial Intelligence* | **yes, with no other file at all** |
| `refs.bib` | bibliography, 51 entries, 41 cited by `main.tex` | not a document |
| `figures/*.pdf` | figures for the manuscript and the supplement | not documents |
| `make_figures.py` | regenerates `figures/` from the pipeline artefacts | not a document |
| `.gitignore` | LaTeX build artefacts, **plus two named figure exclusions** | — |

`supplement.tex` and `cover_letter.tex` do **not** `\input`, `\include`, `\ref` or
otherwise depend on `main.tex`, and neither loads a `.bib`. The cover letter still
builds from nothing at all.

**The supplement no longer does.** As of 2026-07-30 it carries three figures in
Section S8 and therefore needs the sibling `figures/` directory. That directory travels
in the same Overleaf zip, so nothing about the submission workflow changes, but the old
"delete every other file and it still builds" guarantee now applies only to the cover
letter. The header comment in `supplement.tex` says the same thing.

---

## Uploading to Overleaf

1. Zip the **contents** of this folder — `main.tex`, `supplement.tex`,
   `cover_letter.tex`, `refs.bib`, and the `figures/` directory. Zip the files,
   not the enclosing folder, or Overleaf will nest everything one level deep and
   `figures/fig1_collapse.pdf` will not resolve. The `figures/` directory is now
   needed by **both** `main.tex` and `supplement.tex`.
   **Two of the nine figures are not in this repository** — see *Figures*, below.
   Run `make_figures.py` before you zip, or the zip will be missing
   `fig6_qualitative_phase.pdf` and `figS3_qualitative_cohorts.pdf`.
2. In Overleaf: **New Project → Upload Project → drag the zip.**
3. **Menu → Settings → Compiler → pdfLaTeX.** Do not use XeLaTeX or LuaLaTeX. They
   will probably work, but nothing here has been checked against them, and `siunitx`
   plus `fontenc` behave differently enough to be worth avoiding on submission day.
4. **Menu → Settings → TeX Live version → the newest available.** These files use
   nothing version-sensitive, but newer is a shorter path to a fixed bug than older.
5. **Menu → Settings → Main document.** Overleaf picks one document as the one the
   green *Recompile* button builds. Set it to whichever of the three you are working
   on, and change it when you switch. This is the single most common reason someone
   reports that "the supplement did not update".

### Producing three separate PDFs

Overleaf builds one main document at a time. To export all three:

1. Set *Main document* to `main.tex`, recompile, **Download PDF**, rename to
   `manuscript.pdf`.
2. Set *Main document* to `supplement.tex`, recompile, **Download PDF**, rename to
   `supplement.pdf`.
3. Set *Main document* to `cover_letter.tex`, recompile, **Download PDF**, rename to
   `cover_letter.pdf`.

Overleaf caches aggressively. If a PDF looks stale after switching the main document,
use **Menu → Clear cached files** and recompile.

---

## Compile order

### `supplement.tex` — two passes, nothing else

```
pdflatex supplement.tex
pdflatex supplement.tex
```

The second pass is required: `hyperref` needs it for the internal links, `longtable`
needs it to settle column widths across page breaks, and the Section S8 figures are
`\ref`-ed from Section S4 before they appear. A single pass produces a readable PDF
with wrong table alignment and `??` in place of a figure number. On Overleaf, pressing
*Recompile* twice does this. `latexmk -pdf supplement.tex` does it in one command
locally.

There is no bibliography step. The supplement cites no references by key; the works it
names in prose (PRISMA 2020, the 21 CFR special controls, and the NYU fastMRI
references the data sharing agreement requires) are written out in full. **Do not
"tidy" the fastMRI references in Section S4 into `\citep` calls** — there is no `.bib`
here, and they would silently vanish.

### `cover_letter.tex` — one pass

```
pdflatex cover_letter.tex
```

No cross-references, no bibliography, no floats.

### `main.tex` — four passes, because of BibTeX

```
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Overleaf runs this sequence automatically when it sees `\bibliography{refs}`.
Locally, `latexmk -pdf main.tex` also does it.

The bibliography file is `refs.bib`, not `references.bib`, and `main.tex` names it as
`\bibliography{refs}`. Without it the reference list comes out empty and every citation
renders as `[?]` — including the three the fastMRI data sharing agreement requires. The
41 citation keys the manuscript expects are listed in the comment block at the top of
`main.tex`.

---

## Figures

`figures/` contains PDF vector figures. **PDF, not SVG.** pdfLaTeX cannot include SVG,
and the pipeline's own diagnostic figures under `paper/figures/` are SVG — those are
working artefacts, not manuscript figures, and must not be uploaded to Overleaf
directly. `make_figures.py` regenerates the PDF versions from the pipeline JSON, CSV
and HDF5 artefacts, and prints a source ledger naming the artefact behind every mark.

Nine figures, six in the article and three in the supplement:

| file | where | label | what it shows |
|---|---|---|---|
| `fig1_collapse.pdf` | main | `fig:collapse` | RSNA ICH, slice→patient AUROC, all six labels |
| `fig2_trivial_fraction.pdf` | main | `fig:fraction` | trivial fraction per benchmark-arm |
| `fig3_prisma_flow.pdf` | main | `fig:prisma` | PRISMA flow and per-block unreachable rate |
| `fig4_rank_inversion.pdf` | main | `fig:rank` | rank-inversion null, four panels |
| `fig5_case_study.pdf` | main | `fig:case` | case-study AUROCs and the background-only control |
| **`fig6_qualitative_phase.pdf`** | main | `fig:input` | **MRI.** The estimator's input, prostate T2, one slice per class |
| `figS1_acquisition_fingerprint.pdf` | S8.1 | `fig:confound` | AUROC at predicting the acquisition property |
| `figS2_recon_fidelity.pdf` | S8.2 | `fig:reconfid` | reconstruction fidelity and its anatomy-support floor |
| **`figS3_qualitative_cohorts.pdf`** | S8.3 | `fig:qualcohorts` | **MRI.** The same input on the other four cohorts |

`fig6` and `figS3` are the only two that contain pixels. The article had five figures
and no MRI image in it at all until 2026-07-30; `fig6` was added to the article and the
rest went to the supplement.

> **UNVERIFIED, check before submission:** nothing in this repository sources the
> journal's display-item limit. An earlier draft of this file asserted "the journal's
> limit is six" and the supplement asserted a "six-figure limit"; neither had a source,
> and the supplement now says "display-item budget" instead. Read the *Radiology: AI*
> Instructions for Authors and confirm whether figures and tables are capped separately
> or together — the manuscript currently carries **6 figures and 7 tables**.

Both imaging captions say in as many words that the figure **illustrates the input and
is not evidence for or against the phase hypothesis**. Do not soften that. The null
rests on the intervals and the falsification suite; nothing in this work turns on how a
slice looks.

### Three things in these figures that must not be "tidied" back

1. **`figS3`'s breast panels say `label = 1/0 (patient level)`, not "tumour annotated
   on this slice".** The breast label is the release's patient-level lesion status
   broadcast to all 32 cached slices, and its negative class contains nine patients
   with a recorded *benign* lesion — patient 287, the panel shown, among them.
   `make_figures._label_is_slice_specific` measures this from the index; a cohort earns
   the words "on this slice" only where some patient carries more than one distinct
   slice label. Prostate T2 and DWI do (31/67 and 24/45); breast does not.
2. **`figS1`'s within-site block is one row per site stratum, including the one below
   the power floor.** `s09_robustness.json` says so itself: *"the estimate is carried by
   ['NYU']; the ['TH'] stratum/a fall below the power floor and sit as low as 0.578, so
   the within-stratum claim rests on the large stratum and must be written that way."*
   The 0.979 is the NYU stratum, 98 of 136 subjects, **seed 42 of two**; the TH stratum
   is 0.542. Do not collapse this back to a single "stratified within site" number.
3. **`figS2`'s row labels count the slices each row plots, not the slices cached.**
   Breast is 2,224 of 2,240; the script exits if that count ever stops matching
   `per_slice.n` in `recon_fidelity_summary.json`, which is what supplement Table S16
   prints.

### Two figures are deliberately absent from this repository

`fig6_qualitative_phase.pdf` and `figS3_qualitative_cohorts.pdf` embed MRI slices, so
they are **fastMRI-derived image data**. The NYU fastMRI data sharing agreement permits
use "in academic publications and presentations with citations as provided in paragraph
6" but forbids redistribution of "any portion or all of the fastMRI Dataset or any
subsequent variables or data files derived from" it. A public git repository is
redistribution; a manuscript figure is not. Both files are therefore listed by name in
`.gitignore` and must be supplied to the journal out of band.

They are not lost — `make_figures.py` rebuilds both from `pipeline_out/cache/*.h5` in
one command, and prints the cache row, source HDF5, slice index and patient id of every
panel it draws. **Do not "fix" the missing figures by deleting those `.gitignore`
lines.** The line this repository draws is: derived *statistics* may be committed (that
is what `fig1`–`fig5` and `figS1`–`figS2` are), derived *images* may not.

The consequence for anyone building from a fresh clone: `main.tex` and `supplement.tex`
will both compile, but with a missing-file error for those two `\includegraphics` until
`make_figures.py` has been run against a local `pipeline_out/`. That is intended.

---

## The fastMRI attribution is not editorial, and cannot be cut for length

All five worked-example cohorts (brain, knee, breast, prostate T2, prostate DWI) are
NYU fastMRI releases. Paragraph 6 of the fastMRI Data Sharing Agreement makes three
things mandatory, and its own closing words are "further acknowledge that inclusion of
some variation of the language shown above is mandatory". All three are now in place:

| requirement | where it lives | how to check |
|---|---|---|
| ¶6(a) the literal string **"NYU fastMRI"** in the **abstract** | `main.tex`, Materials and Methods sentence of the abstract | `grep -c 'NYU fastMRI' main.tex` |
| ¶6(b) reference **Knoll et al. 2020** *and* the **arXiv paper 1811.08839** | `refs.bib` section 10, keys `knoll2020fastmri` and `zbontar2018fastmri`; written out in full prose in supplement §S4 | `grep -n 'knoll2020fastmri' main.tex` |
| ¶6(c) the **"Data used in the preparation of this article were obtained from the NYU fastMRI Initiative database"** paragraph in the **Methods** | first paragraph of `\subsection{The worked-example cohorts}` in `main.tex`; repeated at the head of §S4 in `supplement.tex` | `grep -c 'NYU fastMRI Initiative database' main.tex supplement.tex` |

The abstract is over its 250-word cap and will be cut before submission. **"NYU fastMRI"
is two words and must survive the cut.** There is a comment saying so directly above the
abstract in `main.tex`.

The supplement loads no `.bib`, so its copies of the Knoll and Zbontar references are
written out as prose. That is deliberate and is not a style slip.

---

## Before you export the final PDFs

A short list, because each of these has a specific consequence at the desk-reject
stage.

- **Check the three fastMRI attribution requirements above are all still present.**
  This is the only item on this list that is a term of a signed agreement rather than a
  journal preference. Run all three `grep`s in the table.
- **Run `make_figures.py` and confirm nine PDFs exist in `figures/`.** Two of them are
  git-ignored by design and will be absent from a fresh clone.
- **Fill the two placeholders in `cover_letter.tex`.** They are marked
  `<<< SUBMISSION DATE >>>` and `<<< REPOSITORY URL / ARCHIVAL DOI >>>`. Search the
  file for `<<<`; if that search returns anything, the letter is not ready.
- **Search `main.tex` for `\todo{`.** The `\todo` macro renders as bold bracketed text
  and is intended to be visible. Every occurrence is a missing affiliation, funding
  statement or conflict-of-interest disclosure. None may survive into the submitted
  PDF.
- **Search all three files for the string `pre-registered`.** The screen was *not*
  deposited with any registry. Supplementary Section S7 and `paper/registration.md`
  §5.4 forbid the unqualified word, along with "registered protocol", "registration
  number", "prospectively registered", and "sealed and timestamped in git" said of the
  screener files. The agreed wording is **"protocol-frozen"**.
- **Check that the manuscript title and the supplement title match**, including that
  wording.
- **Check the reference list is real, not empty.** If `refs.bib` was left out of the
  zip, every citation renders `[?]` and the two references the fastMRI agreement
  requires disappear silently.
- **Check that the numbers in the supplement match the manuscript.** Both should be
  regenerated from `paper/screen/analysis/pooled_final.json` and the
  `pipeline_out/` artefacts rather than retyped. The screen numbers moved once during
  drafting (an intermediate version of the flow reported 38 included / 29.6%
  unreachable, superseded by 91 included / 32.6% over 250 screened records); anything
  carrying the older figures is stale.

---

## If a package is missing

Every package used across the three documents ships with a full TeX Live install,
which is what Overleaf provides:

`fontenc`, `inputenc`, `lmodern`, `geometry`, `amsmath`, `amssymb`, `array`,
`booktabs`, `longtable`, `ragged2e`, `caption`, `microtype`, `parskip`, `hyperref`,
`graphicx`, `siunitx`, `natbib`.

So a "File `xyz.sty` not found" error on Overleaf almost always means one of three
things, in descending order of likelihood:

1. **You are on an old TeX Live release.** *Menu → Settings → TeX Live version →
   newest.* `siunitx` v3 in particular is only in recent releases, and older
   `siunitx` rejects some v3 options.
2. **The upload nested the files one directory deep.** Overleaf then finds `main.tex`
   but not `figures/`. Check the file tree in the left panel: `main.tex` should be at
   the top level, not inside a folder. Re-zip the contents rather than the folder.
3. **You genuinely need a package that is not there** — which should not happen with
   this list. In that case, do not paste the `.sty` file into the project. Replace the
   package instead, and the substitutions are cheap:
   - `microtype` — cosmetic only. Delete the `\usepackage{microtype}` line; nothing
     else changes.
   - `ragged2e` (supplement) — replace `\RaggedRight` with the built-in
     `\raggedright` in the two `\newcolumntype` definitions in the preamble. Slightly
     worse spacing, identical content.
   - `lmodern` — delete the line. You get Computer Modern instead of Latin Modern.
   - `parskip` (cover letter) — delete the line and add
     `\setlength{\parindent}{0pt}\setlength{\parskip}{0.8em}`.
   - `siunitx` (manuscript) — this one is not a one-line deletion; it formats the
     numbers. Fix the TeX Live version instead.

For anything else, the error message names the file. Report the full log rather than
guessing: Overleaf's *Raw logs* view is under the compile-output panel.

---

## Local compilation

If you would rather not use Overleaf, any TeX Live 2021 or newer works:

```bash
latexmk -pdf supplement.tex      # two passes, handled automatically
latexmk -pdf cover_letter.tex
latexmk -pdf main.tex            # runs BibTeX too
latexmk -C                       # clean every build artefact
```

`.gitignore` in this folder excludes the artefacts `latexmk` leaves behind, so
`git status` stays readable. The generated PDFs are ignored too — regenerate them
rather than committing them, so that a stale PDF can never be mistaken for the current
manuscript.

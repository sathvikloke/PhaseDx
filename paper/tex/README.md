# Overleaf packaging

Everything in this folder is written for a **stock Overleaf project on pdfLaTeX**.
No custom document class, no shell-escape, no exotic packages, no fonts to install.
If a file here fails to build on Overleaf, that is a bug in the file, not a missing
dependency — read "If a package is missing" at the bottom before working around it.

---

## What is in this folder

| file | what it is | compiles alone? |
|---|---|---|
| `main.tex` | the manuscript | yes — see the bibliography note |
| `supplement.tex` | supplementary material, sections S1–S7 | **yes, with no other file at all** |
| `cover_letter.tex` | cover letter to *Radiology: Artificial Intelligence* | **yes, with no other file at all** |
| `references.bib` | bibliography, 35 keys | not a document |
| `figures/*.pdf` | figures for the manuscript | not documents |
| `make_figures.py` | regenerates `figures/` from the pipeline artefacts | not a document |
| `.gitignore` | LaTeX build artefacts | — |

`supplement.tex` and `cover_letter.tex` do **not** `\input`, `\include`, `\ref` or
otherwise depend on `main.tex`. They load no graphics and no `.bib`. You can delete
every other file in the folder and they will still produce a PDF. This is deliberate:
the supplement and the cover letter are uploaded to the submission system as separate
files, and a supplement that only builds as part of the manuscript is a submission-day
emergency.

---

## Uploading to Overleaf

1. Zip the **contents** of this folder — `main.tex`, `supplement.tex`,
   `cover_letter.tex`, `references.bib`, and the `figures/` directory. Zip the files,
   not the enclosing folder, or Overleaf will nest everything one level deep and
   `figures/fig1_collapse.pdf` will not resolve.
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

The second pass is required: `hyperref` needs it for the internal links and
`longtable` needs it to settle column widths across page breaks. A single pass
produces a readable PDF with wrong table alignment and a "Rerun to get cross-references
right" warning. On Overleaf, pressing *Recompile* twice does this. `latexmk -pdf
supplement.tex` does it in one command locally.

There is no bibliography step. The supplement cites no references by key; the two
works it names in prose (PRISMA 2020, and the 21 CFR special controls) are written out
in full.

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

Overleaf runs this sequence automatically when it sees `\bibliography{references}`.
Locally, `latexmk -pdf main.tex` also does it.

**If `references.bib` is not in the folder, `main.tex` still compiles.** It emits a
placeholder in place of the reference list, via `\IfFileExists`. That is a drafting
convenience, not a submission state — check that the reference list is real before
exporting the final PDF. The 35 citation keys the manuscript expects are listed in the
comment block at the top of `main.tex`.

---

## Figures

`figures/` contains PDF vector figures. **PDF, not SVG.** pdfLaTeX cannot include SVG,
and the pipeline's own diagnostic figures under `paper/figures/` are SVG — those are
working artefacts, not manuscript figures, and must not be uploaded to Overleaf
directly. `make_figures.py` regenerates the PDF versions from the pipeline JSON
artefacts.

`supplement.tex` deliberately contains **no** `\includegraphics`. Everything in it is
a table or text, so it cannot fail to build because an image is missing or in the
wrong format.

---

## Before you export the final PDFs

A short list, because each of these has a specific consequence at the desk-reject
stage.

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
- **Check the reference list is real, not the `\IfFileExists` placeholder.**
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

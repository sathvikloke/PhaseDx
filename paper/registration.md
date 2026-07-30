# Registration record for the PhaseDx prevalence screen

**Status: the screen is NOT prospectively registered.** No deposit was made with OSF, PROSPERO
or any other public registry before screening began. `paper/screen_protocol.md` §11 named an
OSF deposit as an action item to be completed *before* screening; it was not done, and
screening has since run to completion. Any deposit made now is **retrospective** and must be
labelled so — in the deposit itself, in the manuscript, and in the cover letter.

What does exist is a version-controlled freeze: the protocol, the codebook, the frozen sampling
frame, the seeded permutation and the drawn sample were all committed to a public repository
before any analysis-sample record was coded, and the frame and permutation are content-addressed
by SHA-256 so a reader can verify that the sample was not re-drawn. That is a real and checkable
guarantee, and it is a **weaker** one than prospective registration. This file states exactly
how much weaker, so that nobody — reviewer, editor or co-author — has to take the difference on
trust.

---

## 1. The verifiable timeline

All times UTC, 2026-07-29. Sources: `git log`, the `date_run_utc` / `generated_utc` fields
written by the generating scripts, and filesystem modification times.

| # | event | time (UTC) | evidence | strength |
|---|---|---|---|---|
| 1 | PubMed frame retrieved, 9,979 PMIDs frozen | 06:42:52 | `paper/screen/frame_meta.json` → `date_run_utc`; mtime of `frame_pmids.txt` | script-written field + mtime |
| 2 | Seeded permutation drawn | 06:43:14 | mtime of `paper/screen/permutation.txt` | mtime only |
| 3 | Sample built from the permutation | 06:59:09 | `paper/screen_sample.json` → `generated_utc`; mtime agrees to the second | script-written field + mtime |
| 4 | **Freeze commit `a64d202`** — protocol v1.0, codebook v1.0, sample, frame, permutation, `build_sample.py`, `reproduce_frame.py` | **07:12:02** | `git log`; commit is an ancestor of `origin/main` on the public GitHub repository | git object + public push |
| 5 | `b589efa` (Zenodo citation metadata) | 07:22:11 | `git log` | git object |
| 6 | `f484685` (author list) | 07:46:50 | `git log` | git object |
| 7 | Screener S3 submits batch C | 08:01:22 | mtime of `paper/screen_batch_C.json` | **mtime only — file never committed** |
| 8 | Screener S1 submits batch A | 08:03:10 | mtime of `paper/screen_batch_A.json` | **mtime only — file never committed** |
| 9 | Screener S2 submits batch B | 08:07:06 | mtime of `paper/screen_batch_B.json` | **mtime only — file never committed** |
| 10 | Screener S4 submits batch D | 08:07:36 | mtime of `paper/screen_batch_D.json` | **mtime only — file never committed** |

**The claim this supports.** The freeze commit precedes the earliest screener submission by
**49 minutes 20 seconds**, and precedes the latest by 55 minutes 34 seconds. Commits 5 and 6
fall inside that window and touch neither the protocol, the codebook, nor the sample:
`git log -- paper/screen_protocol.md paper/screen_frame.json paper/screen_sample.json` returns
`a64d202` and nothing else.

**The sample was never re-drawn.** `paper/screen_sample.json` in the working tree is
byte-identical to the blob in `a64d202`
(SHA-256 `56ea35d211f64b2ee1671af77f1b4fa0aaa1ba6e510ed4cfc19468c93c1f3eba` both sides). Which
PMIDs are in the analysis sample and which batch each falls in is determined solely by
`permutation.txt`, whose digest is a frozen constant inside `reproduce_frame.py`.

**The digests verify, offline, today:**

```
$ python paper/screen/reproduce_frame.py --verify
[OK ] frame: d611def0785f3a5e7b7489364959f1d3471b61651f98a3ed049252654264374b
[OK ] permutation-on-disk: dad12a30b77d1213ac5e8ced89cf3a6620977b5734b5076641bb8adb2db74a1a
[OK ] permutation-recomputed: dad12a30b77d1213ac5e8ced89cf3a6620977b5734b5076641bb8adb2db74a1a
[OK ] frame size: 9979 (expected 9979)
```

The third line is the one that matters: the permutation is *recomputed* from the frozen PMID
list with `random.Random(20260729).shuffle` and reproduces the committed digest. The draw
cannot have been re-rolled without changing that digest, and the digest is a literal constant in
a file committed at 07:12:02.

---

## 2. What the evidence above does **not** establish

Stated plainly, because a hostile reader will find each of these and it is better that we found
them first.

1. **Git timestamps are self-asserted.** Both the author and committer dates in `a64d202` were
   written by the committing machine and can be set to any value. The only third-party element
   is that the commit exists on the public GitHub repository
   (`https://github.com/sathvikloke/PhaseDx`, ancestor of `origin/main`), so GitHub's own
   receipt of the push is independent of our clock. That receipt is retrievable from the GitHub
   API and should be captured and quoted if a reviewer asks. **This is not equivalent to a
   registry timestamp**, which is issued by a party with no interest in the result.
2. **The four sealed screener files were never committed.** They are untracked working-tree
   files. Protocol §6 says each overlap batch is *"submitted as a sealed file, timestamped in
   git, before any screener sees another's codes"*. The sealing and the independence held as a
   matter of process — each file carries a written independence statement, and no screener's
   file references another's — but **the git timestamp part of that requirement was not met.**
   Their only timestamps are filesystem mtimes, which are trivially rewritable and are not
   evidence in the sense a reviewer needs.
3. **The screeners' self-declared timestamps are wrong and must not be quoted.** Batch A
   declares `submitted_utc = 2026-07-29T03:05:00Z`; batch D declares
   `coded_utc = 2026-07-29T00:00:00Z`; batches B and C give a date with no time. Read literally
   as UTC, batch A's stamp precedes the retrieval of the frame itself (06:42:52 UTC), which is
   impossible. The explanation is mundane — local time (UTC−05:00) written with a `Z` suffix,
   and a placeholder midnight in batch D — but the fields are unreliable and the filesystem
   mtimes, which are internally consistent with every script-written UTC field, are the record
   used above.
4. **The protocol in the working tree is no longer the protocol that was committed.**
   `paper/screen_protocol.md` and `paper/screen_frame.json` are at v1.2 and are **uncommitted**.
   The committed v1.0 blobs are recoverable with `git show a64d202:paper/screen_protocol.md`,
   and the diff is the amendment record. Both files should be committed so the amendment is
   itself version-controlled rather than living only on one disk.

### Remedy: fix the sealed files now

The four screener files cannot be retroactively given a trustworthy timestamp, but they can be
frozen from this point forward. Their SHA-256 digests as of this record:

| file | SHA-256 |
|---|---|
| `paper/screen_batch_A.json` | `1177139275b2a92e170187348198a9f32d3e948c3c80c853fa613fd86d83ecbe` |
| `paper/screen_batch_B.json` | `a2b3afe7bedfebe2384577cab897032ac4e9ea9d52c7a8d960004271ba665db8` |
| `paper/screen_batch_C.json` | `49cc92f5bf3a6712186330219a4952ba5ae65f9d59acea9bec7604672a781ab8` |
| `paper/screen_batch_D.json` | `aea1f1b1cbf8c7f669df6bfc9c675026c926aad829a16eb4b94fc62c36c74815` |
| `paper/screen_sample.json` | `56ea35d211f64b2ee1671af77f1b4fa0aaa1ba6e510ed4cfc19468c93c1f3eba` |
| `paper/screen/frame_pmids.txt` | `939ea20b591a89f712cec19a5a70b88fcfe469268227813169b663a20d8b2474` |
| `paper/screen/permutation.txt` | `9e4597ba2e9bb3b6d5fc4dcb6f4d28922da0ec36e7beb36423b73adf8d42a563` |

Committing these four files, and the v1.2 protocol and codebook, is a five-minute action and
should be done before any deposit.

---

## 3. The registry deposit that was not made

`paper/screen_protocol.md` §11 reads, verbatim:

> The timestamp of the git commit that adds these five files is the pre-registration timestamp.
> Deposit on OSF, with that commit hash recorded in the deposit, is an outstanding action item
> and should happen before screening begins.

It did not happen before screening began. Screening ran between 08:01 and 08:08 UTC on
2026-07-29, and the adjudication (v1.2) and access-recovery (v1.3) passes ran later the same
day. There is therefore **no prospective registration of any kind** — no OSF registration, no
PROSPERO record, no AsPredicted entry, no third-party timestamp of any sort.

PROSPERO would in any case have been the wrong registry: it accepts systematic reviews with
health-related outcomes, and a methodological prevalence screen of reporting practice is
normally out of scope. OSF Registries accepts meta-research protocols and is the appropriate
destination. That does not soften anything above; it only says where the deposit should go.

### If a deposit is made now, it must say this

A retrospective deposit is still worth making — it fixes the artefacts publicly and lets a
reader diff our claims against a third-party copy — but only if it is honestly labelled. The
deposit must, at minimum:

- Carry **"Retrospective registration"** in the title or the first line of the description.
- State the deposit date, and state that **screening, adjudication and analysis were complete
  before the deposit was made.**
- Record the freeze commit hash `a64d202` and the repository URL, and state that the freeze
  claim rests on that commit and on GitHub's receipt of the push, not on the registry.
- Record `frame_sha256 = d611def0…374b` and
  `permutation_sha256 = dad12a30…4a1a`, and the seven digests in §2 above.
- Attach the protocol at **both** v1.0 and v1.2, so the amendment is visible in the deposit
  rather than only in our changelog.
- Not use the words "pre-registered", "pre-registration" or "registered protocol" without the
  qualifier "retrospectively deposited" adjacent to them.

---

## 4. What changed after coding began

Three changes. Each is logged in `paper/screen_protocol.md` §12 and in the codebook changelog;
this section states the direction of each one's effect, which is the question a reviewer is
actually asking.

### v1.1 — metadata correction. No effect on the sample.

The first build of `screen_sample.json` read identifiers with `iter("ArticleId")`, which walks
each PubMed record's embedded reference list, so 229/400 DOIs and 276/400 PMCIDs were taken from
a *cited* paper rather than the article. Fixed to scoped lookups in `build_sample.py`, which now
carries a seeded 25-DOI Crossref title-match regression test (25/25 agree). **Which PMIDs are
sampled, and which batch each falls in, were never affected**: allocation is a function of
`permutation.txt` alone and its digest is unchanged. The §7 open-access counts were restated
(60 PMC / 3 CC / 31 closed / 6 unknown, previously 65/3/27/5).

### v1.2 — codebook amendment. This is the one that needs justifying.

**Why it fired.** The protocol pre-specified, at §6, that if Fleiss' κ on the primary flag fell
below 0.60 — or, under the paradox guard, if raw agreement fell below 90% — a documented
adjudication round, a codebook amendment and a re-coding of every already-coded paper became
**mandatory**. Observed: κ = −0.015, raw agreement 65.6%. Both floors failed. The amendment is
therefore the execution of a pre-registered remedy, not a discretionary edit.

**What it found.** The disagreement was not about the primary flag. On the six overlap papers
that all four screeners both obtained and included, agreement on the P1 flag is 100% and all
four screeners produce identical vectors on all six `trivial_baseline` sub-flags. Across all 145
sealed codes, no P1-family sub-flag is coded true by anyone. The measured disagreement was in
reachability and eligibility, and on 9 of the 15 overlap papers it was purely the placeholder
the form forced for "could not be assessed": `trivial_baseline` was declared boolean with no
third level, so S1 wrote `false`, S2 wrote `null`, S3 wrote `"unclear"` and S4 wrote `false` for
the same evidence. That is a codebook defect, not a screener failure.

**What changed.** Fourteen decision rules D1–D14, each logged against the specific disagreement
it resolves, and four missing enum levels (`not_assessable` on the six sub-flags;
`not_applicable` on descriptive fields, available only where `final_inclusion ≠ included`;
`lesion_or_roi` on `split_unit`; `not_stated` on `code_availability`).

**What did not change.** The frame, the permutation, the sample, the four sealed screener files,
every endpoint definition, the interval method, and the 15% censoring threshold.

**Effect on the endpoints, including the direction.** The primary is unchanged: P1 = 0/35
complete-case, headline bounding interval [0.0%, 36.4%]. Two secondaries move by one record
each, **in opposite directions**: S4 falls 13/35 → 12/35, and S5 rises 0/35 → **1/35**. S5
moving off zero makes the literature look *better* on the very endpoint the paper accuses it of
ignoring; it was adopted for that reason, not despite it. The codebook carries an explicit
`_amendment_direction_rule` requiring that no rule be changed to improve a number.

**The honest limit.** Post-amendment agreement (κ = 0.932, raw 95.6%) is a **counterfactual
re-encoding of the same four sealed files** under the two amendments that add a missing level
and cannot change a reading. It is not an independent re-rating. A genuine post-amendment
reliability estimate requires a fresh four-screener re-coding under v1.2, which has not been
done. The paper must say so.

### v1.3 — access ladder re-run. Narrows the bound; does not move the primary.

Rungs 1–4 were worked again against every record coded `unreachable_*`. Four of twenty were
recovered and coded in full, including the mandatory 14-term full-text search. One of the four
(36170844) proved **ineligible** on full text and therefore left the eligible denominator
rather than moving into it, so the denominator fell 55 → 54. Unreachable 20 → 16; S6 36.4% →
29.6%; headline bounding interval [0.0%, 36.4%] → [0.0%, 29.6%]. **P1 complete-case 0/35 → 0/38,
still exactly zero.** 29.6% remains above the 15% threshold, so the bound remains the headline.
No rule was changed and no sealed file was edited: the recovery is an analysis-time overlay
(`paper/screen/access_recovery.json`, applied by
`paper/screen/analysis/recompute_with_recovery.py`, which prints as-sealed and post-recovery
numbers side by side).

### Requirements that were not met at all

These are protocol *deviations*, not amendments — nothing was rewritten to accommodate them, and
they remain outstanding. They are itemised in `paper/prisma_flow.md`:

- the extension rule (target 75 included papers) triggered and was not executed;
- rung 5 of the access ladder (interlibrary loan / author request, 21-day wait) was never
  initiated by any screener;
- the 20% within-batch re-coding by the next screener in the A→B→C→D→A cycle was not carried
  out;
- second-screener adjudication of records flagged `low` confidence or `flag_for_adjudication`
  was not carried out for the 49 flagged single-screened records.

---

## 5. Wording for the paper

Drafted here so it is written once and reused verbatim. **None of it implies prospective
registration.**

### 5.1 Methods, prevalence-screen subsection (use this one)

> The screening protocol, extraction codebook, sampling frame and seeded sample were fixed and
> committed to a public version-controlled repository before any record in the analysis sample
> was coded (commit `a64d202`, 2026-07-29 07:12 UTC; frame SHA-256 `d611def0…`, permutation
> SHA-256 `dad12a30…`, both re-derivable offline with the released script). **The screen was
> not deposited with a public registry in advance, and is therefore not prospectively
> registered.** A retrospective deposit of the protocol and all artefacts is available at
> [DOI/URL]. Three amendments were made after coding began and are documented with their
> direction of effect in the protocol changelog: a metadata correction that did not alter the
> sample (v1.1); a codebook amendment executing the protocol's own pre-specified agreement
> remedy, which left the primary endpoint unchanged and moved two secondary endpoints in
> opposite directions (v1.2); and a second pass of the full-text access ladder that recovered
> four of twenty unreachable reports (v1.3).

### 5.2 Other Information — "Registration and protocol" (PRISMA items 24a–24c)

> **24a.** This review was **not registered** with PROSPERO, OSF Registries or any other public
> registry prior to its conduct. A retrospective deposit of the protocol and all artefacts is
> at [DOI/URL], made after screening and analysis were complete and labelled as retrospective.
>
> **24b.** The protocol is available in full at `paper/screen_protocol.md` in the repository
> [URL], together with the extraction codebook (`paper/screen_frame.json`), the frozen sampling
> frame with its SHA-256 digest, the seeded permutation, the four sealed screener submissions,
> and every analysis script. The version frozen before screening is recoverable as
> `git show a64d202:paper/screen_protocol.md`.
>
> **24c.** Three amendments were made after screening began (protocol v1.1, v1.2 and v1.3), each
> dated and logged in the protocol changelog with the specific disagreement or failure that
> prompted it, the endpoints affected, and the direction of the change. The v1.2 amendment was
> the execution of a remedy pre-specified in the frozen protocol, triggered by a pre-specified
> agreement threshold. Protocol deviations that were *not* remedied — an unexecuted sample
> extension rule, an unexecuted access-ladder rung, and unexecuted duplicate re-coding — are
> reported in the flow document rather than in a supplement.

### 5.3 One sentence, for the abstract or cover letter

> The screening protocol and the seeded sample were frozen in a public repository before
> screening, with content digests that a reader can re-derive; the screen was not registered
> with a public registry in advance and is not described as pre-registered.

### 5.4 Phrases that must not appear

Search the manuscript for each of these before submission. The screen currently satisfies none
of them.

| do not write | why |
|---|---|
| "pre-registered on OSF" | no deposit was made |
| "registered protocol", "registration number" | there is no registry record |
| "prospectively registered" | false |
| "pre-registered" **unqualified** | reads as registry registration to a clinical-journal editor; use "protocol frozen in a public repository before screening" |
| "sealed and timestamped in git" (of the screener files) | those four files were never committed |

**Note.** `paper/screen_protocol.md` currently titles itself *"Pre-registered protocol"* and
`paper/screen_frame.json` carries `_status: "PRE-REGISTERED…"`. Those strings are internal to
the artefacts and describe a git-freeze, not a registry deposit. Either qualify them in place or
add a one-line header pointing at this file; do not let the phrase reach the manuscript
unqualified.

---

## 6. Verifying every claim in this file

```bash
# the freeze commit, and that nothing later touched the protocol, codebook or sample
git log --format='%H %aI %s' -- paper/screen_protocol.md paper/screen_frame.json \
                                paper/screen_sample.json

# the freeze commit is public
git merge-base --is-ancestor a64d202 origin/main && echo "on origin/main"

# the sample was never re-drawn
git show a64d202:paper/screen_sample.json | shasum -a 256
shasum -a 256 paper/screen_sample.json

# the frame and the permutation, offline
python paper/screen/reproduce_frame.py --verify

# the screener files have no git timestamp (this returns nothing, which is the point)
git log -- paper/screen_batch_A.json paper/screen_batch_B.json \
           paper/screen_batch_C.json paper/screen_batch_D.json

# submission times, from the filesystem, in UTC
TZ=UTC stat -f '%Sm %N' -t '%Y-%m-%dT%H:%M:%SZ' paper/screen_batch_*.json

# the committed protocol was v1.0
git show a64d202:paper/screen_protocol.md | head -3
```

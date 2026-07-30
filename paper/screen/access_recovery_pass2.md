# Access recovery, pass 2 — the sixteen records left unreachable in the original 100-paper sample

**Companion to** `paper/screen/access_recovery_pass2.json` (the machine-readable overlay).
**Run** 2026-07-30. **Result: 0 of 16 recovered. The bound does not move.**

Sealed files (`screen_batch_{A,B,C,D}.json`, `screen_reserve_R*.json`) are untouched.
`records[]` in the overlay is empty, so `recompute_with_recovery.py` prints exactly what it
printed before this pass ran. The value of this pass is negative evidence, recorded per
record so nobody spends the effort again.

---

## Headline

| | before pass 2 | after pass 2 |
|---|---|---|
| eligible-looking | 54 | 54 |
| included + reachable | 38 | 38 |
| eligible but unreachable | 16 | **16** |
| S6 unreachable rate | 29.6% [19.1%, 42.8%] | **29.6% [19.1%, 42.8%]** |
| P1 complete-case | 0/38 = 0.0% [0.0%, 9.2%] | 0/38 = 0.0% [0.0%, 9.2%] |
| **P1 bounding interval (headline)** | **[0.0%, 29.6%]** | **[0.0%, 29.6%]** |
| 15% threshold breached | yes | yes |

**Endpoint that worked: none.**

## The task's leading hypothesis is falsified, and that is the main finding

The brief said the PMC OA Web Service was "the single most likely win" because machine
endpoints usually bypass the web layer. It is not blocked. It is **inapplicable**:

- PMC ID converter returns **no PMCID** for any of the 16.
- `esearch db=pmc` returns **0 hits** for all 16 by exact title **and** by DOI.
- Europe PMC `fullTextXML` → **HTTP 404 ×16**. `supplementaryFiles` → **HTTP 404 ×16**.
- Europe PMC title search across **all** sources (MED / PMC / PPR preprints / bookXML /
  fulltextRepo) returns only the MED abstract record.

None of these 16 papers has ever been deposited in PMC. There is nothing there to fetch.

## Everything else that was tried

~28 endpoints per record, ~455 lookups. Per-record outcomes are in
`still_unreachable[].endpoints_tried`.

| route | outcome across the 16 |
|---|---|
| PMC OA Web Service / idconv / esearch db=pmc | no PMC deposit exists — route excluded, not blocked |
| Europe PMC fullTextXML, supplementaryFiles, all-source title search | 404 / no full text |
| Europe PMC annotations API | returns **title + abstract** annotations only — corroboration, never a substitute for reading |
| Unpaywall | 13 closed, 3 OA (bronze / hybrid CC BY / gold CC BY-NC-ND); `has_repository_copy` false for 15/16 |
| OpenAlex | agrees with Unpaywall; `any_repository_has_fulltext` false for 15/16 |
| Semantic Scholar Graph | `openAccessPdf` empty for all 16 |
| Crossref registered TDM / syndication / similarity links | Springer → bot "Client Challenge"; Elsevier TDM → **metadata only** (~1.9 kB coredata, body needs an API key); IEEE → JS shell; Wiley/OUP/RSNA → 403 |
| OpenAIRE | records exist, **no fulltext instance** for any |
| CORE v3 | 0 hits ×15. One hit (40147601) is a DOAJ metadata harvest with `downloadUrl:""`; the advertised `core.ac.uk/download/655810328.pdf` resolves to a signed URL returning **404** |
| DOAJ | only 40147601; its fulltext link points back to ScienceDirect |
| DataCite | 404 ×16 |
| arXiv (exact-title and loose all-field) | **0/16** |
| Zenodo / HAL / OSF preprints | 0/16 each |
| Wayback CDX (52 candidate URLs + OUP wildcards) | one Wiley landing-page snapshot (abstract + refs only, already read in pass 1); IEEE JS shells; the one capture of the OUP article is an **archived 403** |
| Internet Archive Scholar | reachable this pass (pass 1 hit a session challenge) — metadata-only records, no `/access/` link |
| PubMed LinkOut | publisher links only; see the 39107903 note below |

## Confirmed and strengthened: the three open-access records are blocked on *our* side

Pass 1 said 37222638, 42153825 and 40147601 are open access and blocked by publisher bot
detection. Pass 2 confirms the OA status independently and establishes something stronger:

> `pubs.rsna.org`, `onlinelibrary.wiley.com`, `www.sciencedirect.com`, `academic.oup.com`,
> `link.springer.com` and `ieeexplore.ieee.org` are blocked **at the origin by this
> environment's browsing policy** — in the built-in browser pane *and* in a real connected
> Chrome instance, both returning "blocked by policy" before any request leaves.

So the obstruction is not the literature and not only publisher bot detection; it is the
execution environment. That is the honest attribution, and it is why these three stay
counted as unreachable rather than quietly reclassified.

Licence evidence, keyless and reproducible:

- **37222638** J Magn Reson Imaging — Unpaywall bronze, publisher PDF at `onlinelibrary.wiley.com/doi/pdfdirect/10.1002/jmri.28787`.
- **42153825** Radiology — Unpaywall **and** OpenAlex **and** OpenAIRE all report hybrid OA under **CC BY**.
- **40147601** NeuroImage — Elsevier's own coredata endpoint, no key required: `openaccess=1`, `openaccessType=Full`, `openaccessUserLicense=CC BY-NC-ND 4.0`; also DOAJ-listed and `journal_is_in_doaj=true`.

## One new lead for the next operator

**39107903** (Dentomaxillofac Radiol, OUP). Unpaywall, OpenAlex, Semantic Scholar and
OpenAIRE all say closed, and Crossref registers OUP's standard *non-OA* reuse-rights URL —
so it stays coded closed. But **NLM's own LinkOut marks the Silverchair full-text link a
"free resource."** That signal is uncorroborated (PubMed esummary lists no free-full-text
attribute, the journal is not in DOAJ, the only Wayback capture is an archived 403), so it is
**not** upgraded here. It is the fourth record to re-test on an unblocked network.

## Routes that are untried, not failed

Worth flagging so the next person starts here rather than repeating the ledger above:

1. **`api.fatcat.wiki`** (Internet Archive Scholar's machine endpoint) — connection hangs from
   this environment, no response in 25 s. Never actually answered.
2. **Memento TimeTravel** (`timetravel.mementoweb.org`) — HTTP 000. Would have covered
   archive.today and non-IA archives.
3. **General web search** — the `WebSearch` tool errored on every call; DuckDuckGo and Mojeek
   both served bot CAPTCHAs, which were **not** solved (prohibited). So no search-engine route
   to an author copy or institutional repository was ever exercised.
4. The six **policy-blocked publisher domains**, from any unblocked network.

## Rules observed

- **No unauthorised source.** Sci-Hub, LibGen, Anna's Archive and equivalents were never
  fetched, not even to test whether they held a record; search hits pointing at such hosts
  were discarded unread.
- **No circumvention.** Springer's client challenge, IEEE's script shell, ScienceDirect's
  "are you a robot" captcha, and the DuckDuckGo and Mojeek challenges were all logged as
  failures and left alone. No browser User-Agent was spoofed — every request identified
  itself as `PhaseDx-screen/1.0` with a contact address. No new Wayback "Save Page Now"
  snapshot was created, because that writes to a public archive.
- **No unevidenced codes.** Nothing was coded, in either direction. Obtaining no body means
  the mandatory 14-term search over body **and** supplement could not be run, and under
  codebook v1.2 an unevidenced negative is invalid. All 16 stay at their sealed codes.

## Looking for a positive

The instruction to hunt a positive as hard as the absence was honoured but could not be
exercised: asserting a zero-image baseline is *present* needs a verbatim quote from the body,
and no body was obtained. Two abstract-level observations are logged so they are not lost,
and neither is coded:

- **37222638**'s abstract describes a clinical-variables-only comparator model
  (AUC 0.741 / 0.772 / 0.675) — would bear on secondary endpoint S1 (noted already in pass 1).
- **42489954**'s abstract suggests a functional-connectivity-graph input — would bear on
  eligibility (E-DERIV).

An abstract is not evidence under this codebook.

## What it would take to move the bound

| scenario | unreachable | rate | P1 bound | headline still the bound? |
|---|---|---|---|---|
| now | 16/54 | 29.6% [19.1%, 42.8%] | [0.0%, 29.6%] | yes |
| the three OA records read on an unblocked network | 13/54 | 24.1% [14.6%, 36.9%] | [0.0%, 24.1%] | yes |
| + 39107903 also free | 12/54 | 22.2% [13.2%, 34.9%] | [0.0%, 22.2%] | yes |

Even recovering every environment-attributable record leaves the rate above the 15%
threshold in `screen_protocol.md` sec.7 rule 4. **The headline is the bounding interval
either way, and it is insensitive to this environment's limitations.** Moving *below* the
threshold would require reading roughly six of the twelve genuinely paywalled records —
which needs institutional subscriptions or author contact, not another scripted pass.

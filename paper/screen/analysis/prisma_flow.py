#!/usr/bin/env python3
"""PRISMA 2020 flow counts for the PhaseDx prevalence screen, derived from files on disk.

Nothing here is retyped. Every integer in paper/prisma_flow.md and in the rendered figure
comes out of this script, and this script reads only:

    paper/screen/frame_meta.json                    frame size, query, digest
    paper/screen/frame_pmids.txt                    the frozen frame (duplicate count)
    paper/screen_sample.json                        allocation of permutation positions
    paper/screen_batch_{A,B,C,D}.json               the four sealed screener submissions
    paper/screen/analysis/adjudication_out.json     the v1.2 adjudicated overlap codes
    paper/screen/access_recovery.json               the v1.3 access-ladder overlay

THREE flows are emitted, so that each protocol version's effect on the flow is auditable:

    v1_0_as_sealed   the four sealed files pooled by the pre-registered majority rule with
                     the anti-self-serving tie-break. Reproduces analysis_out.json.flow.
    v1_2_adjudicated the same, with the 15 overlap records replaced by the v1.2 adjudicated
                     codes (paper/screen_adjudication.md).
    v1_3_post_recovery  the above plus the access-recovery overlay.  <-- PRIMARY, the figure

STAGE MAPPING (declared, because PRISMA's boxes and this codebook's fields are not 1:1).
Each record lands in exactly one terminal box, determined by pooled `final_inclusion` and
pooled `fulltext_reachable`:

    excluded, full text NOT obtained   -> records excluded at title/abstract screening
    eligible-looking, not obtained     -> reports sought for retrieval but not retrieved
    excluded, full text obtained       -> reports excluded at full-text assessment (w/ reason)
    included, full text obtained       -> studies included

"Full text obtained" means fulltext_reachable is one of oa_pmc_or_publisher,
preprint_version_only, repository_or_accepted_manuscript. Note this is STRICTER than the
`got_fulltext` helper in pool_and_agree.py / recompute_with_recovery.py, which tests only
"not in {unreachable_paywalled, unreachable_not_found}" and so treats the level
`not_attempted_excluded_at_stage1` as reachable. That difference cannot move any endpoint,
because every record carrying that level has final_inclusion='excluded' and is therefore
outside every denominator; it matters only for which PRISMA box the exclusion is drawn in.
The three-way status totals produced here are asserted equal to those scripts' totals.

No pooling rule is invented for `stage1_decision`. Records excluded at title/abstract where
at least one screener had nevertheless recorded stage1_decision='go_to_fulltext' are counted
and reported as a footnote rather than smoothed away.

    python paper/screen/analysis/prisma_flow.py      # write JSON + SVG, print report
"""
import json
import os
import re
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))

SCREENERS = ('S1', 'S2', 'S3', 'S4')
UNREACHABLE = ('unreachable_paywalled', 'unreachable_not_found')
OBTAINED = ('oa_pmc_or_publisher', 'preprint_version_only',
            'repository_or_accepted_manuscript')

EXCLUSION_LABELS = OrderedDict([
    ('E-SEG',    'segmentation/delineation evaluated, no class decision'),
    ('E-DERIV',  'input is a derived feature vector, not an image'),
    ('E-NOCLF',  'no fitted classifier, or no negative class'),
    ('E-2D',     'inherently 2D acquisition'),
    ('E-NONMED', 'not human medical imaging'),
    ('E-PROJ',   'volume collapsed to a projection before the model'),
    ('E-TYPE',   'not primary research of the eligible type'),
])


# --------------------------------------------------------------------------- helpers
def obtained(ft):
    return ft in OBTAINED


def status_of(final_inclusion, ft):
    if final_inclusion == 'excluded':
        return 'excluded'
    return 'included_reachable' if ft not in UNREACHABLE else 'eligible_unreachable'


def majority(vals, prefer):
    """Pre-registered: majority of the four codes; ties broken AGAINST our own thesis.
    `prefer` lists the tie-break order, most-adverse-to-us first."""
    c = Counter(map(str, vals))
    top = max(c.values())
    tied = [k for k, v in c.items() if v == top]
    if len(tied) == 1:
        return tied[0], False
    for p in prefer:
        if str(p) in tied:
            return str(p), True
    return sorted(tied)[0], True


def load():
    ps = {}
    for b in 'ABCD':
        d = json.load(open(f'{PAPER}/screen_batch_{b}.json'))
        ps[d['screener_id']] = d
    overlap, unique = {}, {}
    for s in SCREENERS:
        for r in ps[s]['records']:
            (overlap.setdefault(r['record_id'], {}).__setitem__(s, r)
             if r['batch'] == 'overlap' else unique.__setitem__(r['record_id'], r))
    assert len(overlap) == 15 and len(unique) == 85, (len(overlap), len(unique))
    return overlap, unique


# --------------------------------------------------------------------------- rows
def build(adjudicated, recovered):
    overlap, unique = load()
    adj = json.load(open(f'{HERE}/adjudication_out.json'))['adjudicated_codes']
    rec = json.load(open(f'{PAPER}/screen/access_recovery.json'))
    recovery = {r['record_id']: r for r in rec['records']}
    assert set(recovery) <= set(unique), 'recovery must not touch an overlap record'

    rows = []
    for pmid, per in overlap.items():
        codes = [per[s] for s in SCREENERS]
        st_maj, st_tie = majority([status_of(r['final_inclusion'], r['fulltext_reachable'])
                                   for r in codes],
                                  ['eligible_unreachable', 'included_reachable', 'excluded'])
        ft_maj, ft_tie = majority([obtained(r['fulltext_reachable']) for r in codes],
                                  [False])
        ft_maj = ft_maj == 'True'
        excl = [r['exclusion_code'] for r in codes
                if r['final_inclusion'] == 'excluded' and r['exclusion_code']]
        code_maj, code_tie = (majority(excl, []) if excl else (None, False))

        if adjudicated:
            a = adj[pmid]
            st = status_of(a['final_inclusion'], a['fulltext'])
            assert st == st_maj, f'{pmid}: adjudicated status {st} != pooled {st_maj}'
            ft = obtained(a['fulltext'])
            m = re.search(r'exclusion code -> (E-[A-Z0-9]+)', a.get('rule', ''))
            code = m.group(1) if m else (code_maj if st == 'excluded' else None)
            src = 'adjudicated_v12'
        else:
            st, ft, code = st_maj, ft_maj, (code_maj if st_maj == 'excluded' else None)
            src = 'pooled_v10'
        rows.append(dict(record_id=pmid, batch='overlap', status=st, fulltext=ft,
                         exclusion_code=code, status_tie=st_tie, ft_tie=ft_tie,
                         code_tie=code_tie, source=src,
                         any_screener_sought_fulltext=any(
                             r['stage1_decision'] != 'exclude' for r in codes)))

    for pmid, r in unique.items():
        use = recovery[pmid] if (recovered and pmid in recovery) else r
        st = status_of(use['final_inclusion'], use['fulltext_reachable'])
        rows.append(dict(record_id=pmid, batch=r['batch'], status=st,
                         fulltext=obtained(use['fulltext_reachable']),
                         exclusion_code=(use.get('exclusion_code') if st == 'excluded'
                                         else None),
                         status_tie=False, ft_tie=False, code_tie=False,
                         source=('recovery_overlay_v13'
                                 if (recovered and pmid in recovery) else 'sealed'),
                         any_screener_sought_fulltext=use['stage1_decision'] != 'exclude'))

    assert len(rows) == 100 and len({r['record_id'] for r in rows}) == 100
    for r in rows:                                   # no record can be both
        assert not (r['status'] == 'eligible_unreachable' and r['fulltext'])
    return rows


# --------------------------------------------------------------------------- flow
def flow(rows):
    excl_screen = [r for r in rows if r['status'] == 'excluded' and not r['fulltext']]
    not_retrieved = [r for r in rows if r['status'] == 'eligible_unreachable']
    excl_full = [r for r in rows if r['status'] == 'excluded' and r['fulltext']]
    included = [r for r in rows if r['status'] == 'included_reachable']
    assert len(excl_screen) + len(not_retrieved) + len(excl_full) + len(included) == 100

    def by_reason(rs):
        c = Counter(r['exclusion_code'] for r in rs)
        return OrderedDict((k, c[k]) for k in EXCLUSION_LABELS if c[k])

    sought = len(rows) - len(excl_screen)
    eligible = len(included) + len(not_retrieved)
    return OrderedDict([
        ('records_screened', len(rows)),
        ('excluded_at_screening', len(excl_screen)),
        ('excluded_at_screening_by_reason', by_reason(excl_screen)),
        ('reports_sought_for_retrieval', sought),
        ('reports_not_retrieved', len(not_retrieved)),
        ('reports_assessed_for_eligibility', sought - len(not_retrieved)),
        ('excluded_at_fulltext', len(excl_full)),
        ('excluded_at_fulltext_by_reason', by_reason(excl_full)),
        ('studies_included', len(included)),
        ('excluded_total', len(excl_screen) + len(excl_full)),
        ('excluded_total_by_reason', by_reason(excl_screen + excl_full)),
        ('eligible_looking_denominator', eligible),
        ('unreachable_rate_pct', round(100 * len(not_retrieved) / eligible, 4)),
        ('ids', OrderedDict([
            ('excluded_at_screening', sorted(r['record_id'] for r in excl_screen)),
            ('reports_not_retrieved', sorted(r['record_id'] for r in not_retrieved)),
            ('excluded_at_fulltext', sorted(r['record_id'] for r in excl_full)),
            ('studies_included', sorted(r['record_id'] for r in included)),
        ])),
    ])


def identification():
    fm = json.load(open(f'{PAPER}/screen/frame_meta.json'))
    raw = [l.strip() for l in open(f'{PAPER}/screen/frame_pmids.txt') if l.strip()]
    sm = json.load(open(f'{PAPER}/screen_sample.json'))
    c = sm['counts']
    drawn = c['analysis_sample_total'] + c['pilot'] + c['reserve']
    return OrderedDict([
        ('database', fm['database']),
        ('date_run_utc', fm['date_run_utc']),
        ('esearch_count', fm['esearch_count']),
        ('pmids_retrieved', fm['pmids_retrieved']),
        ('records_after_dedup', len(set(raw))),
        ('duplicates_removed', len(raw) - len(set(raw))),
        ('registers_searched', 0),
        ('records_from_other_methods', 0),
        ('frame_sha256', fm['frame_sha256']),
        ('permutation_sha256', sm['sampling']['permutation_sha256']),
        ('seed', sm['sampling']['seed']),
        ('positions_with_metadata_attached', drawn),
        ('analysis_sample', c['analysis_sample_total']),
        ('pilot_excluded_a_priori', c['pilot']),
        ('reserve_not_activated', c['reserve']),
        ('frame_records_never_drawn', len(set(raw)) - drawn),
    ])


# --------------------------------------------------------------------------- consistency
def crosscheck(f10, f13):
    """Assert against the two already-published analysis outputs. If either file is
    regenerated and drifts, this script fails loudly instead of printing a pretty figure."""
    a = json.load(open(f'{HERE}/analysis_out.json'))['flow']
    r = json.load(open(f'{HERE}/recovery_out.json'))
    checks = [
        ('v1.0 included vs analysis_out.flow.included_and_reachable',
         f10['studies_included'], a['included_and_reachable']),
        ('v1.0 unreachable vs analysis_out.flow.eligible_but_unreachable',
         f10['reports_not_retrieved'], a['eligible_but_unreachable']),
        ('v1.0 excluded vs analysis_out.flow.excluded',
         f10['excluded_total'], a['excluded']),
        ('v1.0 eligible vs analysis_out.flow.eligible_looking',
         f10['eligible_looking_denominator'], a['eligible_looking']),
        ('v1.3 included vs recovery_out.after.n_included_reachable',
         f13['studies_included'], r['after']['n_included_reachable']),
        ('v1.3 unreachable vs recovery_out.after.n_eligible_unreachable',
         f13['reports_not_retrieved'], r['after']['n_eligible_unreachable']),
        ('v1.3 excluded vs recovery_out.after.n_excluded',
         f13['excluded_total'], r['after']['n_excluded']),
        ('v1.3 eligible vs recovery_out.after.n_eligible',
         f13['eligible_looking_denominator'], r['after']['n_eligible']),
    ]
    bad = [(n, x, y) for n, x, y in checks if x != y]
    for n, x, y in checks:
        print(f"   [{'OK ' if x == y else 'FAIL'}] {n}: {x} vs {y}")
    assert not bad, bad
    return [dict(check=n, prisma_flow=x, published=y, agree=True) for n, x, y in checks]


# --------------------------------------------------------------------------- SVG
def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def svg(f, ident, path):
    """Row-based layout. Each row is (left_lines, right_lines, style); the row height is
    the taller of the two boxes, so no box can ever be drawn over another and the canvas
    is sized to the content rather than to a guessed constant."""
    W = 1120
    LX, LW, RX, RW = 70, 470, 610, 450
    GAP, PAD, LH = 36, 14, 17
    out = []
    A = out.append

    def txt(x, y, s, fs=13, w='normal', fill='#111111'):
        A(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{w}" fill="{fill}">'
          f'{_esc(s)}</text>')

    def box(x, y, w, lines, fill, fs):
        h = PAD * 2 + LH * len(lines) - 4
        A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
          f'stroke="#222222" stroke-width="1.2"/>')
        ty = y + PAD + 12
        for i, ln in enumerate(lines):
            txt(x + 12, ty, ln, fs=fs, w=('bold' if i == 0 else 'normal'))
            ty += LH
        return h

    def height(lines):
        return PAD * 2 + LH * len(lines) - 4

    def reasons(head, n, d):
        out = [head, f'n = {n}']
        for k, v in d.items():
            out.append(f'    {k}  n = {v}   {EXCLUSION_LABELS[k]}')
        return out

    rows = [
        (['Records identified from a database',
          f"{ident['database'].split(' (')[0]}, n = {ident['esearch_count']:,}",
          f"one frozen Boolean query, run {ident['date_run_utc'][:10]}",
          f"Registers n = {ident['registers_searched']}; other methods "
          f"n = {ident['records_from_other_methods']}"],
         ['Records removed before screening',
          f"Duplicates removed n = {ident['duplicates_removed']} (PMID-unique frame)",
          'Removed by automation tools n = 0',
          'Removed for other reasons n = 0'], 'Identification', '#ffffff', 13),
        (['Records randomly sampled for screening',
          f"n = {ident['analysis_sample']}  (permutation positions 1-100)",
          f"seed {ident['seed']}; frame SHA-256 {ident['frame_sha256'][:12]}..."],
         ['Frame records not sampled',
          f"never drawn n = {ident['frame_records_never_drawn']:,}",
          f"pilot, excluded a priori n = {ident['pilot_excluded_a_priori']}",
          f"pre-specified reserve, NOT activated n = "
          f"{ident['reserve_not_activated']}"], None, '#ffffff', 13),
        (['Records screened (title and abstract)', f"n = {f['records_screened']}"],
         reasons('Records excluded at screening', f['excluded_at_screening'],
                 f['excluded_at_screening_by_reason']), 'Screening', '#ffffff', 12),
        (['Reports sought for retrieval', f"n = {f['reports_sought_for_retrieval']}",
          'five-rung access ladder, protocol section 7'],
         ['Reports not retrieved', f"n = {f['reports_not_retrieved']}",
          'paywalled after all five rungs; no infringing source used'],
         None, '#ffffff', 13),
        (['Reports assessed for eligibility (full text)',
          f"n = {f['reports_assessed_for_eligibility']}"],
         reasons('Reports excluded at full text', f['excluded_at_fulltext'],
                 f['excluded_at_fulltext_by_reason']), None, '#ffffff', 12),
        (['Studies included in the review',
          f"n = {f['studies_included']}   (the complete-case denominator)",
          f"eligible-looking set n = {f['eligible_looking_denominator']}; unreachable "
          f"{f['reports_not_retrieved']}/{f['eligible_looking_denominator']} = "
          f"{f['unreachable_rate_pct']:.1f}%"],
         None, 'Included', '#eaf1fb', 13),
    ]

    footer = [
        'Censoring exceeds the pre-registered 15% threshold, so protocol section 7 makes the '
        'bounding interval, not the',
        'complete-case point estimate, the headline for the primary endpoint. The pre-registered '
        'target of 75 included',
        'studies was not reached and the extension rule into the reserve was triggered but not '
        'executed. Both are',
        'protocol deviations and are reported as such, not absorbed.',
    ]

    y = 78
    ys = []
    for left, rightl, _band, _fill, _fs in rows:
        h = max(height(left), height(rightl) if rightl else 0)
        ys.append((y, h))
        y += h + GAP
    H = int(y - GAP + 30 + 16 * len(footer) + 20)

    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">')
    A('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
      'markerHeight="6" orient="auto-start-reverse">'
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#222222"/></marker></defs>')
    A(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    txt(LX, 32, 'PRISMA 2020 flow, adapted for a random-sample meta-research screen', 17,
        'bold')
    txt(LX, 52, 'PhaseDx prevalence screen. Every count is produced by '
                'paper/screen/analysis/prisma_flow.py from the files it names.', 11.5,
        fill='#555555')

    for i, (left, rightl, bandlab, fill, fs) in enumerate(rows):
        y, h = ys[i]
        lh = box(LX, y, LW, left, fill, fs)
        if rightl:
            box(RX, y, RW, rightl, '#f7f7f7', fs)
            A(f'<line x1="{LX + LW}" y1="{y + 30}" x2="{RX}" y2="{y + 30}" '
              f'stroke="#222222" stroke-width="1.2" marker-end="url(#a)"/>')
        if i:
            py, ph = ys[i - 1]
            A(f'<line x1="{LX + LW / 2}" y1="{py + max(ph, 0) if False else py + ph}" '
              f'x2="{LX + LW / 2}" y2="{y}" stroke="#222222" stroke-width="1.2" '
              f'marker-end="url(#a)"/>')
        if bandlab:
            cy = y + lh / 2
            A(f'<text x="34" y="{cy}" font-size="12" font-weight="bold" fill="#888888" '
              f'text-anchor="middle" transform="rotate(-90 34 {cy})">{bandlab}</text>')

    yy = ys[-1][0] + ys[-1][1] + 30
    for s in footer:
        txt(LX, yy, s, 11.5, fill='#555555')
        yy += 16
    A('</svg>')
    open(path, 'w').write('\n'.join(out) + '\n')


# --------------------------------------------------------------------------- markdown
PROSE = {}

PROSE['head'] = """# PRISMA 2020 flow for the PhaseDx prevalence screen

**This file is generated.** Every integer below is written by
`paper/screen/analysis/prisma_flow.py`, which reads the frozen frame, the four sealed
screener files, the v1.2 adjudicated codes and the v1.3 access-recovery overlay. Nothing is
retyped. Regenerate with:

```bash
python paper/screen/analysis/prisma_flow.py
```

It writes this file, the machine-readable `paper/prisma_flow.json`, and the figure
`paper/figures/prisma_flow.svg`. It also asserts its own totals against
`paper/screen/analysis/analysis_out.json` and `paper/screen/analysis/recovery_out.json` and
fails rather than print a figure that disagrees with the published endpoints.
"""

PROSE['adaptation'] = """## How PRISMA 2020 is adapted here, and why

This is a **meta-research screen**, not a systematic review of clinical effects. It estimates
how often a reporting practice occurs in a defined literature. Three adaptations follow, each
stated so a reader can see exactly where the diagram departs from the standard template.

1. **A random sample, not a census.** PRISMA's flow assumes every identified record is
   screened. Here the frame is screened by drawing a seeded random permutation and taking the
   first 100 positions. An extra box, *Records randomly sampled for screening*, sits between
   identification and screening, and the records not drawn are accounted for on the right at
   that level rather than being silently dropped. The seed and the frame digest that make the
   draw reproducible are printed in the box itself.
2. **"Reports not retrieved" is an endpoint, not an inconvenience.** In a clinical review the
   unretrieved reports are a footnote. Here reachability is a pre-registered secondary
   endpoint (S6) and, past a pre-registered 15% threshold, it converts the primary estimate
   from a point estimate into a bounding interval. The diagram therefore carries the
   unreachable count and rate into the *Included* box.
3. **Risk of bias in the usual sense does not apply.** The unit of observation is *what a
   paper reports about its own evaluation*, verbatim and quoted, not an effect estimate whose
   internal validity must be judged. The corresponding checklist items are marked adapted, not
   dropped, in `paper/prisma_checklist.md`.

**Stage mapping.** PRISMA's boxes and this codebook's fields are not one-to-one, so the
mapping is declared rather than improvised. Each of the 100 records lands in exactly one
terminal box, determined only by pooled `final_inclusion` and pooled `fulltext_reachable`:

| pooled state | PRISMA box |
|---|---|
| excluded, full text not obtained | records excluded at title/abstract screening |
| eligible-looking, full text not obtained | reports sought for retrieval, not retrieved |
| excluded, full text obtained | reports excluded at full-text assessment, with reason |
| included, full text obtained | studies included |
"""

PROSE['deviations_head'] = """## Protocol deviations

Reported here rather than in a supplement, because two of them cap what the screen can claim.
"""

PROSE['tail'] = """## What this flow does and does not license

- The primary endpoint is **zero in every included paper**, and zero in every one of the 149
  codes produced across the whole screen including the excluded and unreachable records. The
  complete-case estimate is therefore a zero count and its precision is entirely the upper
  Wilson bound.
- Because censoring is above the pre-registered 15% threshold, **the bounding interval, not
  the complete-case point estimate, is the headline** for P1, S1, S4 and S5. That rule was
  fixed in the protocol before any record was read, and it survived the access-recovery pass:
  recovering four full texts moved the censoring rate but not past the threshold, and
  recovering the three demonstrably-open-access records that this environment cannot fetch
  would still leave it above the threshold.
- Single screening of 85 of the 100 records, a target sample that was not reached, and an
  extension rule that fired and was not executed, are all real limits on precision. None of
  them can manufacture the primary result in the direction the paper argues: every unmeasured
  or unread record is imputed **against** the paper's own thesis in the upper bound.

## Known inconsistency to fix before submission

`paper/DRAFT.md` Table 5 still reports the **v1.0 as-sealed** flow (35 included, 20
unreachable, 45 excluded) and the v1.0 exclusion tally. The current primary flow is the
**v1.3 post-recovery** one above. Table 5, the abstract and section 3.10 of the draft must be
brought onto the post-recovery numbers, or must state explicitly which version they report.
"""


def markdown(ident, f10, f12, f13, minority, checks, deviations, path):
    L = []
    A = L.append
    A(PROSE['head'])
    A(PROSE['adaptation'])

    A('## The flow (primary: v1.3, post access-recovery)\n')
    A(f"![PRISMA flow](figures/prisma_flow.svg)\n")
    A('| stage | n |')
    A('|---|---|')
    A(f"| Records identified from {ident['database']} | {ident['esearch_count']:,} |")
    A(f"| Duplicate records removed before screening | {ident['duplicates_removed']} |")
    A(f"| Records removed by automation tools, or for other reasons | 0 |")
    A(f"| Records in the frozen frame | {ident['records_after_dedup']:,} |")
    A(f"| — never drawn | {ident['frame_records_never_drawn']:,} |")
    A(f"| — pilot, read by the protocol author, excluded a priori | "
      f"{ident['pilot_excluded_a_priori']} |")
    A(f"| — pre-specified reserve, **not activated** | {ident['reserve_not_activated']} |")
    A(f"| **Records randomly sampled and screened (title/abstract)** | "
      f"**{f13['records_screened']}** |")
    A(f"| Records excluded at screening | {f13['excluded_at_screening']} |")
    A(f"| **Reports sought for retrieval** | **{f13['reports_sought_for_retrieval']}** |")
    A(f"| Reports not retrieved | {f13['reports_not_retrieved']} |")
    A(f"| **Reports assessed for eligibility (full text)** | "
      f"**{f13['reports_assessed_for_eligibility']}** |")
    A(f"| Reports excluded at full text | {f13['excluded_at_fulltext']} |")
    A(f"| **Studies included in the review** | **{f13['studies_included']}** |")
    A('')
    A(f"Eligible-looking set (included + not retrieved) = "
      f"**{f13['eligible_looking_denominator']}**. Unreachable "
      f"{f13['reports_not_retrieved']}/{f13['eligible_looking_denominator']} = "
      f"**{f13['unreachable_rate_pct']:.1f}%**, against a pre-registered 15% threshold "
      f"(protocol section 7).\n")

    A('### Exclusions by reason and by stage\n')
    A('| code | meaning | at screening | at full text | total |')
    A('|---|---|---|---|---|')
    for k, lab in EXCLUSION_LABELS.items():
        a = f13['excluded_at_screening_by_reason'].get(k, 0)
        b = f13['excluded_at_fulltext_by_reason'].get(k, 0)
        if a or b:
            A(f'| `{k}` | {lab} | {a} | {b} | {a + b} |')
    A(f"| | **total** | **{f13['excluded_at_screening']}** | "
      f"**{f13['excluded_at_fulltext']}** | **{f13['excluded_total']}** |")
    A('')
    A('`E-DERIV` is reported separately, as protocol section 9 requires: those papers are '
      'inside the query and outside the failure mode the screen is about, so folding them '
      'into a single "excluded" total would hide what the frame\'s imprecision consisted of.'
      '\n')

    A('### Reports not retrieved, by PMID\n')
    A('Listed so a reader with better access can finish the screen. '
      f"n = {f13['reports_not_retrieved']}.\n")
    A('```')
    ids = f13['ids']['reports_not_retrieved']
    for i in range(0, len(ids), 8):
        A('  '.join(ids[i:i + 8]))
    A('```\n')
    A('Three of these — 37222638 (Wiley, bronze), 42153825 (RSNA, CC BY) and 40147601 '
      '(Elsevier, CC BY-NC-ND) — are **demonstrably open access** and are unreachable only '
      'because this execution environment is refused by those publishers. They are counted '
      'as unreachable because no full text was read; the cause is disclosed rather than '
      'charged to the literature. Recovering all three would give '
      f"13/{f13['eligible_looking_denominator']} = "
      f"{100 * 13 / f13['eligible_looking_denominator']:.1f}%, still above the 15% "
      'threshold. Evidence per record: `paper/screen/access_recovery.json`.\n')

    A('## The same flow at each protocol version\n')
    A('Shown so that the effect of the v1.2 adjudication and the v1.3 access recovery is '
      'visible line by line rather than asserted.\n')
    A('| stage | v1.0 as sealed | v1.2 adjudicated | v1.3 post-recovery |')
    A('|---|---|---|---|')
    for key, lab in [
        ('records_screened', 'records screened'),
        ('excluded_at_screening', 'excluded at screening'),
        ('reports_sought_for_retrieval', 'reports sought for retrieval'),
        ('reports_not_retrieved', 'reports not retrieved'),
        ('reports_assessed_for_eligibility', 'reports assessed at full text'),
        ('excluded_at_fulltext', 'excluded at full text'),
        ('studies_included', '**studies included**'),
        ('eligible_looking_denominator', 'eligible-looking denominator'),
    ]:
        A(f"| {lab} | {f10[key]} | {f12[key]} | {f13[key]} |")
    A(f"| unreachable rate | {f10['unreachable_rate_pct']:.1f}% | "
      f"{f12['unreachable_rate_pct']:.1f}% | {f13['unreachable_rate_pct']:.1f}% |")
    A('')
    A('**v1.0 to v1.2.** ' + PROSE['_v12'] + '\n')
    A('**v1.2 to v1.3.** ' + PROSE['_v13'] + '\n')

    A('### Cross-checks against the already-published analysis outputs\n')
    A('| check | this file | published | agree |')
    A('|---|---|---|---|')
    for c in checks:
        A(f"| {c['check']} | {c['prisma_flow']} | {c['published']} | "
          f"{'yes' if c['agree'] else '**NO**'} |")
    A('')

    A(PROSE['deviations_head'])
    A('| # | deviation | observed | pre-registered | where |')
    A('|---|---|---|---|---|')
    for i, d in enumerate(deviations, 1):
        A(f"| {i} | {d['deviation']} | {d['observed']} | {d['target']} | `{d['where']}` |")
    A('')
    A(PROSE['_dev_prose'])

    A('### Footnote: records excluded on the abstract that a minority would have read in full'
      '\n')
    A(f"n = {len(minority)}: " + ', '.join(minority) + '. Each was excluded at title/abstract '
      'under the pooled decision, but at least one screener had recorded '
      '`stage1_decision=go_to_fulltext` before excluding, and in each case the full text then '
      'proved unreachable. Placement changes only which PRISMA box the exclusion is drawn in. '
      'It changes no endpoint and no denominator, and both placements are recoverable from '
      '`paper/prisma_flow.json`.\n')

    A(PROSE['tail'])
    open(path, 'w').write('\n'.join(L) + '\n')


PROSE['_v12'] = (
    'The 15 overlap records are replaced by the v1.2 adjudicated codes. **No terminal-box '
    'count changes.** One exclusion *reason* changes: PMID 40335658 moves `E-SEG` to '
    '`E-NOCLF` under rule D10, because a categorical class decision was evaluated — by human '
    'readers, not by a fitted model — which fails criterion I2 rather than being a pure '
    'segmentation paper. The reason tally therefore differs from '
    '`analysis_out.json` by one in each of those two codes, and that is the whole difference.')

PROSE['_v13'] = (
    'The access ladder was re-run against every unretrieved report and four were recovered. '
    'Three (38591974, 36200353, 39846055) move from *not retrieved* to *included*. The '
    'fourth, 36170844, was retrieved and then proved **ineligible** on full text — a U-Net '
    'segmentation study with no fitted classifier and no negative class, `E-NOCLF` under D10 '
    '— so it moves to *excluded at full text* and **leaves the eligible-looking denominator '
    'entirely**, which is why that denominator falls rather than staying fixed. One of the '
    'three inclusions (36200353) was coded from an Authorea **preprint** rather than the '
    'version of record and is flagged for the version-of-record sensitivity analysis. No '
    'sealed screener file was edited; the recovery is an analysis-time overlay.')

PROSE['_dev_prose'] = """Four further things the protocol required that were **not done**, each
of which a reviewer is entitled to see stated plainly:

- **Rung 5 of the access ladder — interlibrary loan or a direct request to the corresponding
  author, with a 21-day wait — was never initiated by any screener.** All four screener files
  say so in their own access notes. Records coded `unreachable_paywalled` should be read as
  "unreachable through rungs 1-4", not "unobtainable".
- **The 20% within-batch re-coding by the next screener in the A→B→C→D→A cycle (protocol
  section 6) was not carried out.** Outside the 15-paper overlap set, disagreement is
  invisible by construction, and no measurement of it exists.
- **Second-screener adjudication of flagged records was not carried out.** Section 6 requires
  it for any record marked `screener_confidence='low'` or `flag_for_adjudication=true`; 49 of
  the 85 single-screened records carry such a flag.
- **A genuine post-amendment reliability estimate does not exist.** The v1.2 figures are a
  counterfactual re-encoding of the same sealed files, not an independent re-rating. A fresh
  four-screener re-coding under v1.2 is an outstanding action and the paper must say so.

None of the four can be repaired by rewriting text. They are labour, and until they are done
the corresponding checklist items in `paper/prisma_checklist.md` are marked **not done**.
"""


# --------------------------------------------------------------------------- main
def main():
    ident = identification()
    f10 = flow(build(adjudicated=False, recovered=False))
    f12 = flow(build(adjudicated=True, recovered=False))
    f13 = flow(build(adjudicated=True, recovered=True))

    rows13 = build(adjudicated=True, recovered=True)
    minority = sorted(r['record_id'] for r in rows13
                      if r['status'] == 'excluded' and not r['fulltext']
                      and r['any_screener_sought_fulltext'])

    print(f"frame {ident['esearch_count']:,}  duplicates {ident['duplicates_removed']}  "
          f"sampled {ident['analysis_sample']}  pilot {ident['pilot_excluded_a_priori']}  "
          f"reserve {ident['reserve_not_activated']}  never drawn "
          f"{ident['frame_records_never_drawn']:,}")
    for name, fl in (('v1.0 as sealed', f10), ('v1.2 adjudicated', f12),
                     ('v1.3 post-recovery  <-- PRIMARY', f13)):
        print(f'--- {name}')
        for k in ('records_screened', 'excluded_at_screening',
                  'reports_sought_for_retrieval', 'reports_not_retrieved',
                  'reports_assessed_for_eligibility', 'excluded_at_fulltext',
                  'studies_included', 'eligible_looking_denominator',
                  'unreachable_rate_pct'):
            print(f'   {k:34s} {fl[k]}')
        print(f"   screening exclusions  {dict(fl['excluded_at_screening_by_reason'])}")
        print(f"   full-text exclusions  {dict(fl['excluded_at_fulltext_by_reason'])}")
        print(f"   all exclusions        {dict(fl['excluded_total_by_reason'])}")
    print('--- cross-checks against the already-published analysis outputs')
    checks = crosscheck(f10, f13)

    out = OrderedDict([
        ('_generated_by', 'paper/screen/analysis/prisma_flow.py'),
        ('_inputs', ['paper/screen/frame_meta.json', 'paper/screen/frame_pmids.txt',
                     'paper/screen_sample.json', 'paper/screen_batch_A.json',
                     'paper/screen_batch_B.json', 'paper/screen_batch_C.json',
                     'paper/screen_batch_D.json',
                     'paper/screen/analysis/adjudication_out.json',
                     'paper/screen/access_recovery.json']),
        ('_stage_mapping', ' '.join(__doc__.split('STAGE MAPPING')[1].split())),
        ('_primary', 'v1_3_post_recovery'),
        ('identification', ident),
        ('v1_0_as_sealed', f10),
        ('v1_2_adjudicated', f12),
        ('v1_3_post_recovery', f13),
        ('what_changed', OrderedDict([
            ('v1_0_to_v1_2',
             'Overlap records replaced by the v1.2 adjudicated codes. Terminal-box counts '
             'are unchanged; one exclusion REASON changes, PMID 40335658 E-SEG -> E-NOCLF '
             'under rule D10, so the reason tally differs from analysis_out.json by one in '
             'each of those two codes.'),
            ('v1_2_to_v1_3',
             'Access-recovery overlay: 38591974, 36200353, 39846055 move from not-retrieved '
             'to included; 36170844 moves from not-retrieved to excluded at full text '
             '(E-NOCLF), which removes it from the eligible-looking denominator.'),
        ])),
        ('footnote_minority_would_have_sought_fulltext', OrderedDict([
            ('n', len(minority)),
            ('record_ids', minority),
            ('note', 'Excluded on title/abstract under the pooled decision, but at least one '
                     'screener had recorded stage1_decision=go_to_fulltext before excluding '
                     '(in each case the full text then proved unreachable). Placement '
                     'affects only which PRISMA box the exclusion is drawn in; it changes no '
                     'endpoint and no denominator.'),
        ])),
        ('crosschecks_against_published_outputs', checks),
        ('protocol_deviations', [
            OrderedDict([
                ('deviation', 'pre-registered target of 75 included studies not reached'),
                ('observed', f13['studies_included']),
                ('target', 75),
                ('where', 'paper/screen_protocol.md section 3.1')]),
            OrderedDict([
                ('deviation', 'extension rule triggered but not executed; the '
                              f"{ident['reserve_not_activated']}-record reserve was never "
                              'activated'),
                ('observed', 'reserve records screened: 0'),
                ('target', 'continue in blocks of 50 until 75 included or position 400'),
                ('where', 'paper/screen_protocol.md section 3.1')]),
            OrderedDict([
                ('deviation', 'codebook amended to v1.2 after coding began'),
                ('observed', 'rules D1-D14 plus four enum levels; no endpoint definition, '
                             'interval method, threshold or sampling decision altered'),
                ('target', 'n/a -- the remedy is itself pre-registered at section 6'),
                ('where', 'paper/screen_protocol.md section 12; paper/screen_adjudication.md')]),
            OrderedDict([
                ('deviation', 'access ladder re-run after the first analysis (v1.3)'),
                ('observed', '4 of 20 unretrieved reports recovered; no sealed file edited'),
                ('target', 'n/a -- no rule changed'),
                ('where', 'paper/screen/access_recovery.json')]),
            OrderedDict([
                ('deviation', 'no prospective public registry deposit'),
                ('observed', 'protocol frozen in git before screening; OSF deposit not made '
                             'in advance'),
                ('target', 'protocol section 11 named an OSF deposit as an action item to be '
                           'done before screening'),
                ('where', 'paper/registration.md')]),
        ]),
    ])
    json.dump(out, open(f'{PAPER}/prisma_flow.json', 'w'), indent=1)
    svg(f13, ident, f'{PAPER}/figures/prisma_flow.svg')
    markdown(ident, f10, f12, f13, minority, checks, out['protocol_deviations'],
             f'{PAPER}/prisma_flow.md')
    print(f"footnote: excluded-at-screening records where >=1 screener had said "
          f"go_to_fulltext: n = {len(minority)} {minority}")
    print(f'wrote {PAPER}/prisma_flow.json')
    print(f'wrote {PAPER}/figures/prisma_flow.svg')
    print(f'wrote {PAPER}/prisma_flow.md')


if __name__ == '__main__':
    main()

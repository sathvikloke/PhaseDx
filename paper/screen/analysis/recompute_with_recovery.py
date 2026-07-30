#!/usr/bin/env python3
"""Recompute the flow, S6 and the P1 bounding interval after the access-recovery pass.

The four sealed batch files are read-only here: the recovery records in
paper/screen/access_recovery.json are applied as an OVERLAY, keyed on record_id, and
every number is printed BEFORE and AFTER so the effect of the recovery is auditable
line by line.

Denominator and status logic are copied verbatim from pool_and_agree.py so the two
scripts cannot drift.

    python paper/screen/analysis/recompute_with_recovery.py
"""
import json, math, os
from collections import Counter

HERE  = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))

NA          = 'NOT_ASSESSABLE'
P1_KEYS     = ('constant_or_prevalence', 'positional',
               'acquisition_metadata', 'permuted_or_shuffled_label')
S1_KEYS     = P1_KEYS + ('clinical_or_demographic_only', 'other_non_imaging')
UNREACHABLE = ('unreachable_paywalled', 'unreachable_not_found')
SCREENERS   = ('S1', 'S2', 'S3', 'S4')


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _flag(r, keys):
    vs = [r['trivial_baseline'].get(k) for k in keys]
    if any(v is True for v in vs):
        return True
    if all(v is False for v in vs):
        return False
    return NA


def p1(r): return _flag(r, P1_KEYS)
def s1(r): return _flag(r, S1_KEYS)
def got_fulltext(r): return r['fulltext_reachable'] not in UNREACHABLE


def status(r):
    if r['final_inclusion'] == 'excluded':
        return 'excluded'
    return 'included_reachable' if got_fulltext(r) else 'eligible_unreachable'


# ------------------------------------------------------------------ load, read-only
PS = {}
for b in 'ABCD':
    d = json.load(open(f'{PAPER}/screen_batch_{b}.json'))
    PS[d['screener_id']] = d

overlap, unique = {}, {}
for s in SCREENERS:
    for r in PS[s]['records']:
        if r['batch'] == 'overlap':
            overlap.setdefault(r['record_id'], {})[s] = r
        else:
            unique[r['record_id']] = r
assert len(overlap) == 15 and len(unique) == 85

REC = json.load(open(f'{PAPER}/screen/access_recovery.json'))
recovery = {r['record_id']: r for r in REC['records']}
assert set(recovery) <= set(unique), \
    'recovery records must all be non-overlap; an overlap record would need all four screeners re-coded'


def majority_status(codes):
    """Overlap papers: majority of the four codes, ties broken AGAINST our own thesis
    (i.e. toward eligible_unreachable, the state that widens the bound)."""
    c = Counter(codes)
    top = max(c.values())
    tied = sorted(k for k, v in c.items() if v == top)
    for pref in ('eligible_unreachable', 'included_reachable', 'excluded'):
        if pref in tied:
            return pref, len(tied) > 1
    return tied[0], len(tied) > 1


def build(apply_recovery):
    rows = []
    for pmid, per in overlap.items():
        st, _ = majority_status([status(per[s]) for s in SCREENERS])
        codes = [per[s] for s in SCREENERS]
        vals = [p1(r) for r in codes]
        P1 = True if any(v is True for v in vals) else (
             False if all(v is False for v in vals) else NA)
        vals = [s1(r) for r in codes]
        S1 = True if any(v is True for v in vals) else (
             False if all(v is False for v in vals) else NA)
        rows.append(dict(record_id=pmid, status=st, p1=P1, s1=S1))
    for pmid, r in unique.items():
        src = recovery[pmid] if (apply_recovery and pmid in recovery) else r
        rows.append(dict(record_id=pmid, status=status(src), p1=p1(src), s1=s1(src)))
    assert len(rows) == 100
    return rows


def endpoints(rows):
    CC   = [r for r in rows if r['status'] == 'included_reachable']
    UNRE = [r for r in rows if r['status'] == 'eligible_unreachable']
    EXCL = [r for r in rows if r['status'] == 'excluded']
    ELIG = CC + UNRE
    k_p1 = sum(1 for r in CC if r['p1'] is True)
    k_s1 = sum(1 for r in CC if r['s1'] is True)

    def ep(k, n):
        p, lo, hi = wilson(k, n)
        return dict(k=k, n=n, pct=100 * p, ci=[100 * lo, 100 * hi])

    return dict(
        n_screened=len(rows), n_included_reachable=len(CC),
        n_eligible_unreachable=len(UNRE), n_excluded=len(EXCL), n_eligible=len(ELIG),
        S6_unreachable=ep(len(UNRE), len(ELIG)),
        P1_complete_case=ep(k_p1, len(CC)),
        P1_lower=ep(k_p1, len(ELIG)),
        P1_upper=ep(k_p1 + len(UNRE), len(ELIG)),
        S1_complete_case=ep(k_s1, len(CC)),
        S1_upper=ep(k_s1 + len(UNRE), len(ELIG)),
        threshold_breached=len(UNRE) / len(ELIG) > 0.15,
    )


def fmt(e):
    return (f"  eligible-looking          {e['n_eligible']}\n"
            f"  included + reachable      {e['n_included_reachable']}\n"
            f"  eligible but unreachable  {e['n_eligible_unreachable']}\n"
            f"  excluded                  {e['n_excluded']}\n"
            f"  S6 unreachable            {e['S6_unreachable']['k']}/{e['S6_unreachable']['n']} = "
            f"{e['S6_unreachable']['pct']:.1f}%  [{e['S6_unreachable']['ci'][0]:.1f}%, {e['S6_unreachable']['ci'][1]:.1f}%]\n"
            f"  P1 complete-case          {e['P1_complete_case']['k']}/{e['P1_complete_case']['n']} = "
            f"{e['P1_complete_case']['pct']:.1f}%  [{e['P1_complete_case']['ci'][0]:.1f}%, {e['P1_complete_case']['ci'][1]:.1f}%]\n"
            f"  P1 BOUNDING INTERVAL      [{e['P1_lower']['pct']:.1f}%, {e['P1_upper']['pct']:.1f}%]\n"
            f"    lower  {e['P1_lower']['k']}/{e['P1_lower']['n']}  [{e['P1_lower']['ci'][0]:.1f}%, {e['P1_lower']['ci'][1]:.1f}%]\n"
            f"    upper  {e['P1_upper']['k']}/{e['P1_upper']['n']}  [{e['P1_upper']['ci'][0]:.1f}%, {e['P1_upper']['ci'][1]:.1f}%]\n"
            f"  >15% threshold breached   {e['threshold_breached']}  "
            f"({'bound REPLACES the point estimate as the headline' if e['threshold_breached'] else 'point estimate may head the paper'})\n")


before = endpoints(build(False))
after  = endpoints(build(True))

print('=' * 78)
print('BEFORE  (four sealed batch files as submitted)')
print('=' * 78)
print(fmt(before))
print('=' * 78)
print('AFTER   (+ paper/screen/access_recovery.json overlay)')
print('=' * 78)
print(fmt(after))
print('=' * 78)
print('CHANGE')
print('=' * 78)
print(f"  full texts recovered      {len(recovery)}")
for pmid, r in recovery.items():
    print(f"    {pmid} pos {r['permutation_position']:>3}  {r['_changes_vs_sealed']['status_for_analysis'][0]}"
          f" -> {r['_changes_vs_sealed']['status_for_analysis'][1]}   ({r['access_rung_succeeded'].split(' -- ')[0]})")
print(f"  unreachable               {before['n_eligible_unreachable']} -> {after['n_eligible_unreachable']}")
print(f"  eligible denominator      {before['n_eligible']} -> {after['n_eligible']}")
print(f"  unreachable rate          {before['S6_unreachable']['pct']:.1f}% -> {after['S6_unreachable']['pct']:.1f}%")
print(f"  P1 bound                  [{before['P1_lower']['pct']:.1f}%, {before['P1_upper']['pct']:.1f}%]"
      f" -> [{after['P1_lower']['pct']:.1f}%, {after['P1_upper']['pct']:.1f}%]")
print(f"  bound width               {before['P1_upper']['pct'] - before['P1_lower']['pct']:.1f} pp"
      f" -> {after['P1_upper']['pct'] - after['P1_lower']['pct']:.1f} pp")
print(f"  P1 numerator (any measured zero-image baseline, complete case): "
      f"{before['P1_complete_case']['k']} -> {after['P1_complete_case']['k']}")

out = dict(_generated_by='recompute_with_recovery.py', before=before, after=after,
           recovered=[dict(record_id=k, position=v['permutation_position'],
                           rung=v['access_rung_succeeded'],
                           version=v['fulltext_version_used'],
                           status_change=v['_changes_vs_sealed']['status_for_analysis'])
                      for k, v in recovery.items()],
           still_unreachable=[r['record_id'] for r in REC['still_unreachable']])
json.dump(out, open(f'{HERE}/recovery_out.json', 'w'), indent=1)
print(f"\nwrote {HERE}/recovery_out.json")

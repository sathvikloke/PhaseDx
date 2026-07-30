#!/usr/bin/env python3
"""
PhaseDx prevalence screen -- pooling + agreement.
Pre-registered in paper/screen_protocol.md v1.0/v1.1 and paper/screen_frame.json v1.0.

Inputs : paper/screen_batch_{A,B,C,D}.json  (four sealed, independent screener files)
Output : paper/screen/analysis/analysis_out.json  + a console report

Computes
  (1) agreement on the 15-paper overlap set: Fleiss' kappa (primary, 4 raters), the six
      pairwise Cohen's kappas, raw pairwise percent agreement, Gwet's AC1
      (paradox guard, pre-specified). Bootstrap percentile 95% CI, 2000 resamples,
      seed 20260729. All of this is fixed in screen_frame.json -> agreement.
  (2) pooled prevalence endpoints P1, S1-S9 with Wilson 95% intervals, de-duplicating
      the overlap set by majority vote with a pre-declared anti-self-serving tie-break.
  (3) the unconditional unreachable bounding analyses (protocol sec.7).

THREE-VALUED trivial_baseline. screen_frame.json declares the six trivial_baseline
sub-flags as booleans and gives NO level for "could not be assessed". The four screeners
independently invented four different conventions for excluded / unreachable records:
S1 wrote false, S2 wrote null, S3 wrote the string "unclear", S4 wrote false. This script
therefore reads the flags three-valued -- TRUE / FALSE / NOT_ASSESSABLE -- and reports the
codebook defect rather than silently coercing. Coercing "unclear" to truthy (the naive
read) manufactures nine phantom disagreements and drives Fleiss' kappa to -0.18; that
number is reported too, because it is what the pre-registered field would give if run
without this correction.
"""
import json, itertools, math, random
from collections import Counter, defaultdict

PAPER = '/Users/sathvikloke/Downloads/PhaseDx/paper'
SEED, NBOOT = 20260729, 2000
SCREENERS = ['S1', 'S2', 'S3', 'S4']

P1_KEYS = ['constant_or_prevalence', 'positional', 'acquisition_metadata',
           'permuted_or_shuffled_label']
S1_KEYS = P1_KEYS + ['clinical_or_demographic_only', 'other_non_imaging']
UNREACHABLE = {'unreachable_paywalled', 'unreachable_not_found'}
NA = 'NOT_ASSESSABLE'

# ------------------------------------------------------------------ load
PS = {}
for b in 'ABCD':
    d = json.load(open(f'{PAPER}/screen_batch_{b}.json'))
    PS[d['screener_id']] = d

overlap, unique = defaultdict(dict), {}
for s in SCREENERS:
    for r in PS[s]['records']:
        (overlap[r['record_id']].__setitem__(s, r) if r['batch'] == 'overlap'
         else unique.__setitem__(r['record_id'], r))
OV = sorted(overlap, key=lambda p: overlap[p]['S1']['permutation_position'])
assert len(OV) == 15 and len(unique) == 85
for p in OV:
    assert set(overlap[p]) == set(SCREENERS)

# ------------------------------------------------------------------ derived codes
def _flag(r, keys):
    """Three-valued OR over the named sub-flags."""
    vs = [r['trivial_baseline'].get(k) for k in keys]
    if any(v is True for v in vs):
        return True
    if all(v is False for v in vs):
        return False
    return NA                       # null / "unclear" / mixed -> not assessable

def p1(r): return _flag(r, P1_KEYS)
def s1(r): return _flag(r, S1_KEYS)
def got_fulltext(r): return r['fulltext_reachable'] not in UNREACHABLE

def status(r):
    """Canonical three-way status. A paper whose full text was never obtained cannot
    support an inclusion decision or a P1 code, whatever label the screener attached;
    this is the pre-registered denominator in screen_frame.json
    (endpoints._denominator_default), applied uniformly."""
    if r['final_inclusion'] == 'excluded':
        return 'excluded'
    return 'included_reachable' if got_fulltext(r) else 'eligible_unreachable'

# ------------------------------------------------------------------ agreement maths
def fleiss_and_friends(mat):
    n, m = len(mat), len(mat[0])
    cats = sorted({c for row in mat for c in row}, key=str)
    q = len(cats)
    if q == 1:
        return dict(pa=1.0, fleiss_kappa=float('nan'), gwet_ac1=1.0, n_items=n,
                    n_raters=m, n_categories=1,
                    note='one category observed; kappa and AC1 are 0/0 -- perfect agreement')
    idx = {c: i for i, c in enumerate(cats)}
    cnt = [[0] * q for _ in range(n)]
    for i, row in enumerate(mat):
        for c in row:
            cnt[i][idx[c]] += 1
    pa = sum(sum(k * (k - 1) for k in cnt[i]) / (m * (m - 1)) for i in range(n)) / n
    pk = [sum(cnt[i][j] for i in range(n)) / (n * m) for j in range(q)]
    pe_f = sum(v * v for v in pk)
    kap = (pa - pe_f) / (1 - pe_f) if abs(1 - pe_f) > 1e-12 else float('nan')
    pi = [sum(cnt[i][j] / m for i in range(n)) / n for j in range(q)]
    pe_g = sum(v * (1 - v) for v in pi) / (q - 1)
    ac1 = (pa - pe_g) / (1 - pe_g) if abs(1 - pe_g) > 1e-12 else float('nan')
    return dict(pa=pa, fleiss_kappa=kap, gwet_ac1=ac1, fleiss_pe=pe_f, gwet_pe=pe_g,
                n_items=n, n_raters=m, n_categories=q,
                category_counts={str(c): sum(cnt[i][idx[c]] for i in range(n)) for c in cats})

def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(map(str, a)), Counter(map(str, b))
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    return po, ((po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else float('nan'))

def boot_ci(mat, fn, nboot=NBOOT, seed=SEED):
    rng, n, vals = random.Random(seed), len(mat), defaultdict(list)
    for _ in range(nboot):
        try:
            out = fn([mat[rng.randrange(n)] for _ in range(n)])
        except Exception:
            continue
        for k, v in out.items():
            if isinstance(v, float) and not math.isnan(v):
                vals[k].append(v)
    ci = {}
    for k, v in vals.items():
        v.sort()
        ci[k] = [v[max(0, int(0.025 * len(v)) - 1)],
                 v[min(len(v) - 1, math.ceil(0.975 * len(v)) - 1)], len(v)]
    return ci

def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'),) * 3
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)

def agree_block(mat):
    base = fleiss_and_friends(mat)
    pw = {}
    for a, b in itertools.combinations(SCREENERS, 2):
        ia, ib = SCREENERS.index(a), SCREENERS.index(b)
        po, k = cohen([r[ia] for r in mat], [r[ib] for r in mat])
        pw[f'{a}-{b}'] = {'percent_agreement': po, 'cohen_kappa': k}
    dk = [v['cohen_kappa'] for v in pw.values() if not math.isnan(v['cohen_kappa'])]
    unan = sum(len(set(map(str, r))) == 1 for r in mat)
    return {**base, 'bootstrap_ci95': boot_ci(mat, fleiss_and_friends),
            'pairwise_cohen': pw,
            'mean_pairwise_cohen': (sum(dk) / len(dk)) if dk else float('nan'),
            'n_pairwise_kappa_defined': len(dk),
            'unanimous_items': unan, 'unanimous_frac': unan / len(mat)}

# ------------------------------------------------------------------ agreement
GETTERS = {
    'final_inclusion_as_coded':        lambda r: r['final_inclusion'],
    'status_canonical':                status,
    'P1_flag_three_valued':            lambda r: str(p1(r)),
    'P1_flag_naive_truthy':            lambda r: str(any(bool(r['trivial_baseline'].get(k))
                                                        for k in P1_KEYS)),
    'evaluation_unit_reported':        lambda r: r['evaluation_unit_reported'],
    'headline_unit':                   lambda r: str(r.get('headline_unit')),
    'split_unit':                      lambda r: r['split_unit'],
    'positional_distribution_reported':lambda r: r['positional_distribution_reported'],
    'fulltext_obtained':               lambda r: str(got_fulltext(r)),
}

agreement = {}
for f, g in GETTERS.items():
    mat = [[g(overlap[p][s]) for s in SCREENERS] for p in OV]
    agreement[f] = {**agree_block(mat),
                    'per_paper': {p: [g(overlap[p][s]) for s in SCREENERS] for p in OV}}

# restricted: papers where ALL FOUR obtained the full text and included it -- the only
# papers on which a P1 / evaluation-unit / split-unit code is even defined
CORE = [p for p in OV if all(status(overlap[p][s]) == 'included_reachable' for s in SCREENERS)]
restricted = {'pmids': CORE, 'n': len(CORE),
              '_meaning': 'papers all four screeners obtained AND included; the analysis-'
                          'relevant denominator, and the only set on which these fields '
                          'are defined'}
for f, g in GETTERS.items():
    if len(CORE) >= 2:
        restricted[f] = agree_block([[g(overlap[p][s]) for s in SCREENERS] for p in CORE])
agreement['_restricted_to_core'] = restricted

# ------------------------------------------------------------------ pooling
# Pre-declared, applied identically to every field: majority of the 4 independent codes;
# ties broken AGAINST our own thesis (toward the code that flatters the literature), so no
# headline number can be produced by an arbitrary tie-break.
def majority(vals, prefer=None):
    c = Counter(map(str, vals))
    top = max(c.values())
    uniq = []
    for v in vals:
        if c[str(v)] == top and str(v) not in map(str, uniq):
            uniq.append(v)
    if len(uniq) == 1:
        return uniq[0], False
    if prefer is not None and any(str(u) == str(prefer) for u in uniq):
        return prefer, True
    return sorted(uniq, key=str)[0], True

def row_from(rec_id, batch, st, P1, S1, eu, hu, su, pd, ur, npos):
    return dict(record_id=rec_id, batch=batch, status=st, _p1=P1, _s1=S1,
                evaluation_unit_reported=eu, headline_unit=hu, split_unit=su,
                positional_distribution_reported=pd,
                uncertainty_interval_reported=ur, n_positive_reported=npos)

rows, pool_log = [], {}
for p in OV:
    rs = [overlap[p][s] for s in SCREENERS]
    st,  t_st = majority([status(r) for r in rs], 'included_reachable')
    f1,  t_p1 = majority([p1(r) for r in rs], True)          # tie -> HAS a baseline
    fs1, _    = majority([s1(r) for r in rs], True)
    eu,  t_eu = majority([r['evaluation_unit_reported'] for r in rs])
    hu,  _    = majority([r.get('headline_unit') for r in rs])
    su,  t_su = majority([r['split_unit'] for r in rs], 'patient_subject')
    pd,  _    = majority([r['positional_distribution_reported'] for r in rs], 'figure_or_table')
    ur,  _    = majority([r['uncertainty_interval_reported'] for r in rs], 'ci_clustered_by_subject')
    nz,  _    = majority([r['n_positive_reported'] for r in rs], 'patients_and_slices')
    rows.append(row_from(p, 'overlap', st, f1, fs1, eu, hu, su, pd, ur, nz))
    pool_log[p] = dict(tie_status=t_st, tie_P1=t_p1, tie_eval_unit=t_eu, tie_split=t_su,
                       codes={s: dict(status=status(overlap[p][s]),
                                      final_inclusion=overlap[p][s]['final_inclusion'],
                                      fulltext=overlap[p][s]['fulltext_reachable'],
                                      P1=str(p1(overlap[p][s])),
                                      eval_unit=overlap[p][s]['evaluation_unit_reported'],
                                      headline_unit=overlap[p][s].get('headline_unit'),
                                      split_unit=overlap[p][s]['split_unit'],
                                      pos_dist=overlap[p][s]['positional_distribution_reported'])
                              for s in SCREENERS})
for p, r in unique.items():
    rows.append(row_from(p, r['batch'], status(r), p1(r), s1(r),
                         r['evaluation_unit_reported'], r.get('headline_unit'),
                         r['split_unit'], r['positional_distribution_reported'],
                         r['uncertainty_interval_reported'], r['n_positive_reported']))
assert len(rows) == 100 and len({r['record_id'] for r in rows}) == 100

CC   = [r for r in rows if r['status'] == 'included_reachable']
UNRE = [r for r in rows if r['status'] == 'eligible_unreachable']
EXCL = [r for r in rows if r['status'] == 'excluded']
ELIG = CC + UNRE

def ep(k, n, note=''):
    p, lo, hi = wilson(k, n)
    return dict(k=k, n=n, pct=100 * p if n else None,
                wilson95=[100 * lo, 100 * hi] if n else None, note=note)

def cnt(rs, f): return sum(1 for r in rs if f(r)), len(rs)
def hl_slice(r):
    return r['headline_unit'] == 'slice' or (r['headline_unit'] == 'na_only_one_unit_reported'
                                             and r['evaluation_unit_reported'] == 'slice')

k_p1 = sum(1 for r in CC if r['_p1'] is True)
k_s1 = sum(1 for r in CC if r['_s1'] is True)
k_s4 = sum(1 for r in CC if r['split_unit'] == 'patient_subject')
E = {}
E['P1_complete_case'] = ep(k_p1, len(CC), 'PRIMARY. Any measured zero-image baseline: '
                           'constant/prevalence, positional, acquisition-metadata or permuted-label.')
E['P1_lower_bound']   = ep(k_p1, len(ELIG), 'all unreachable coded as NOT reporting one')
E['P1_upper_bound']   = ep(k_p1 + len(UNRE), len(ELIG),
                           'all unreachable coded as reporting one -- HEADLINE, because '
                           'unreachable exceeds the protocol sec.7 15% threshold')
E['S1_any_non_imaging']  = ep(k_s1, len(CC))
E['S1_lower_bound']      = ep(k_s1, len(ELIG))
E['S1_upper_bound']      = ep(k_s1 + len(UNRE), len(ELIG))
E['S2_headline_slice']   = ep(*cnt(CC, hl_slice))
SL = [r for r in CC if r['evaluation_unit_reported'] in ('slice', 'both')]
E['S3_slice_also_patient'] = ep(sum(1 for r in SL if r['evaluation_unit_reported'] == 'both'), len(SL))
E['S4_patient_split']    = ep(k_s4, len(CC))
E['S4_lower_bound']      = ep(k_s4, len(ELIG))
E['S4_upper_bound']      = ep(k_s4 + len(UNRE), len(ELIG))
E['S5_positional_distribution'] = ep(*cnt(CC, lambda r: r['positional_distribution_reported']
                                          in ('figure_or_table', 'text_with_numbers')))
E['S5b_incl_qualitative'] = ep(*cnt(CC, lambda r: r['positional_distribution_reported']
                                    in ('figure_or_table', 'text_with_numbers', 'text_qualitative_only')))
E['S5_upper_bound']       = ep(sum(1 for r in CC if r['positional_distribution_reported']
                                   in ('figure_or_table', 'text_with_numbers')) + len(UNRE), len(ELIG))
E['S6_unreachable']       = ep(len(UNRE), len(ELIG))
E['S8_clustered_ci']      = ep(*cnt(CC, lambda r: r['uncertainty_interval_reported'] == 'ci_clustered_by_subject'))
E['S9_positive_patients_and_slices'] = ep(*cnt(CC, lambda r: r['n_positive_reported'] == 'patients_and_slices'))
E['ANY_slice_level_metric'] = ep(*cnt(CC, lambda r: r['evaluation_unit_reported'] in ('slice', 'both')))
E['sub_patient_eval_unit']  = ep(*cnt(CC, lambda r: r['evaluation_unit_reported']
                                      in ('slice', 'both', 'lesion', 'volume_or_scan_not_patient')))

s7 = defaultdict(lambda: {'n': 0, 'P1_true': 0})
for r in CC:
    key = (r['headline_unit'] if r['headline_unit'] != 'na_only_one_unit_reported'
           else f"only_{r['evaluation_unit_reported']}")
    s7[key]['n'] += 1
    s7[key]['P1_true'] += int(r['_p1'] is True)

def excl_code(rid):
    return (Counter(overlap[rid][s]['exclusion_code'] for s in SCREENERS).most_common(1)[0][0]
            if rid in overlap else unique[rid]['exclusion_code'])

flow = dict(records_screened=len(rows), included_and_reachable=len(CC),
            eligible_but_unreachable=len(UNRE), excluded=len(EXCL), eligible_looking=len(ELIG),
            exclusion_codes=dict(Counter(excl_code(r['record_id']) for r in EXCL)),
            protocol_target_included=75, target_met=len(CC) >= 75,
            extension_rule_triggered=len(CC) < 75)

audit = dict(
    p1_true_records_anywhere=[f"{s}/{r['record_id']}/{k}" for s in SCREENERS
                              for r in PS[s]['records'] for k in P1_KEYS
                              if r['trivial_baseline'].get(k) is True],
    clinical_only_true_records=[f"{s}/{r['record_id']}" for s in SCREENERS
                                for r in PS[s]['records']
                                if r['trivial_baseline'].get('clinical_or_demographic_only') is True],
    trivial_baseline_na_convention={s: dict(Counter(
        type(r['trivial_baseline']['positional']).__name__ + ':' + repr(r['trivial_baseline']['positional'])
        for r in PS[s]['records'])) for s in SCREENERS},
    included_but_unreachable_as_coded={s: sum(1 for r in PS[s]['records']
                                              if r['final_inclusion'] == 'included' and not got_fulltext(r))
                                       for s in SCREENERS},
)

dists = {f: dict(Counter(r[f] for r in CC)) for f in
         ['evaluation_unit_reported', 'headline_unit', 'split_unit',
          'positional_distribution_reported', 'uncertainty_interval_reported',
          'n_positive_reported']}

json.dump(dict(_generated_by='paper/screen/analysis/pool_and_agree.py',
               _inputs=[f'paper/screen_batch_{b}.json' for b in 'ABCD'],
               _protocol='paper/screen_protocol.md v1.1; paper/screen_frame.json v1.0',
               _seed=SEED, _nboot=NBOOT, flow=flow, agreement=agreement,
               pooling_rule=('majority of the 4 independent codes on the 15 overlap papers; '
                             'ties broken AGAINST our own thesis'),
               pooling_log=pool_log, endpoints=E, S7_headline_unit_by_P1=dict(s7),
               distributions_complete_case=dists, codebook_audit=audit,
               per_record=sorted(rows, key=lambda r: r['record_id'])),
          open(f'{PAPER}/screen/analysis/analysis_out.json', 'w'), indent=1, default=str)

# ------------------------------------------------------------------ report
def line(t, a):
    ci = a['bootstrap_ci95']
    g = lambda k: (f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else '   n/a        ')
    kk = a['fleiss_kappa']; ac = a['gwet_ac1']
    print(f"  {t:34s} raw {a['pa']*100:5.1f}% {g('pa'):18s} "
          f"kappa {('  n/d' if math.isnan(kk) else f'{kk:6.3f}')} {g('fleiss_kappa'):18s} "
          f"AC1 {('  n/d' if math.isnan(ac) else f'{ac:6.3f}')} {g('gwet_ac1'):18s} "
          f"unan {a['unanimous_items']}/{a['n_items']}  cats {a['n_categories']}")

print('=' * 130); print('FLOW'); print(json.dumps(flow, indent=1))
print('=' * 130); print(f'AGREEMENT -- all 15 overlap papers, 4 screeners (bootstrap 95% CI, {NBOOT} resamples, seed {SEED})')
for f in GETTERS: line(f, agreement[f])
print(f'\nAGREEMENT -- restricted to the {len(CORE)} papers all four obtained AND included')
for f in GETTERS:
    if f in restricted: line(f, restricted[f])
print('=' * 130); print('ENDPOINTS')
for k, v in E.items():
    w = v['wilson95']
    print(f"  {k:34s} {v['k']:3d}/{v['n']:<3d} = {v['pct']:5.1f}%   Wilson95 [{w[0]:5.1f}, {w[1]:5.1f}]")
print('\nS7 headline unit x P1:'); print(json.dumps(dict(s7), indent=1))
print('\nCOMPLETE-CASE DISTRIBUTIONS:'); print(json.dumps(dists, indent=1))
print('\nCODEBOOK AUDIT:'); print(json.dumps(audit, indent=1))
print('\nOVERLAP TIES:')
for p, l in pool_log.items():
    ts = [k for k in ('tie_status', 'tie_P1', 'tie_eval_unit', 'tie_split') if l[k]]
    if ts: print('  ', p, ts)

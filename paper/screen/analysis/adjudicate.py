#!/usr/bin/env python3
"""
PhaseDx prevalence screen -- ADJUDICATION ROUND (pre-registered remedy).

Triggered by screen_frame.json -> agreement.threshold_and_remedy: Fleiss' kappa on the
P1 flag was -0.015 (floor 0.60) and raw agreement 65.6% (paradox-guard floor 90%), so a
documented adjudication round, a codebook amendment and a re-coding are mandatory.

Inputs  : paper/screen_batch_{A,B,C,D}.json  (four sealed screener files, UNCHANGED)
Outputs : paper/screen/analysis/adjudication_out.json + a console report
Writes   nothing back into the sealed files. The amended codebook is
          paper/screen_frame.json v1.2; the audit trail is paper/screen_adjudication.md.

WHAT IS AND IS NOT A RELIABILITY STATISTIC
------------------------------------------
Three numbers are produced and they mean three different things.

  (1) PRE-RECONCILIATION.  The four sealed files as coded under codebook v1.0.
      This is the honest reliability of the original round and is reported first.

  (2) COUNTERFACTUAL v1.2-ENCODING.  The same four sealed files, with NO screener's
      reading of any paper altered, re-expressed under the two amendments that only add
      a MISSING LEVEL to the form:
        D2  trivial_baseline sub-flags gain a third value, not_assessable, used when the
            mandatory 14-term full-text search could not be run.
        D3  descriptive fields gain not_applicable, used when the screener's own
            final_inclusion is not 'included', so the field does not exist on that record.
      Each screener's own final_inclusion and own fulltext_reachable drive the transform,
      so genuine eligibility disagreements SURVIVE it.  This is the number that answers
      'was the agreement failure a codebook defect or a screener failure?'.
      It is NOT an independent re-rating.

  (3) POST-ADJUDICATION.  One consensus code per paper.  Agreement on it is 1.000 BY
      CONSTRUCTION and is reported only to make that explicit.  It is not evidence of
      anything and the protocol floor is NOT assessed against it.

Intervals: bootstrap percentile 95%, 2000 resamples of the 15 papers, seed 20260729,
exactly as pre-registered in screen_frame.json -> agreement.interval.
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
NA_ASSESS, NA_APPLIC = 'not_assessable', 'not_applicable'

# fields the protocol names in agreement.fields_with_reported_agreement, plus the two
# upstream fields the adjudication showed to be driving them
DESCRIPTIVE = ['evaluation_unit_reported', 'headline_unit', 'split_unit',
               'positional_distribution_reported']

# ------------------------------------------------------------------ load (read-only)
PS = {}
for b in 'ABCD':
    d = json.load(open(f'{PAPER}/screen_batch_{b}.json'))
    PS[d['screener_id']] = d
overlap = defaultdict(dict)
for s in SCREENERS:
    for r in PS[s]['records']:
        if r['batch'] == 'overlap':
            overlap[r['record_id']][s] = r
OV = sorted(overlap, key=lambda p: overlap[p]['S1']['permutation_position'])
assert len(OV) == 15 and all(set(overlap[p]) == set(SCREENERS) for p in OV)

def obtained(r):  return r['fulltext_reachable'] not in UNREACHABLE

# ------------------------------------------------------------------ codings
def p1_v10(r):
    """v1.0 three-valued read, keyed on the VALUE the screener wrote (what
    pool_and_agree.py does).  S1/S4 wrote False on unreachable records, S2 wrote null,
    S3 wrote 'unclear' -- so this still splits four ways in substance."""
    vs = [r['trivial_baseline'].get(k) for k in P1_KEYS]
    if any(v is True for v in vs):  return 'True'
    if all(v is False for v in vs): return 'False'
    return 'NOT_ASSESSABLE'

def p1_v12(r):
    """v1.2 read, keyed on WHETHER THE EVIDENCE EXISTS, not on which placeholder the
    screener happened to type.  Amendments D2 + D3."""
    if r['final_inclusion'] == 'excluded':      # D3: field undefined on this record
        return NA_APPLIC
    if not obtained(r):                          # D2: mandatory full-text search impossible
        return NA_ASSESS
    vs = [r['trivial_baseline'].get(k) for k in P1_KEYS]
    if any(v is True for v in vs):  return 'True'
    if all(v is False for v in vs): return 'False'
    return NA_ASSESS

def p1_v12_collapsed(r):
    v = p1_v12(r)
    return 'NOT_CODED' if v in (NA_APPLIC, NA_ASSESS) else v

def desc_v12(field):
    def g(r):
        if r['final_inclusion'] != 'included':   # D3
            return NA_APPLIC
        return str(r.get(field))
    return g

GET_V10 = {
    'final_inclusion':                  lambda r: r['final_inclusion'],
    'fulltext_obtained':                lambda r: str(obtained(r)),
    'P1_flag':                          p1_v10,
    'evaluation_unit_reported':         lambda r: r['evaluation_unit_reported'],
    'headline_unit':                    lambda r: str(r.get('headline_unit')),
    'split_unit':                       lambda r: r['split_unit'],
    'positional_distribution_reported': lambda r: r['positional_distribution_reported'],
}
GET_V12 = {
    'final_inclusion':                  lambda r: r['final_inclusion'],   # untouched by D2/D3
    'fulltext_obtained':                lambda r: str(obtained(r)),       # untouched
    'P1_flag':                          p1_v12,
    'P1_flag_collapsed':                p1_v12_collapsed,
    **{f: desc_v12(f) for f in DESCRIPTIVE},
}

# ------------------------------------------------------------------ agreement maths
def fleiss_and_friends(mat):
    n, m = len(mat), len(mat[0])
    cats = sorted({c for row in mat for c in row}, key=str)
    q = len(cats)
    if q == 1:
        return dict(pa=1.0, fleiss_kappa=float('nan'), gwet_ac1=1.0, n_items=n,
                    n_raters=m, n_categories=1)
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

def boot_ci(mat, nboot=NBOOT, seed=SEED):
    rng, n, vals = random.Random(seed), len(mat), defaultdict(list)
    for _ in range(nboot):
        out = fleiss_and_friends([mat[rng.randrange(n)] for _ in range(n)])
        for k, v in out.items():
            if isinstance(v, float) and not math.isnan(v):
                vals[k].append(v)
    ci = {}
    for k, v in vals.items():
        v.sort()
        ci[k] = [v[max(0, int(0.025 * len(v)) - 1)],
                 v[min(len(v) - 1, math.ceil(0.975 * len(v)) - 1)], len(v)]
    return ci

def agree_block(mat):
    base = fleiss_and_friends(mat)
    pw = {}
    for a, b in itertools.combinations(SCREENERS, 2):
        ia, ib = SCREENERS.index(a), SCREENERS.index(b)
        po, k = cohen([r[ia] for r in mat], [r[ib] for r in mat])
        pw[f'{a}-{b}'] = {'percent_agreement': po, 'cohen_kappa': k}
    dk = [v['cohen_kappa'] for v in pw.values() if not math.isnan(v['cohen_kappa'])]
    unan = sum(len(set(map(str, r))) == 1 for r in mat)
    return {**base, 'bootstrap_ci95': boot_ci(mat), 'pairwise_cohen': pw,
            'mean_pairwise_cohen': (sum(dk) / len(dk)) if dk else float('nan'),
            'n_pairwise_kappa_defined': len(dk),
            'unanimous_items': unan, 'unanimous_frac': unan / len(mat)}

def block_for(getters, papers=OV):
    out = {}
    for f, g in getters.items():
        mat = [[g(overlap[p][s]) for s in SCREENERS] for p in papers]
        out[f] = {**agree_block(mat),
                  'per_paper': {p: [g(overlap[p][s]) for s in SCREENERS] for p in papers}}
    return out

def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float('nan'),) * 3
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)

# ------------------------------------------------------------------ (1) and (2)
PRE  = block_for(GET_V10)
POST = block_for(GET_V12)

# CORE = papers all four screeners both obtained AND included: the only papers on which a
# P1 code is even defined.  The diagnostic that identifies where the disagreement is.
CORE = [p for p in OV if all(obtained(overlap[p][s]) and
                             overlap[p][s]['final_inclusion'] == 'included'
                             for s in SCREENERS)]
CORE_BLOCK = block_for(GET_V10, CORE)
# and the raw six-sub-flag vector on those papers, the sharpest form of the diagnostic
core_vec = [[tuple(overlap[p][s]['trivial_baseline'].get(k) for k in S1_KEYS)
             for s in SCREENERS] for p in CORE]
CORE_SUBFLAG = agree_block([[str(v) for v in row] for row in core_vec])

# ------------------------------------------------------------------ (3) adjudicated codes
# One row per overlap paper.  'rule' names the v1.2 amendment (or the v1.0 rule) that
# decides it; 'changed_from' lists the screeners whose original code it overturns.
ADJ = {
 '36776294': dict(final_inclusion='included', fulltext='oa_pmc_or_publisher',
   P1='False', eval_unit='lesion', headline_unit='na_only_one_unit_reported',
   split_unit='lesion_or_roi', pos_dist='no',
   rule='D8 (new split_unit level lesion_or_roi)',
   changed_from='all four coded slice_or_image by forced mapping; no agreement effect',
   note='unanimous on every agreement field; the change is a more accurate level, not a dispute'),
 '41617832': dict(final_inclusion='unreachable_eligibility_unresolved',
   fulltext='unreachable_paywalled', P1=NA_ASSESS, eval_unit=NA_APPLIC,
   headline_unit=NA_APPLIC, split_unit=NA_APPLIC, pos_dist=NA_APPLIC,
   rule='D1 (unreachable dominates included); D2; D3',
   changed_from='S1, S4 (coded included)',
   note='S1/S4 also coded clinical_or_demographic_only=TRUE, but the abstract names the '
        'clinical arm without reporting its value; the codebook already says an assertion '
        'with no number is FALSE for every flag, so under D2 this is not_assessable, not TRUE'),
 '39423605': dict(final_inclusion='unreachable_eligibility_unresolved',
   fulltext='unreachable_paywalled', P1=NA_ASSESS, eval_unit=NA_APPLIC,
   headline_unit=NA_APPLIC, split_unit=NA_APPLIC, pos_dist=NA_APPLIC,
   rule='D2; D3', changed_from='none (unanimous on final_inclusion)',
   note='the only divergence was the placeholder written into the sub-flags'),
 '42130124': dict(final_inclusion='included', fulltext='oa_pmc_or_publisher',
   P1='False', eval_unit='other', headline_unit='na_only_one_unit_reported',
   split_unit='random_unit_not_stated', pos_dist='figure_or_table',
   rule='D6 (patient-naming noun required) for split_unit; D9 (unit-ordering test) for pos_dist',
   changed_from='S1, S2 on split_unit; S2, S3, S4 on positional_distribution',
   note='TIGHTENS split (removes a paper from S4) and LOOSENS pos_dist (adds a paper to S5) '
        '-- the two adjudications move the headline numbers in opposite directions'),
 '36789248': dict(final_inclusion='excluded', fulltext='oa_pmc_or_publisher', P1=NA_APPLIC,
   eval_unit=NA_APPLIC, headline_unit=NA_APPLIC, split_unit=NA_APPLIC, pos_dist=NA_APPLIC,
   rule='D3', changed_from='S2, S3, S4 (coded slice / real design values)',
   note='exclusion code E-DERIV unanimous'),
 '40335658': dict(final_inclusion='excluded', fulltext='not_attempted_excluded_at_stage1',
   P1=NA_APPLIC, eval_unit=NA_APPLIC, headline_unit=NA_APPLIC, split_unit=NA_APPLIC,
   pos_dist=NA_APPLIC,
   rule='D10 (exclusion code -> E-NOCLF); D4 (fulltext on a stage-1 exclusion); D3; D12',
   changed_from='S1, S3, S4 on exclusion_code (E-SEG -> E-NOCLF); S1, S3, S4 on fulltext_reachable; S4 on modality',
   note='S2 was right: E-SEG is qualified "with NO categorical class decision evaluated", '
        'and a class decision WAS evaluated -- by human readers, which fails I2, i.e. E-NOCLF'),
 '40194851': dict(final_inclusion='excluded', fulltext='not_attempted_excluded_at_stage1',
   P1=NA_APPLIC, eval_unit=NA_APPLIC, headline_unit=NA_APPLIC, split_unit=NA_APPLIC,
   pos_dist=NA_APPLIC, rule='D4; D3', changed_from='S1, S3 on fulltext_reachable',
   note='E-DERIV unanimous; all divergence was in placeholder conventions'),
 '42489954': dict(final_inclusion='unreachable_eligibility_unresolved',
   fulltext='unreachable_paywalled', P1=NA_ASSESS, eval_unit=NA_APPLIC,
   headline_unit=NA_APPLIC, split_unit=NA_APPLIC, pos_dist=NA_APPLIC,
   rule='v1.0 stage1_decision rule + D1 + D11', changed_from='S1, S4 (coded excluded E-DERIV)',
   note='decided by the EXISTING rule, not a new one: stage-1 exclusion is permitted only '
        'when the code is unambiguous from the abstract, all four coded '
        'stage1_decision=go_to_fulltext, and S1 and S4 both recorded low confidence and '
        'asked for adjudication in their own notes'),
 '39061744': dict(final_inclusion='included', fulltext='oa_pmc_or_publisher', P1='False',
   eval_unit='patient', headline_unit='na_only_one_unit_reported',
   split_unit='patient_subject', pos_dist='no', rule='none needed',
   changed_from='none', note='unanimous on every agreement field AND on all six sub-flags '
        '(clinical_or_demographic_only TRUE, S1 endpoint only)'),
 '31093705': dict(final_inclusion='included', fulltext='oa_pmc_or_publisher', P1='False',
   eval_unit='lesion', headline_unit='na_only_one_unit_reported', split_unit='unclear',
   pos_dist='no', rule='D8 (methods deferred to an unsampled companion paper -> unclear)',
   changed_from='S1 (coded slice_or_image)',
   note='the new lesion_or_roi level exists, but this paper states no split unit at all: '
        'it defers to Part I, which is not in the sample'),
 '36016875': dict(final_inclusion='included', fulltext='oa_pmc_or_publisher', P1='False',
   eval_unit='slice', headline_unit='na_only_one_unit_reported', split_unit='slice_or_image',
   pos_dist='no', rule='D5 (Methods govern over Abstract on a factual contradiction)',
   changed_from='S1 (coded patient / patient_subject from the abstract)',
   note='Abstract says "70% of cases"; Methods split 2,458 image units against only 119 '
        'patients, which is arithmetically incompatible with a patient-level split'),
 '36072854': dict(final_inclusion='excluded', fulltext='oa_pmc_or_publisher', P1=NA_APPLIC,
   eval_unit=NA_APPLIC, headline_unit=NA_APPLIC, split_unit=NA_APPLIC, pos_dist=NA_APPLIC,
   rule='D3', changed_from='S2, S4 (coded real design values on an excluded record)',
   note='E-SEG unanimous'),
 '37222638': dict(final_inclusion='unreachable_eligibility_unresolved',
   fulltext='unreachable_paywalled', P1=NA_ASSESS, eval_unit=NA_APPLIC,
   headline_unit=NA_APPLIC, split_unit=NA_APPLIC, pos_dist=NA_APPLIC,
   rule='D1; D2 (positive/negative asymmetry); D3',
   changed_from='S1, S4 on final_inclusion; S3 on clinical_or_demographic_only',
   note='clinical_or_demographic_only = TRUE IS RECORDED and held pending eligibility '
        'resolution: the abstract itself reports three measured AUCs for a pixel-free '
        'clinical arm (0.741 / 0.772 / 0.675). Under D2 a POSITIVE may rest on an abstract '
        'quote that carries the value; only a NEGATIVE requires the full-text search'),
 '40239684': dict(final_inclusion='excluded', fulltext='not_attempted_excluded_at_stage1',
   P1=NA_APPLIC, eval_unit=NA_APPLIC, headline_unit=NA_APPLIC, split_unit=NA_APPLIC,
   pos_dist=NA_APPLIC, rule='D4; D3', changed_from='S1, S3 on fulltext_reachable',
   note='E-SEG unanimous'),
 '41068276': dict(final_inclusion='included', fulltext='oa_pmc_or_publisher', P1='False',
   eval_unit='unclear', headline_unit='na_only_one_unit_reported',
   split_unit='slice_or_image', pos_dist='no',
   rule='D7 (a unit named in a table caption is a named unit); D14 (headline_unit)',
   changed_from='S1 on split_unit; S4 on headline_unit',
   note='Table 2 is headed "Distribution of MRI images by dataset split", so the unit IS '
        'named -- random_unit_not_stated is for when it is not named anywhere'),
}
assert set(ADJ) == set(OV)

# ------------------------------------------------------------------ endpoint effect
# Only the overlap set is re-adjudicated here; the 85 unique records are re-coded by the
# mechanical D2/D3/D4 rules alone, which touch no endpoint numerator or denominator.
old = json.load(open(f'{PAPER}/screen/analysis/analysis_out.json'))
old_rows = {r['record_id']: r for r in old['per_record']}
new_rows = dict(old_rows)
for p, a in ADJ.items():
    r = dict(old_rows[p])
    r['status'] = ('excluded' if a['final_inclusion'] == 'excluded' else
                   'included_reachable' if a['fulltext'] not in UNREACHABLE
                   else 'eligible_unreachable')
    r['_p1'] = True if a['P1'] == 'True' else (False if a['P1'] == 'False' else a['P1'])
    r['evaluation_unit_reported'] = a['eval_unit']
    r['headline_unit'] = a['headline_unit']
    r['split_unit'] = a['split_unit']
    r['positional_distribution_reported'] = a['pos_dist']
    new_rows[p] = r

def endpoints(rows):
    CC = [r for r in rows.values() if r['status'] == 'included_reachable']
    UN = [r for r in rows.values() if r['status'] == 'eligible_unreachable']
    EL = CC + UN
    def ep(k, n):
        p, lo, hi = wilson(k, n)
        return dict(k=k, n=n, pct=round(100 * p, 2) if n else None,
                    wilson95=[round(100 * lo, 2), round(100 * hi, 2)] if n else None)
    hl_slice = lambda r: (r['headline_unit'] == 'slice' or
                          (r['headline_unit'] == 'na_only_one_unit_reported'
                           and r['evaluation_unit_reported'] == 'slice'))
    kp1 = sum(1 for r in CC if r['_p1'] is True)
    return dict(
        n_complete_case=len(CC), n_unreachable=len(UN), n_eligible=len(EL),
        n_excluded=sum(1 for r in rows.values() if r['status'] == 'excluded'),
        P1_complete_case=ep(kp1, len(CC)),
        P1_lower_bound=ep(kp1, len(EL)),
        P1_upper_bound=ep(kp1 + len(UN), len(EL)),
        S2_headline_slice=ep(sum(1 for r in CC if hl_slice(r)), len(CC)),
        S4_patient_split=ep(sum(1 for r in CC if r['split_unit'] == 'patient_subject'), len(CC)),
        S5_positional_distribution=ep(sum(1 for r in CC if r['positional_distribution_reported']
                                          in ('figure_or_table', 'text_with_numbers')), len(CC)),
        S6_unreachable=ep(len(UN), len(EL)))

EP_BEFORE, EP_AFTER = endpoints(old_rows), endpoints(new_rows)

# ------------------------------------------------------------------ where the P1 flag TRUE
p1_true_anywhere = [f"{s}/{r['record_id']}/{k}" for s in SCREENERS for r in PS[s]['records']
                    for k in P1_KEYS if r['trivial_baseline'].get(k) is True]
clin_true = sorted({r['record_id'] for s in SCREENERS for r in PS[s]['records']
                    if r['trivial_baseline'].get('clinical_or_demographic_only') is True})

FLOOR = dict(
    kappa_floor=0.60, raw_floor_under_paradox_guard=0.90,
    pre_kappa=PRE['P1_flag']['fleiss_kappa'], pre_raw=PRE['P1_flag']['pa'],
    pre_met=False,
    post_kappa=POST['P1_flag']['fleiss_kappa'], post_raw=POST['P1_flag']['pa'],
    post_collapsed_kappa=POST['P1_flag_collapsed']['fleiss_kappa'],
    post_collapsed_raw=POST['P1_flag_collapsed']['pa'],
    adjudicated_raw=1.0,
    adjudicated_kappa='undefined -- one category, and 1.000 BY CONSTRUCTION; not a '
                      'reliability statistic and the floor is NOT assessed against it')

out = dict(
    _generated_by='paper/screen/analysis/adjudicate.py',
    _protocol='paper/screen_protocol.md v1.2; paper/screen_frame.json v1.2',
    _inputs=[f'paper/screen_batch_{b}.json' for b in 'ABCD'],
    _sealed_files_modified=False, _seed=SEED, _nboot=NBOOT,
    pre_reconciliation_v10=PRE,
    counterfactual_v12_encoding=POST,
    restricted_to_core=dict(pmids=CORE, n=len(CORE), by_field=CORE_BLOCK,
                            six_subflag_vector=CORE_SUBFLAG),
    adjudicated_codes=ADJ,
    endpoints_before=EP_BEFORE, endpoints_after=EP_AFTER,
    floor=FLOOR,
    p1_true_records_anywhere=p1_true_anywhere,
    clinical_only_true_records=clin_true)
json.dump(out, open(f'{PAPER}/screen/analysis/adjudication_out.json', 'w'),
          indent=1, default=str)

# ------------------------------------------------------------------ report
def line(t, a):
    ci = a['bootstrap_ci95']
    g = lambda k: (f"[{ci[k][0]:6.3f},{ci[k][1]:6.3f}]" if k in ci else '      n/a      ')
    kk, ac = a['fleiss_kappa'], a['gwet_ac1']
    print(f"  {t:34s} raw {a['pa']*100:5.1f}% {g('pa')} "
          f"k {('  n/d' if math.isnan(kk) else f'{kk:6.3f}')} {g('fleiss_kappa')} "
          f"AC1 {('  n/d' if math.isnan(ac) else f'{ac:6.3f}')} {g('gwet_ac1')} "
          f"unan {a['unanimous_items']:2d}/{a['n_items']}  cats {a['n_categories']}")

W = 138
print('=' * W); print(f'(1) PRE-RECONCILIATION -- four sealed files as coded under v1.0, 15 overlap papers')
for f in GET_V10: line(f, PRE[f])
print(f"\n    pairwise Cohen on the P1 flag: " +
      '  '.join(f"{k} {v['cohen_kappa']:.3f}" for k, v in PRE['P1_flag']['pairwise_cohen'].items()))

print('=' * W)
print(f'(1b) DIAGNOSTIC -- restricted to the {len(CORE)} papers all four OBTAINED and INCLUDED')
print('     (the only papers on which a P1 code is defined at all)')
for f in GET_V10: line(f, CORE_BLOCK[f])
print(f"  {'ALL SIX trivial_baseline sub-flags':34s} raw {CORE_SUBFLAG['pa']*100:5.1f}%"
      f"  unanimous {CORE_SUBFLAG['unanimous_items']}/{CORE_SUBFLAG['n_items']}"
      f"  categories observed {CORE_SUBFLAG['n_categories']}")

print('=' * W)
print('(2) COUNTERFACTUAL v1.2 ENCODING -- same sealed files, no reading altered, only the')
print('    two missing levels supplied (D2 not_assessable, D3 not_applicable)')
for f in GET_V12: line(f, POST[f])
print(f"\n    pairwise Cohen on the P1 flag: " +
      '  '.join(f"{k} {v['cohen_kappa']:.3f}" for k, v in POST['P1_flag']['pairwise_cohen'].items()))

print('=' * W); print('(3) FLOOR')
print(json.dumps(FLOOR, indent=1, default=str))

print('=' * W); print('ENDPOINT EFFECT OF THE ADJUDICATION (overlap set only)')
for k in EP_BEFORE:
    a, b = EP_BEFORE[k], EP_AFTER[k]
    if isinstance(a, dict):
        mark = '   <-- CHANGED' if a != b else ''
        print(f"  {k:28s} before {a['k']:3d}/{a['n']:<3d} {a['pct']:5.1f}% "
              f"[{a['wilson95'][0]:5.1f},{a['wilson95'][1]:5.1f}]   "
              f"after {b['k']:3d}/{b['n']:<3d} {b['pct']:5.1f}% "
              f"[{b['wilson95'][0]:5.1f},{b['wilson95'][1]:5.1f}]{mark}")
    else:
        print(f"  {k:28s} before {a}   after {b}"
              + ('   <-- CHANGED' if a != b else ''))

print('=' * W)
print(f'P1 sub-flag coded TRUE anywhere in all 145 coded records: {p1_true_anywhere or "NONE"}')
print(f'clinical_or_demographic_only TRUE (S1 endpoint, not P1): {clin_true}')
print(f'\nwritten: {PAPER}/screen/analysis/adjudication_out.json')

"""Scoring contract v28. Modes: clinical_external (locked recipe, no overrides) | synthetic_point_only (never RNG/bootstrap/torch). Labels opened only via opener after admission, and only after schema + grouping binding verification."""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from . import protocol28 as P, reference28 as R, scorer26 as S26
BOOT = S26.cluster_bootstrap

class ContractError(RuntimeError):
    pass

def _unique_rows(path, key):
    rows = list(csv.DictReader(open(path)))
    seen = set()
    dup = []
    for r in rows:
        if r[key] in seen:
            dup.append(r[key])
        seen.add(r[key])
    return (rows, dup)

def load_person_map(path, roster=None):
    """Rejects duplicate exam keys (even identical), blank exam/person keys. Supplied-but-bad linkage is an error, never a fallback."""
    rows = list(csv.DictReader(open(path)))
    m = {}
    for r in rows:
        k, p = ((r.get('exam_key') or '').strip(), (r.get('person_key') or '').strip())
        if not k or not p:
            raise P.AdmissionError('person map has blank exam_key or person_key')
        if k in m:
            raise P.AdmissionError('person map has duplicate exam_key %r (%s)' % (k, 'conflicting' if m[k] != p else 'repeated'))
        m[k] = p
    return m

def verify_grouping_binding(path):
    b = json.load(open(path))
    for k in ('map_path', 'map_sha256', 'evidence_path', 'evidence_sha256', 'evidence_locator'):
        if not b.get(k):
            raise P.AdmissionError('grouping binding missing ' + k)
    for pk, hk in (('map_path', 'map_sha256'), ('evidence_path', 'evidence_sha256')):
        if not os.path.exists(b[pk]) or P.sha_file(b[pk]) != b[hk]:
            raise P.AdmissionError('grouping binding file missing or changed: ' + pk)
    if os.path.getsize(b['evidence_path']) == 0:
        raise P.AdmissionError('empty linkage evidence')
    load_person_map(b['map_path'])
    return b

def evaluate(design_path, exec_path, receipt_path, runs, mdir, candidates_csv, label_csv, schema_path, grouping_binding=None, **overrides):
    design = json.load(open(design_path))
    mode = design['mode']
    if mode == 'clinical_external' and overrides:
        raise ContractError('runtime overrides rejected in clinical mode: ' + ','.join(overrides))
    if mode == 'synthetic_point_only' and overrides.get('n_draws'):
        raise ContractError('point-only mode takes no draws')
    state = {}

    def opener():
        schema = json.load(open(schema_path))
        R.validate_schema(schema)
        if mode == 'clinical_external' and P.sha_file(schema_path) != design['clinical_contract']['reference_schema_sha256']:
            raise P.AdmissionError('reference schema differs from the locked schema identity')
        gb = verify_grouping_binding(grouping_binding) if grouping_binding else None
        state['gb'] = gb
        state['schema'] = schema
        return R.adapt(list(csv.DictReader(open(label_csv))), schema)
    adm, labs = P.admit(design_path, exec_path, receipt_path, runs, mdir, candidates_csv, opener)
    cand, dupc = _unique_rows(candidates_csv, 'exam_key')
    if dupc:
        raise P.AdmissionError('duplicate candidate keys: ' + ','.join(dupc))
    roster = [r['exam_key'] for r in cand]
    rs = set(roster)
    elig = {r['exam_key']: r['header_eligibility'] for r in cand}
    lab, dup_l, extra_l = ({}, [], [])
    for r in labs:
        k = r['exam_key']
        if k is None or k not in rs:
            extra_l.append(r['source_key'])
            continue
        if k in lab:
            dup_l.append(k)
        lab[k] = r['label']
    if dup_l:
        raise P.AdmissionError('duplicate label keys: ' + ','.join(sorted(set(dup_l))))

    def preds(role):
        rows, dup = _unique_rows(runs[role][1], 'exam_key')
        if dup:
            raise P.AdmissionError('duplicate prediction keys (%s)' % role)
        o, extra = ({}, [])
        for r in rows:
            if r['exam_key'] not in rs:
                extra.append(r['exam_key'])
                continue
            ok = r.get('status') in ('scored', 'scored_complete') and r.get('score') not in ('', 'nan') and np.isfinite(float(r['score']))
            o[r['exam_key']] = float(r['score']) if ok else float('nan')
        return (o, extra)
    img, ex_i = preds('primary_image')
    geo, ex_g = preds('geometry')
    sec, ex_s = preds('secondary_image') if adm['secondary']['status'] == 'admitted' else ({}, [])
    part = {}
    for k in roster:
        e = elig[k]
        if e == 'unresolved':
            part[k] = 'unresolved_eligibility'
        elif e != 'eligible':
            part[k] = 'source_or_clinically_ineligible'
        elif lab.get(k, '') == '':
            part[k] = 'eligible_reference_missing'
        else:
            fi, fg = (not np.isfinite(img.get(k, np.nan)), not np.isfinite(geo.get(k, np.nan)))
            part[k] = 'labeled_both_fail' if fi and fg else 'labeled_image_fail' if fi else 'labeled_geometry_fail' if fg else 'labeled_paired_evaluable'
    flow = {}
    for v in part.values():
        flow[v] = flow.get(v, 0) + 1
    L = [k for k in roster if part[k].startswith('labeled')]
    y = np.array([int(lab[k]) for k in L])
    sa = np.array([img.get(k, np.nan) for k in L])
    sb = np.array([geo.get(k, np.nan) for k in L])
    okp = np.isfinite(sa) & np.isfinite(sb)
    res = {'mode': mode, 'label': 'CLINICAL_EXTERNAL' if mode == 'clinical_external' else 'SYNTHETIC_POINT_ONLY_NOT_CLINICAL', 'admission': adm, 'flow_partition': flow, 'n_candidates': len(roster), 'eligible_labeled_denominator': len(L), 'reference_rule': R.RULE_ID, 'schema_provenance': state['schema']['provenance'], 'unknown_keys': {'labels_not_in_roster': extra_l, 'image_predictions_not_in_roster': ex_i, 'geometry_predictions_not_in_roster': ex_g, 'secondary_predictions_not_in_roster': ex_s}, 'completion_bounds': S26.completion_bounds(y, sa, sb, okp)}
    res['reference_missing_policy_counts'] = {'eligible_reference_missing': flow.get('eligible_reference_missing', 0)}
    if okp.any():
        A, n1, n0 = S26.auroc(y[okp], sa[okp])
        G, _, _ = S26.auroc(y[okp], sb[okp])
        res['primary'] = {'status': 'ok' if np.isfinite(A) else 'no_estimate_one_class', 'auroc_image_seed42': A, 'auroc_geometry_transported': G, 'difference': A - G if np.isfinite(A) else None, 'n_pos': n1, 'n_neg': n0, 'paired_support': int(okp.sum()), 'eligible_labeled_denominator': len(L), 'estimand': 'examination-weighted AUROC difference'}
    else:
        res['primary'] = {'status': 'no_estimate_no_paired_exams', 'paired_support': 0, 'eligible_labeled_denominator': len(L)}
    if adm['secondary']['status'] != 'admitted':
        res['secondary_seed123'] = {'status': 'unavailable', 'reason': adm['secondary']['reason']}
    else:
        v = np.array([sec.get(k, np.nan) for k in L])
        o2 = np.isfinite(v) & np.isfinite(sb)
        a2, m1, m0 = S26.auroc(y[o2], v[o2]) if o2.any() else (float('nan'), 0, 0)
        g2, _, _ = S26.auroc(y[o2], sb[o2]) if o2.any() else (float('nan'), 0, 0)
        res['secondary_seed123'] = {'status': 'descriptive_own_paired_subset', 'n_pos': m1, 'n_neg': m0, 'paired_support': int(o2.sum()), 'auroc_seed123': a2, 'auroc_geometry_on_this_set': g2, 'difference': a2 - g2 if np.isfinite(a2) and np.isfinite(g2) else None, 'overlap_with_primary_set': int((o2 & okp).sum()), 'secondary_only': int((o2 & ~okp).sum()), 'primary_only': int((~o2 & okp).sum())}
    gb = state['gb']
    pol = design['clinical_contract']['grouping_rule'] if mode == 'clinical_external' else design.get('grouping_policy', 'fallback_if_linkage_unavailable')
    kc = [k for k, o in zip(L, okp) if o]
    if gb:
        m = load_person_map(gb['map_path'])
        if any((not m.get(k) for k in kc)):
            res['grouping'] = {'status': 'linkage_incomplete', 'note': 'supplied linkage does not cover all paired exams; fallback NOT applied'}
            cl = None
        else:
            res['grouping'] = {'status': 'evidence_bound_linkage', 'evidence_locator': gb['evidence_locator'], 'map_sha256': gb['map_sha256'], 'caveat': 'structural binding only; semantics rest on the cited source'}
            cl = np.array([m[k] for k in kc])
    elif pol == 'fallback_if_linkage_unavailable':
        res['grouping'] = {'status': 'examination_cluster_fallback', 'caveat': 'repeat dependence unresolved; not a person-level interval'}
        cl = np.array(kc)
    else:
        res['grouping'] = {'status': 'no_linkage_and_fallback_not_permitted'}
        cl = None
    if mode == 'clinical_external':
        if cl is None or res['primary']['status'] != 'ok':
            res['bootstrap'] = {'status': 'not_run', 'reason': res['grouping']['status'] if cl is None else res['primary']['status']}
        else:
            r = P.CLIN_RECIPE
            res['bootstrap'] = BOOT(y[okp], sa[okp], sb[okp], cl, r['n_draws'], r['seed'])
            res['bootstrap']['recipe'] = r
    else:
        res['bootstrap'] = {'status': 'not_applicable_point_only_mode'}
    return res

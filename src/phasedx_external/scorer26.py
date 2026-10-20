"""Primary paired external scorer v26 (full-roster denominator, two-stage lock/receipt, strict keys, explicit grouping, deterministic completion bounds).
Stage 1 PROTOCOL_LOCK (before inference): intended use, models, preprocessing, aggregation, eligibility rule, label rule, cluster policy, bootstrap recipe.
Stage 2 PREDICTION_RECEIPT (after inference, before label join): hashes of roster, predictions and run identities. Labels are opened only after both verify."""
import os, sys, json, hashlib, csv
import numpy as np
RECIPE = {'n_draws': 2000, 'seed': 2026101327, 'percentile_method': 'linear', 'percentiles': [2.5, 97.5]}
LOCK_KEYS = ('protocol_id', 'locked_utc', 'mode', 'target', 'primary_image_model_id', 'geometry_model_id', 'checkpoint_sha256', 'geometry_model_sha256', 'recipe_sha256', 'precision_path', 'aggregation', 'eligibility_rule_id', 'series_policy_sha256', 'label_rule', 'cluster_policy', 'bootstrap', 'secondary_image_model_ids')

class ProtocolError(RuntimeError):
    pass

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)

def auroc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = (int(y.sum()), int((~y).sum()))
    if n1 == 0 or n0 == 0:
        return (float('nan'), n1, n0)
    r = ranks(np.asarray(s, float))
    return (float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1, n0)

def concordance_sum(y, s):
    """W = sum over positive-negative pairs of 1 (pos>neg), 1/2 (tie), 0; computed from ranks (equals direct pair sum)."""
    a, n1, n0 = auroc(y, s)
    return a * n1 * n0 if n1 and n0 else 0.0

def verify_lock(lock):
    for k in LOCK_KEYS:
        if k not in lock:
            raise ProtocolError('lock missing ' + k)
    if lock['mode'] == 'clinical_primary':
        b = lock['bootstrap']
        if (b.get('n_draws'), b.get('seed'), b.get('percentile_method')) != (RECIPE['n_draws'], RECIPE['seed'], RECIPE['percentile_method']):
            raise ProtocolError('clinical_primary mode requires the frozen bootstrap recipe (2000 draws, seed 2026101327, linear)')
        if lock['primary_image_model_id'] != 'I1_seed42':
            raise ProtocolError('primary image model must be seed42')
    elif lock['mode'] != 'synthetic_test':
        raise ProtocolError('unknown mode')
    return lock

def verify_receipt(lock_path, receipt_path, files):
    """files: dict name->path actually used. Receipt must bind lock hash and every file hash."""
    R = json.load(open(receipt_path))
    if R.get('protocol_lock_sha256') != sha(lock_path):
        raise ProtocolError('receipt does not bind this protocol lock')
    for k, p in files.items():
        if R.get('files', {}).get(k) != sha(p):
            raise ProtocolError('file changed or unbound since receipt: ' + k)
    if not R.get('receipt_utc') or not R.get('lock_utc_before_receipt_ok'):
        raise ProtocolError('receipt missing time ordering attestation')
    return R

def _read(p, key_cols):
    rows = list(csv.DictReader(open(p)))
    seen = set()
    for r in rows:
        k = tuple((r[c] for c in key_cols))
        if k in seen:
            raise ProtocolError('duplicate key %s in %s' % (k, os.path.basename(p)))
        seen.add(k)
    return rows

def completion_bounds(y, sa, sb, ok_pair):
    """y labels of ALL intended eligible labeled exams; sa/sb scores (nan where failed); ok_pair mask of paired complete subset."""
    y = np.asarray(y).astype(int)
    Np, Nn = (int((y == 1).sum()), int((y == 0).sum()))
    T = Np * Nn
    mp, mn = (int(((y == 1) & ok_pair).sum()), int(((y == 0) & ok_pair).sum()))
    E = mp * mn
    U = T - E
    out = {'N_pos': Np, 'N_neg': Nn, 'm_pos': mp, 'm_neg': mn, 'T': T, 'E': E, 'U': U, 'pair_coverage': E / T if T else None}
    if T == 0:
        out.update(status='undefined_T0', complete_case=None, W_image=None, W_geometry=None, bounds=None)
        return out
    Wa = concordance_sum(y[ok_pair], np.asarray(sa)[ok_pair])
    Wb = concordance_sum(y[ok_pair], np.asarray(sb)[ok_pair])
    out.update(status='ok', W_image=Wa, W_geometry=Wb)
    out['complete_case'] = {'auroc_image': Wa / E, 'auroc_geometry': Wb / E, 'difference': (Wa - Wb) / E} if E else {'auroc_image': None, 'auroc_geometry': None, 'difference': None, 'reason': 'E=0 no observed paired positive-negative pair'}
    out['bounds'] = {'image': [Wa / T, (Wa + U) / T], 'geometry': [Wb / T, (Wb + U) / T], 'difference': [(Wa - Wb - U) / T, (Wa - Wb + U) / T]}
    return out

def cluster_bootstrap(y, sa, sb, cluster, n_draw, seed):
    cl = np.asarray(cluster)
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    d = np.full(n_draw, np.nan)
    for b in range(n_draw):
        ii = np.concatenate([idx[c] for c in rng.choice(u, size=len(u), replace=True)])
        a1, _, _ = auroc(y[ii], sa[ii])
        a2, _, _ = auroc(y[ii], sb[ii])
        if np.isfinite(a1) and np.isfinite(a2):
            d[b] = a1 - a2
    v = d[np.isfinite(d)]
    r = {'n_draws': n_draw, 'n_undefined_draws': int(n_draw - len(v)), 'seed': seed, 'n_clusters': int(len(u)),
         'draws': [float(x) if np.isfinite(x) else None for x in d],
         'draw_encoding': 'ordered AUROC differences; null denotes undefined; no replacement draws',
         'draws_float64_le_sha256': hashlib.sha256(d.astype('<f8').tobytes()).hexdigest()}
    if len(v) == 0:
        r.update(status='no_interval_all_draws_undefined', lo=None, hi=None)
    else:
        r.update(status='ok', lo=float(np.quantile(v, 0.025, method='linear')), hi=float(np.quantile(v, 0.975, method='linear')))
    return r

def evaluate(lock_path, receipt_path, candidates_csv, image_pred_csv, geometry_pred_csv, labels_csv, person_map_csv=None, secondary_pred_csvs=None, n_draws_override=None):
    lock = verify_lock(json.load(open(lock_path)))
    files = {'candidates': candidates_csv, 'image_predictions': image_pred_csv, 'geometry_predictions': geometry_pred_csv}
    for i, p in enumerate(secondary_pred_csvs or []):
        files['secondary_%d' % i] = p
    verify_receipt(lock_path, receipt_path, files)
    nd = lock['bootstrap']['n_draws']
    if n_draws_override is not None:
        if lock['mode'] == 'clinical_primary':
            raise ProtocolError('draw override forbidden in clinical_primary mode')
        nd = n_draws_override
    cand = _read(candidates_csv, ['exam_key'])
    roster = [r['exam_key'] for r in cand]
    elig = {r['exam_key']: r['header_eligibility'] == 'eligible' for r in cand}

    def preds(p, mid):
        out = {}
        for r in _read(p, ['exam_key', 'model_id']):
            if r['model_id'] != mid:
                raise ProtocolError('unexpected model_id %s in %s' % (r['model_id'], os.path.basename(p)))
            ok = r.get('status', 'scored_complete') in ('scored', 'scored_complete') and r['score'] not in ('', 'nan', 'NaN') and np.isfinite(float(r['score']))
            out[r['exam_key']] = float(r['score']) if ok else float('nan')
        return out
    img = preds(image_pred_csv, lock['primary_image_model_id'])
    geo = preds(geometry_pred_csv, lock['geometry_model_id'])
    labs = {}
    for r in _read(labels_csv, ['exam_key']):
        if r['label'] == '':
            continue
        if r['label'] not in ('0', '1'):
            raise ProtocolError('non-binary label %r for %s' % (r['label'], r['exam_key']))
        labs[r['exam_key']] = int(r['label'])
    pmap = None
    if person_map_csv:
        pmap = {r['exam_key']: r['person_key'] for r in _read(person_map_csv, ['exam_key'])}
    flow = collections_counter()
    part = {}
    for k in roster:
        if not elig[k]:
            part[k] = 'header_ineligible'
        elif k not in labs:
            part[k] = 'eligible_unlabeled'
        else:
            fi = not np.isfinite(img.get(k, float('nan')))
            fg = not np.isfinite(geo.get(k, float('nan')))
            part[k] = 'labeled_both_fail' if fi and fg else 'labeled_image_fail' if fi else 'labeled_geometry_fail' if fg else 'labeled_paired_evaluable'
        flow[part[k]] += 1
    extra_labels = sorted(set(labs) - set(roster))
    extra_preds = sorted((set(img) | set(geo)) - set(roster))
    L = [k for k in roster if part[k].startswith('labeled')]
    y = np.array([labs[k] for k in L])
    sa = np.array([img.get(k, float('nan')) for k in L])
    sb = np.array([geo.get(k, float('nan')) for k in L])
    ok = np.isfinite(sa) & np.isfinite(sb)
    res = {'mode': lock['mode'], 'label': 'CLINICAL_PRIMARY_ANALYSIS' if lock['mode'] == 'clinical_primary' else 'SYNTHETIC_TEST_NOT_CLINICAL', 'n_candidates': len(roster), 'flow_partition': dict(flow), 'n_labels_not_in_roster': len(extra_labels), 'n_predictions_not_in_roster': len(extra_preds), 'failure_overlap': {'image_only': flow['labeled_image_fail'], 'geometry_only': flow['labeled_geometry_fail'], 'both': flow['labeled_both_fail']}}
    res['completion'] = completion_bounds(y, sa, sb, ok)
    if res['completion']['status'] != 'ok' or not ok.any():
        res['primary'] = {'status': 'no_estimate', 'reason': 'one_class_or_empty_labeled_population' if res['completion']['status'] != 'ok' else 'no_paired_evaluable'}
        return res
    yc, ac, bc, kc = (y[ok], sa[ok], sb[ok], [k for k, o in zip(L, ok) if o])
    A, n1, n0 = auroc(yc, ac)
    G, _, _ = auroc(yc, bc)
    if not np.isfinite(A):
        res['primary'] = {'status': 'no_estimate', 'reason': 'complete_case_one_class', 'n_pos': n1, 'n_neg': n0}
        return res
    pol = lock['cluster_policy']
    if pmap is not None:
        miss = [k for k in kc if not pmap.get(k)]
        if miss:
            res['primary'] = {'status': 'no_estimate', 'reason': 'person_map_missing_or_blank_for_%d_evaluable_exams' % len(miss)}
            return res
        cluster = np.array([pmap[k] for k in kc])
        grouping = 'verified_person_map'
    elif pol == 'examination_cluster_assumption_authorized':
        cluster = np.array(kc)
        grouping = 'examination_clusters_ASSUMED_independent_repeats_unknown'
    else:
        res['primary'] = {'status': 'no_estimate', 'reason': 'no_person_map_and_protocol_does_not_authorize_examination_clusters'}
        return res
    cnt = collections_counter(cluster.tolist())
    res['primary'] = {'status': 'ok', 'auroc_image': A, 'auroc_geometry': G, 'difference': A - G, 'n_evaluable': int(len(kc)), 'n_pos': n1, 'n_neg': n0, 'grouping': grouping, 'n_clusters': int(len(cnt)), 'clusters_with_multiple_exams': int(sum((v > 1 for v in cnt.values()))), 'estimand': 'examination-weighted AUROC difference', 'bootstrap': cluster_bootstrap(yc, ac, bc, cluster, nd, lock['bootstrap']['seed'])}
    sec = {}
    for i, p in enumerate(secondary_pred_csvs or []):
        mid = lock['secondary_image_model_ids'][i]
        s2 = preds(p, mid)
        v = np.array([s2.get(k, float('nan')) for k in L])
        o2 = np.isfinite(v) & np.isfinite(sb)
        a2, m1, m0 = auroc(y[o2], v[o2])
        g2, _, _ = auroc(y[o2], sb[o2])
        sec[mid] = {'role': 'descriptive_secondary', 'n_evaluable': int(o2.sum()), 'n_pos': m1, 'n_neg': m0, 'same_set_as_primary': bool(np.array_equal(o2, ok)), 'auroc': a2, 'geometry_auroc_on_this_set': g2, 'difference_on_this_set': a2 - g2 if np.isfinite(a2) else None, 'completion': completion_bounds(y, v, sb, o2)}
    if sec:
        res['secondary'] = sec
    return res

def collections_counter(it=None):
    import collections
    return collections.Counter(it) if it is not None else collections.Counter()

def make_receipt(lock_path, files, receipt_path, receipt_utc, lock_utc_before_receipt_ok=True):
    json.dump({'protocol_lock_sha256': sha(lock_path), 'files': {k: sha(p) for k, p in files.items()}, 'receipt_utc': receipt_utc, 'lock_utc_before_receipt_ok': lock_utc_before_receipt_ok}, open(receipt_path, 'w'), indent=1)

"""Geometry adapter v26 (original reconstructed G2_tree, 29 continuous + 5 missingness-indicator = 34 inputs; HistGradientBoosting; NOT refit here).
Header-only. Per-exam status row for EVERY roster exam. Representation rule (portability decision in geometry/PORTABILITY_DECISION.md):
  bin20 := rational_bin(rank_ascending_by_projected_position(series) , n-1)  [original: rational_bin(counter-lo, hi-lo) with counter = RSNA mirror filename index IMnnnnnn].
  Identical to the original iff the series' counter sequence is gapless and increases with position (the only case present in the original geometry fit set: 1727/1727 series verified_ascending, 0 gapped).
Orientation-invalid series -> status not_applicable_orientation_invalid (no score). Original imputation policy (median + missingness indicators for span/gap/near-coincident features) is applied only to features
that are undefined for a VALID series (e.g. single slice has no gaps)."""
import os, sys, json, math, pickle
from collections import Counter
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
from . import bgeom
ORTHO_TOL, ANGLE_DEG, NEAR_MM, B20 = (0.001, 1.0, 0.1, 20)
DEV_SUPPORT = {'n_slices': (24, 60), 'median_gap_mm': (2.25, 6.0236), 'span_mm': (67.5, 250.0)}

def rational_bin(a, b, B=B20):
    return min(B - 1, B * int(a) // int(b)) if b > 0 else 0

def verify_series(vals):
    n = len(vals)
    res = {'n_slices': n, 'orientation_valid': False, 'z': None, 'span_mm': None, 'gaps_mm': None, 'reason': None}
    if n == 0:
        res['reason'] = 'empty'
        return res
    if not np.isfinite(vals).all():
        res['reason'] = 'non-finite IPP/IOP'
        return res
    cnts = Counter((tuple(v) for v in vals[:, 3:]))
    top = max(cnts.values())
    modal = min((t for t, c in cnts.items() if c == top))
    nrm = lambda iop: (np.asarray(iop[:3], float), np.asarray(iop[3:], float), np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float)))
    n0 = nrm(modal)[2]
    nref = n0 / np.linalg.norm(n0)
    for v in vals[:, 3:]:
        r, c, nn = nrm(v)
        if abs(np.linalg.norm(r) - 1) > ORTHO_TOL or abs(np.linalg.norm(c) - 1) > ORTHO_TOL or abs(float(r @ c)) > ORTHO_TOL:
            res['reason'] = 'direction cosines not unit/orthogonal'
            return res
    for v in vals[:, 3:]:
        nn = nrm(v)[2]
        if math.degrees(math.acos(float(np.clip(abs(nn @ nref) / np.linalg.norm(nn), -1, 1)))) > ANGLE_DEG:
            res['reason'] = 'slice normal > 1 degree from reference'
            return res
    for v in vals[:, 3:]:
        if float(nrm(v)[2] @ nref) < 0:
            res['reason'] = 'slice normal reversed relative to reference'
            return res
    z = vals[:, :3] @ nref
    zs = np.sort(z)
    res.update(orientation_valid=True, z=z, span_mm=float(zs[-1] - zs[0]) if n > 1 else 0.0, gaps_mm=np.diff(zs).tolist())
    return res

def series_row(series_key, exam_key, g):
    """g: DataFrame rows of ONE series with ipp0..2, iop0..5 and optional instance_number, sop. Returns (series_dict, image_rows)."""
    g = g.reset_index(drop=True)
    vals = g[[f'ipp{i}' for i in range(3)] + [f'iop{i}' for i in range(6)]].to_numpy(float)
    r = verify_series(vals)
    n = len(g)
    inst = pd.to_numeric(g['instance_number'], errors='coerce').to_numpy(float) if 'instance_number' in g else np.full(n, np.nan)
    d = {'SeriesInstanceUID': series_key, 'StudyInstanceUID': exam_key, 'n_slices': n, 'orientation_valid': r['orientation_valid'], 'span_mm': r['span_mm'], 'gaps_mm': json.dumps(r['gaps_mm']) if r['gaps_mm'] is not None else None, 'orientation_reason': r['reason']}
    flags = []
    if np.isfinite(inst).all() and n > 0:
        d['instance_number_gaps'] = int(inst.max() - inst.min() + 1 - len(set(inst.tolist())))
        if r['orientation_valid'] and n > 1:
            o = np.argsort(inst, kind='stable')
            dz = np.diff(r['z'][o])
            d['instance_order_vs_position'] = 'ascending' if (dz >= -NEAR_MM).all() else 'descending' if (dz <= NEAR_MM).all() else 'mixed'
        else:
            d['instance_order_vs_position'] = 'na'
    else:
        d['instance_number_gaps'] = None
        d['instance_order_vs_position'] = 'unavailable'
    if r['orientation_valid']:
        tie = np.nan_to_num(inst, nan=0.0)
        order = np.lexsort((tie, r['z']))
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        d['duplicate_positions'] = int(sum((1 for x in np.diff(np.sort(r['z'])) if x <= NEAR_MM)))
    else:
        rank = np.arange(n)
        d['duplicate_positions'] = None
    im = pd.DataFrame({'StudyInstanceUID': exam_key, 'SeriesInstanceUID': series_key, 'bin20': [rational_bin(a, n - 1) for a in rank]})
    gp = [x for x in r['gaps_mm'] or [] if x > 0]
    d['median_gap_mm'] = float(np.median(gp)) if gp else None
    for k, (lo, hi) in DEV_SUPPORT.items():
        v = d.get(k)
        if v is not None and (not lo <= v <= hi):
            flags.append('out_of_dev_support:' + k)
    d['support_flags'] = ';'.join(flags)
    return (d, im)

def load_model(pkl, sha=None):
    import hashlib
    if sha and hashlib.sha256(open(pkl, 'rb').read()).hexdigest() != sha:
        raise RuntimeError('geometry model sha mismatch')
    return pickle.load(open(pkl, 'rb'))

def predict_external(model, headers, roster_keys, selected_series, reject_instance_number_gaps=False):
    """headers: instance table (exam_key, series_key, ipp*, iop*, instance_number[, sop]); selected_series: dict exam_key -> series_key (from the shared eligibility manifest; absent/None => no eligible series).
    Returns DataFrame: exam_key, model_id, n_eligible_slices, score, status, failure_reason, support_flags, instance_number_gaps, instance_order_vs_position."""
    imp = bgeom.Imputer()
    imp.med = model['imputer_medians']
    out = []
    for ek in roster_keys:
        sk = selected_series.get(ek)
        base = {'exam_key': ek, 'model_id': 'G2_tree_orig_frozen', 'n_eligible_slices': 0, 'score': float('nan'), 'status': '', 'failure_reason': '', 'support_flags': '', 'instance_number_gaps': None, 'instance_order_vs_position': '', 'representation_status': 'unusable', 'config_class': ''}
        if not sk:
            out.append({**base, 'config_class': 'no_selected_series', 'status': 'not_applicable_no_selected_series', 'failure_reason': 'no series passed the frozen header policy'})
            continue
        g = headers[(headers['exam_key'] == ek) & (headers['series_key'] == sk)]
        if len(g) == 0:
            out.append({**base, 'status': 'not_applicable_no_headers', 'failure_reason': 'selected series has no header rows'})
            continue
        d, im = series_row(sk, ek, g)
        base.update(n_eligible_slices=int(d['n_slices']), support_flags=d['support_flags'], instance_number_gaps=d['instance_number_gaps'], instance_order_vs_position=d['instance_order_vs_position'])
        if not d['orientation_valid']:
            out.append({**base, 'config_class': 'orientation_invalid', 'status': 'not_applicable_orientation_invalid', 'failure_reason': str(d['orientation_reason'])})
            continue
        if reject_instance_number_gaps and d['instance_number_gaps']:
            out.append({**base, 'status': 'not_applicable_instance_number_gaps', 'failure_reason': 'gaps=%d' % d['instance_number_gaps']})
            continue
        X = bgeom.study_features(im, pd.DataFrame([{k: d[k] for k in ('SeriesInstanceUID', 'orientation_valid', 'span_mm', 'gaps_mm')}]))
        Z = imp.transform(X)
        cc = []
        if d['instance_order_vs_position'] in ('descending', 'mixed', 'unavailable'):
            cc.append('order_' + d['instance_order_vs_position'])
        if d['instance_number_gaps']:
            cc.append('instance_number_gaps')
        if d['duplicate_positions']:
            cc.append('duplicate_positions')
        if d['support_flags']:
            cc.append(d['support_flags'].replace('out_of_dev_support:', 'out_of_support_'))
        cls = ';'.join(cc) if cc else 'development_domain_configuration'
        out.append({**base, 'score': float(model['hgb'].decision_function(Z.to_numpy())[0]), 'status': 'scored', 'representation_status': 'transported_physical_rank_not_original_equivalent_verified', 'config_class': cls})
    return pd.DataFrame(out)

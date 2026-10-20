"""Lane B (Batch 03) geometry comparators under laneB_design_v1: G0, G1 (rational B20 mean/max), G2 (logistic,
HistGradientBoosting) on study geometry only; selection of G* on tune_select studies.

Inputs are the bdata tables (images with bin20/label_any/partition/sets; series with physical per-series
summaries). Every function is pure (no I/O) so it can be unit-tested on synthetic tables."""
from __future__ import annotations
import json
from fractions import Fraction
import numpy as np
import pandas as pd
from . import bstats
B20 = 20
NEAR_MM = 0.1
CONT_COLS = ['log1p_n_slices', 'n_series', 'series_count_median', 'series_count_iqr', 'span_median_mm', 'span_iqr_mm', 'gap_median_mm', 'gap_iqr_mm', 'near_coincident_fraction'] + [f'occ_{k:02d}' for k in range(B20)]
MISS_FEATS = ['span_median_mm', 'span_iqr_mm', 'gap_median_mm', 'gap_iqr_mm', 'near_coincident_fraction']
IND_COLS = [f'miss_{c}' for c in MISS_FEATS]
ALL_COLS = CONT_COLS + IND_COLS
CANDIDATES = ['G1_mean', 'G1_max', 'G2_logistic', 'G2_tree']

def _iqr(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 75) - np.percentile(v, 25))

def study_features(images, series):
    """images: rows of the studies to featurise (StudyInstanceUID, SeriesInstanceUID, bin20).
    series: per-series table with orientation_valid, span_mm, gaps_mm (json list), n_slices.
    Returns DataFrame indexed by StudyInstanceUID with CONT_COLS (NaN = missing before imputation)."""
    ser = series.set_index('SeriesInstanceUID')
    rows = {}
    for st, g in images.groupby('StudyInstanceUID', sort=True):
        sids = sorted(g['SeriesInstanceUID'].unique())
        counts = g.groupby('SeriesInstanceUID').size().reindex(sids).to_numpy()
        n = len(g)
        f = {'log1p_n_slices': float(np.log1p(n)), 'n_series': float(len(sids)), 'series_count_median': float(np.median(counts)), 'series_count_iqr': _iqr(counts) if len(counts) > 1 else 0.0}
        spans, pos_gaps, all_gaps = ([], [], [])
        for s in sids:
            r = ser.loc[s]
            if not bool(r['orientation_valid']):
                continue
            spans.append(float(r['span_mm']))
            gl = json.loads(r['gaps_mm']) if isinstance(r['gaps_mm'], str) else []
            all_gaps.extend(gl)
            pos_gaps.extend([x for x in gl if x > 0.0])
        f['span_median_mm'] = float(np.median(spans)) if spans else np.nan
        f['span_iqr_mm'] = (_iqr(spans) if len(spans) > 1 else 0.0) if spans else np.nan
        f['gap_median_mm'] = float(np.median(pos_gaps)) if pos_gaps else np.nan
        f['gap_iqr_mm'] = (_iqr(pos_gaps) if len(pos_gaps) > 1 else 0.0) if pos_gaps else np.nan
        f['near_coincident_fraction'] = float(np.sum([0.0 <= x <= NEAR_MM for x in all_gaps])) / len(all_gaps) if all_gaps else np.nan
        occ = np.bincount(g['bin20'].astype(int).to_numpy(), minlength=B20)[:B20]
        for k in range(B20):
            f[f'occ_{k:02d}'] = float(occ[k]) / n
        rows[st] = f
    return pd.DataFrame.from_dict(rows, orient='index')[CONT_COLS]

class Imputer:
    """Fit-study median imputation + missingness indicators (frozen column list)."""

    def fit(self, X):
        self.med = {}
        self.all_missing = []
        for c in CONT_COLS:
            v = X[c].to_numpy(float)
            ok = np.isfinite(v)
            if ok.any():
                self.med[c] = float(np.median(v[ok]))
            else:
                self.med[c] = 0.0
                self.all_missing.append(c)
        return self

    def transform(self, X):
        Z = pd.DataFrame(index=X.index)
        for c in MISS_FEATS:
            Z[f'miss_{c}'] = (~np.isfinite(X[c].to_numpy(float))).astype(float)
        for c in CONT_COLS:
            v = X[c].to_numpy(float).copy()
            v[~np.isfinite(v)] = self.med[c]
            Z[c] = v
        return Z[ALL_COLS]

class Standardiser:

    def fit(self, Z):
        self.mu = Z[CONT_COLS].mean(axis=0).to_numpy()
        sd = Z[CONT_COLS].std(axis=0, ddof=0).to_numpy()
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, Z):
        A = Z.copy()
        A[CONT_COLS] = (Z[CONT_COLS].to_numpy() - self.mu) / self.sd
        return A[ALL_COLS]

def fit_g1(fit_images):
    """Exact per-bin Fraction rates from fit slices (official any label); unseen bin -> fit slice prevalence."""
    y = fit_images['label_any'].astype(int).to_numpy()
    b = fit_images['bin20'].astype(int).to_numpy()
    pos = np.bincount(b[y == 1], minlength=B20)
    tot = np.bincount(b, minlength=B20)
    prev = Fraction(int(y.sum()), int(len(y)))
    rates = [Fraction(int(pos[k]), int(tot[k])) if tot[k] > 0 else prev for k in range(B20)]
    return {'rates': rates, 'pos': pos.tolist(), 'tot': tot.tolist(), 'fallback': prev, 'unseen_bins': [k for k in range(B20) if tot[k] == 0]}

def score_g1(images, g1):
    """Return DataFrame per study: G1_mean, G1_max as Fractions + count of slices scored by fallback."""
    unseen = set(g1['unseen_bins'])
    out = {}
    for st, g in images.groupby('StudyInstanceUID', sort=True):
        r = [g1['rates'][int(k)] for k in g['bin20']]
        out[st] = {'G1_mean': sum(r, Fraction(0)) / len(r), 'G1_max': max(r), 'n_fallback_slices': int(sum((int(k) in unseen for k in g['bin20'])))}
    return pd.DataFrame.from_dict(out, orient='index')

def study_labels(images):
    return images.groupby('StudyInstanceUID', sort=True).agg(label=('label_any', 'max'), patient_id=('patient_id', 'first'), rank=('rank', 'first'), partition=('partition', 'first'), n_slices=('image_id', 'size'))

def fit_all(images, series):
    """images: assembled image table. Fits on geometry_fit_set studies; scores analysis_set studies.
    Returns dict with study score table (Fractions for G1 kept as str), fit diagnostics, selection."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    fit_im = images[images['geometry_fit_set']]
    ana_im = images[images['analysis_set']]
    lab_fit = study_labels(fit_im)
    lab_ana = study_labels(ana_im)
    X_fit = study_features(fit_im, series).loc[lab_fit.index]
    X_ana = study_features(ana_im, series).loc[lab_ana.index]
    imp = Imputer().fit(X_fit)
    Z_fit, Z_ana = (imp.transform(X_fit), imp.transform(X_ana))
    std = Standardiser().fit(Z_fit)
    y_fit = lab_fit['label'].astype(int).to_numpy()
    diag = {'n_fit_studies': int(len(y_fit)), 'n_fit_pos_studies': int(y_fit.sum()), 'n_fit_slices': int(len(fit_im)), 'imputation_medians': imp.med, 'all_missing_in_fit': imp.all_missing, 'missing_counts_fit': {c: int(np.sum(~np.isfinite(X_fit[c].to_numpy(float)))) for c in MISS_FEATS}, 'missing_counts_analysis': {c: int(np.sum(~np.isfinite(X_ana[c].to_numpy(float)))) for c in MISS_FEATS}, 'feature_columns_frozen': ALL_COLS}
    scores = pd.DataFrame(index=lab_ana.index)
    scores['G0'] = float(y_fit.mean()) if len(y_fit) else np.nan
    g1 = fit_g1(fit_im)
    s1 = score_g1(ana_im, g1).loc[lab_ana.index]
    scores['G1_mean_frac'] = s1['G1_mean']
    scores['G1_max_frac'] = s1['G1_max']
    scores['G1_n_fallback_slices'] = s1['n_fallback_slices']
    diag['G1'] = {'pos_by_bin': g1['pos'], 'tot_by_bin': g1['tot'], 'unseen_bins': g1['unseen_bins'], 'fallback_rate': str(g1['fallback']), 'analysis_slices_scored_by_fallback': int(s1['n_fallback_slices'].sum())}
    if len(set(y_fit.tolist())) == 2:
        lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000, tol=0.0001, fit_intercept=True)
        lr.fit(std.transform(Z_fit).to_numpy(), y_fit)
        scores['G2_logistic'] = lr.decision_function(std.transform(Z_ana).to_numpy())
        hgb = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, max_depth=3, min_samples_leaf=20, learning_rate=0.1, l2_regularization=1.0, early_stopping=False, random_state=42)
        hgb.fit(Z_fit.to_numpy(), y_fit)
        scores['G2_tree'] = hgb.decision_function(Z_ana.to_numpy())
        diag['G2_logistic'] = {'n_iter': int(lr.n_iter_[0]), 'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'converged': bool(lr.n_iter_[0] < 5000)}
        diag['G2_tree'] = {'n_iter': int(hgb.n_iter_)}
    else:
        scores['G2_logistic'] = np.nan
        scores['G2_tree'] = np.nan
        diag['G2_unavailable'] = 'fit studies have one class'
    scores['G1_mean'] = bstats.dense_rank(list(scores['G1_mean_frac'])).astype(float)
    scores['G1_max'] = bstats.dense_rank(list(scores['G1_max_frac'])).astype(float)
    scores = scores.join(lab_ana)
    sel = scores[scores['partition'] == 'tune_select']
    sel_auc = {}
    for c in CANDIDATES:
        if sel[c].isna().any():
            sel_auc[c] = {'auc': None, 'reason': 'candidate unavailable'}
        else:
            sel_auc[c] = bstats.auroc(sel['label'].to_numpy(), sel[c].to_numpy())
    one_class = sel['label'].nunique() < 2
    if one_class:
        gstar = None
        sel_reason = 'selection partition has one study class: G* unavailable; comparative pilot unavailable'
    else:
        gstar = bstats.select_with_ties({c: sel_auc[c]['auc'] for c in CANDIDATES}, CANDIDATES, 1e-12)
        sel_reason = None
    selection = {'criterion': 'study AUROC on tune_select studies', 'candidates_order': CANDIDATES, 'auroc': {c: sel_auc[c] for c in CANDIDATES}, 'G_star': gstar, 'unavailable_reason': sel_reason, 'n_selection_studies': int(len(sel)), 'n_selection_pos': int(sel['label'].sum()), 'n_selection_patients': int(sel['patient_id'].nunique())}
    scores['G1_mean_frac'] = scores['G1_mean_frac'].map(str)
    scores['G1_max_frac'] = scores['G1_max_frac'].map(str)
    return {'scores': scores, 'diagnostics': diag, 'selection': selection}

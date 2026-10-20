"""WS3 controlled transport experiment: one shard = one (label_mode, n_eval, regime-set?) configuration index list given by config. Spec frozen in LOCK25."""
import sys, os, json, hashlib, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from prospective_guard import require_registration
CFG = sys.argv[1]; require_registration(__file__, CFG)
import numpy as np
from sim25 import auroc, auroc_within, sigmoid
B = os.path.dirname(HERE); cfg = json.load(open(CFG))
lock = json.load(open(B + '/state/LOCK25.json'))
for k, v in lock['hashes'].items():
    assert hashlib.sha256(open(os.path.join(B, k), 'rb').read()).hexdigest() == v, 'changed after lock: ' + k
MASTER = 2026101325; NSLICE = 20; Z = (np.arange(NSLICE) + 0.5) / NSLICE
ARMS = {'null_b0_g0': (0, 0), 'path_b2_g0': (2, 0), 'nuis_b0_g2': (0, 2), 'mixed_b2_g2': (2, 2)}
PCOND = {'source': (0.2, 0.8), 'unchanged': (0.2, 0.8), 'decorrelated': (0.5, 0.5), 'reversed': (0.8, 0.2)}
LOC = (np.arange(NSLICE) >= 7) & (np.arange(NSLICE) <= 12)
def rng(*key): return np.random.default_rng(np.random.SeedSequence(MASTER, spawn_key=tuple(int(k) for k in key)))
def make_pop(n, pc, r):
    C = (r.random(n) < 0.5).astype(int); py = np.where(C == 0, pc[0], pc[1]); Y = (r.random(n) < py).astype(int)
    U = r.normal(0, 0.5, n); E = r.normal(0, 1, (n, NSLICE)); return C, Y, U, E
def slice_labels(Y, mode): return np.repeat(Y[:, None], NSLICE, 1) if mode == 'broadcast' else (Y[:, None] * LOC[None, :]).astype(int)
def fit_geom(C, L):
    rate = np.zeros((2, NSLICE))
    for c in (0, 1):
        m = C == c; rate[c] = (L[m].sum(0) + 1) / (m.sum() + 2)   # per (C,bin): (pos slices +1)/(all slices +2)
    return rate
def evaluate(C, Y, U, E, mode, rate):
    L = slice_labels(Y, mode); out = {}
    scores = {'geometry': rate[C][:, :]}
    for nm, (b, g) in ARMS.items(): scores[nm] = sigmoid(b * (2 * L - 1) + g * (2 * C[:, None] - 1) + U[:, None] + E)
    Cs = np.repeat(C[:, None], NSLICE, 1).ravel()
    for nm, S in scores.items():
        a_sl, n1, n0 = auroc(L.ravel(), S.ravel()); a_mean, m1, m0 = auroc(Y, S.mean(1)); a_max, _, _ = auroc(Y, S.max(1))
        w, wp = auroc_within(L.ravel(), S.ravel(), Cs)
        out[nm] = dict(slice=a_sl, mean=a_mean, max=a_max, withinC_slice=w, withinC_pairs=wp, n_pos_slices=n1, n_neg_slices=n0, n_pos_subjects=m1, n_neg_subjects=m0)
    return out
rows = []
LM = {'broadcast': 0, 'localized': 1}; RG = {'unchanged': 0, 'decorrelated': 1, 'reversed': 2}
for conf in cfg['configs']:
    mode, n, reg = conf['label_mode'], conf['n_eval'], conf['regime']; nidx = {75: 0, 300: 1}[n]
    for rep in range(cfg['n_rep']):
        rd = rng(1, LM[mode], rep); Cd, Yd, Ud, Ed = make_pop(800, PCOND['source'], rd)
        rate = fit_geom(Cd, slice_labels(Yd, mode))                      # development realization shared across target regimes and n within a replicate
        ri = rng(2, LM[mode], nidx, rep); popi = make_pop(n, PCOND['source'], ri)
        rt = rng(3, LM[mode], nidx, RG[reg], rep); popt = make_pop(n, PCOND[reg], rt)
        for popn, pop in (('internal', popi), ('target', popt)):
            res = evaluate(*pop, mode, rate)
            for arm, d in res.items(): rows.append({'label_mode': mode, 'n_eval': n, 'regime': reg, 'rep': rep, 'population': popn, 'arm': arm, **d})
os.makedirs(B + '/transport/shards', exist_ok=True)
with open(B + f"/transport/shards/transport_shard_{cfg['shard']}.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print(cfg['shard'], len(rows))

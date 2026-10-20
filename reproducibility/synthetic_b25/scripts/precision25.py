"""WS4B precision-feasibility binormal experiment: shard = list of scenario indices. Spec frozen in LOCK25."""
import sys, os, json, hashlib, csv, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from prospective_guard import require_registration
CFG = sys.argv[1]; require_registration(__file__, CFG)
import numpy as np
from scipy.stats import norm
from sim25 import auroc
B = os.path.dirname(HERE); cfg = json.load(open(CFG)); lock = json.load(open(B + '/state/LOCK25.json'))
for k, v in lock['hashes'].items():
    assert hashlib.sha256(open(os.path.join(B, k), 'rb').read()).hexdigest() == v, 'changed after lock: ' + k
MASTER = 2026101326; NREP = 500
GRID = list(itertools.product((75, 250, 491), (0.2, 0.4, 0.6), (0.60, 0.75), (0.0, 0.05, 0.10), (0.25, 0.75)))   # N, p, A_g, delta, rho -> 108 scenarios in this fixed order
assert len(GRID) == 108
rows = []
for si in cfg['scenarios']:
    N, p, Ag, dl, rho = GRID[si]; npos = int(round(N * p)); nneg = N - npos
    mu_g, mu_i = np.sqrt(2) * norm.ppf(Ag), np.sqrt(2) * norm.ppf(Ag + dl)
    cov = np.array([[1, rho], [rho, 1]]); Lc = np.linalg.cholesky(cov)
    ctr = []; undefined = 0
    for rep in range(NREP):
        r = np.random.default_rng(np.random.SeedSequence(MASTER, spawn_key=(si, rep)))
        neg = r.standard_normal((nneg, 2)) @ Lc.T; pos = r.standard_normal((npos, 2)) @ Lc.T + np.array([mu_g, mu_i])
        y = np.r_[np.zeros(nneg), np.ones(npos)]; S = np.vstack([neg, pos])
        ag, _, _ = auroc(y, S[:, 0]); ai, _, _ = auroc(y, S[:, 1])
        if not (np.isfinite(ag) and np.isfinite(ai)): undefined += 1; continue
        ctr.append(ai - ag)
    c = np.array(ctr); err = c - dl
    rows.append({'scenario': si, 'N': N, 'p_pos': p, 'n_pos': npos, 'n_neg': nneg, 'A_geom': Ag, 'delta': dl, 'rho': rho, 'n_rep_valid': len(c), 'n_undefined': undefined, 'mean_contrast': c.mean(), 'sd_contrast': c.std(ddof=1), 'bias_vs_delta': c.mean() - dl,
                 'mc_se_mean': c.std(ddof=1) / np.sqrt(len(c)), 'err_q025': np.quantile(err, 0.025), 'err_q975': np.quantile(err, 0.975), 'normal_proxy_1p96_sd': 1.96 * c.std(ddof=1)})
os.makedirs(B + '/precision/shards', exist_ok=True)
with open(B + f"/precision/shards/precision_shard_{cfg['shard']}.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print(cfg['shard'], len(rows))

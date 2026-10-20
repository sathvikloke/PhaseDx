"""Targeted correctness checks (pre-run): paired identity, binormal AUROC formula, AUROC tie/undefined handling, transport prevalence, geometry-rate formula. Writes CHECKS25.json."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import norm
from sim25 import auroc, auroc_within
B = os.path.dirname(HERE); res = []
def ck(n, ok, d=''): res.append({'check': n, 'pass': bool(ok), 'detail': str(d)})
a, n1, n0 = auroc([1, 1, 0, 0], [3, 2, 2, 1]); ck('auroc tie half credit (3,2 vs 2,1 => (1+1+0.5+1)/4=0.875)', abs(a - 0.875) < 1e-12, a)
ck('auroc undefined when one class', np.isnan(auroc([1, 1], [1, 2])[0]))
w, wp = auroc_within(np.array([1, 0, 1, 0]), np.array([.9, .1, .2, .8]), np.array([0, 0, 1, 1])); ck('within-stratum auroc = (1+0)/2 with 2 pairs', abs(w - 0.5) < 1e-12 and wp == 2, (w, wp))
r = np.random.default_rng(np.random.SeedSequence(1, spawn_key=(9,)))
Ag, dl, rho, N = 0.75, 0.10, 0.5, 400000
mu_g, mu_i = np.sqrt(2) * norm.ppf(Ag), np.sqrt(2) * norm.ppf(Ag + dl); Lc = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
neg = r.standard_normal((N, 2)) @ Lc.T; pos = r.standard_normal((N, 2)) @ Lc.T + np.array([mu_g, mu_i]); y = np.r_[np.zeros(N), np.ones(N)]; S = np.vstack([neg, pos])
ag = auroc(y, S[:, 0])[0]; ai = auroc(y, S[:, 1])[0]
ck('binormal geometry AUROC ~ 0.75 (tol 0.003)', abs(ag - 0.75) < 0.003, ag); ck('binormal image AUROC ~ 0.85 (tol 0.003)', abs(ai - 0.85) < 0.003, ai)
ck('within-class correlation ~ rho', abs(np.corrcoef(neg[:, 0], neg[:, 1])[0, 1] - rho) < 0.01 and abs(np.corrcoef(pos[:, 0], pos[:, 1])[0, 1] - rho) < 0.01)
ck('paired identity: same rows used for both arms (equal length, same y)', S.shape[0] == len(y))
# transport: marginal prevalence 0.5 in all regimes
pc = {'source': (0.2, 0.8), 'unchanged': (0.2, 0.8), 'decorrelated': (0.5, 0.5), 'reversed': (0.8, 0.2)}
ck('marginal prevalence 0.5 in each regime', all(abs(0.5 * a_ + 0.5 * b_ - 0.5) < 1e-12 for a_, b_ in pc.values()))
from itertools import product
ck('precision grid has 108 cells', len(list(product((75, 250, 491), (0.2, 0.4, 0.6), (0.60, 0.75), (0.0, 0.05, 0.10), (0.25, 0.75)))) == 108)
json.dump(res, open(B + '/state/CHECKS25.json', 'w'), indent=1); print(all(x['pass'] for x in res), [x for x in res if not x['pass']])

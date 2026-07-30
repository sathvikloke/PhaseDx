"""
BUNDLED unit+metric scan.

Real detection benchmarks report a fine-unit metric and a coarse-unit metric side by side
(lesion AP vs patient AUROC in PI-CAI; lesion FROC vs slide AUC in CAMELYON16). The unit
change is CONFOUNDED with a metric change, so this is weaker evidence than a same-metric
comparison -- but it is how the benchmarks that actually drive method choice are scored,
so it must be scanned separately and reported with the confound stated.

Accepts a fine-unit column and a coarse-unit column even when the metric differs.
"""
import re, html, os, json, glob, itertools, sys

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
     'scan_single_table.py')).read().split('results = []')[0])

FINE_METRICS   = ['ap','average precision','froc','cpm','dice','iou','sensitivity','recall']
COARSE_METRICS = ['auroc','auc','accuracy','specificity','kappa']

results = []
CORPORA = sys.argv[1:] or ['xml','xml2','xml3']
files = []
for d in CORPORA:
    files += sorted(glob.glob(os.path.join(d,'*.xml')))
stat = dict(scanned=0, cand_tables=0)
for p in files:
    if os.path.getsize(p) < 2000: continue
    stat['scanned'] += 1
    pmcid = os.path.basename(p)[:-4]
    try: tbs = get_tables(p)
    except Exception: continue
    for tb in tbs:
        H, B = tb['H'], tb['B']
        if not H or not B: continue
        ncol = max(len(r) for r in H)
        collab = []
        for c in range(ncol):
            parts = []
            for r in H:
                if c < len(r) and r[c] and (not parts or parts[-1] != r[c]): parts.append(r[c])
            collab.append(' '.join(parts))
        fine_cols, coarse_cols = {}, {}
        for c, lab in enumerate(collab):
            u, mt = unit_of(lab), metric_of(lab)
            if not mt: continue
            if u == 'fine'   and mt in FINE_METRICS:   fine_cols.setdefault(mt, c)
            if u == 'coarse' and mt in COARSE_METRICS: coarse_cols.setdefault(mt, c)
        if not fine_cols or not coarse_cols: continue
        stat['cand_tables'] += 1
        data = {}
        for row in B:
            if not row: continue
            name = row[0].strip()
            if not name or not MODELWORD.search(name): continue
            vals = {}
            for mt, ci in fine_cols.items():
                if ci < len(row):
                    v = num(row[ci])
                    if v is not None: vals[('fine', mt)] = v
            for mt, ci in coarse_cols.items():
                if ci < len(row):
                    v = num(row[ci])
                    if v is not None: vals[('coarse', mt)] = v
            if vals: data[name] = vals
        if len(data) < 2: continue
        for fm in fine_cols:
            for cm in coarse_cols:
                pairs = [(k, v[('fine',fm)], v[('coarse',cm)]) for k, v in data.items()
                         if ('fine',fm) in v and ('coarse',cm) in v]
                if len(pairs) < 3: continue
                fv=[x[1] for x in pairs]; cv=[x[2] for x in pairs]
                if len(set(fv))<2 or len(set(cv))<2: continue
                disc = sum(1 for a,b in itertools.combinations(pairs,2) if (a[1]-b[1])*(a[2]-b[2])<0)
                tot = len(pairs)*(len(pairs)-1)//2
                fb = max(pairs,key=lambda x:x[1]); cb = max(pairs,key=lambda x:x[2])
                if disc > 0:
                    results.append(dict(pmcid=pmcid, table=tb['label'], caption=tb['caption'][:220],
                        fine_metric=fm, coarse_metric=cm, n=len(pairs), disc=disc, tot=tot,
                        tau=round((tot-2*disc)/tot,3),
                        top_changed=fb[0]!=cb[0],
                        top_changed_strict=(fb[0]!=cb[0] and fv.count(max(fv))==1 and cv.count(max(cv))==1),
                        fine_best=(fb[0],fb[1]), coarse_best=(cb[0],cb[2]), pairs=pairs,
                        collab=[collab[fine_cols[fm]], collab[coarse_cols[cm]]]))
json.dump(results, open('scan_bundled_out.json','w'), indent=1)
print("@@@", stat)
print("@@@ rows:", len(results), "papers:", len({r['pmcid'] for r in results}),
      "strict-top1 papers:", len({r['pmcid'] for r in results if r['top_changed_strict']}))
for r in sorted(results, key=lambda x: x['tau']):
    if r['top_changed_strict'] and r['n'] >= 4:
        print("@@@ HIT", r['pmcid'], r['table'], r['fine_metric'], 'vs', r['coarse_metric'],
              'n=%d disc=%d/%d tau=%.3f' % (r['n'], r['disc'], r['tot'], r['tau']))
        print("@@@    ", r['collab'])
        print("@@@     fine_best", r['fine_best'], "coarse_best", r['coarse_best'])

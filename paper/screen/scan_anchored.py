"""
BENCHMARK-ANCHORED scan.

Comparison tables for detection benchmarks label columns with the bare metric name
("AP", "AUROC", "FROC", "AUC") and define the evaluation unit only in the methods text.
A header-driven scan therefore misses them. Here the unit assignment comes from the
benchmark's own published metric definition:

  PI-CAI     : "Lesion-level detection performance is evaluated using the Average
                Precision (AP) metric."  /  "Patient-level diagnosis performance is
                evaluated using the Area Under Receiver Operating Characteristic
                (AUROC) metric."          -> AP = fine (lesion), AUROC = coarse (patient)
                source: https://pi-cai.grand-challenge.org/AI/
  CAMELYON16 : FROC = average sensitivity at 6 FP rates per WSI (lesion localisation)
                AUC  = whole-slide-image classification
                source: https://camelyon16.grand-challenge.org/Results/
                                          -> FROC = fine (lesion), AUC = coarse (slide)

Only applied to papers that actually mention the benchmark, so the unit assignment is
licensed by that benchmark's definition.
"""
import re, html, os, json, glob, itertools, sys

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
     'scan_single_table.py')).read().split('results = []')[0])

def col_metric_exact(lab):
    """bare metric name in a column header, unit words NOT required"""
    l = re.sub(r'\s+', ' ', lab.lower()).strip()
    l = re.sub(r'[\(\)\[\]%↑↓*†‡]', ' ', l)
    l = re.sub(r'\s+', ' ', l).strip()
    if re.fullmatch(r'(mean )?(ap|average precision)( score)?', l): return 'AP'
    if re.fullmatch(r'(mean )?(auroc|auc|auc roc|au roc)( score)?', l): return 'AUROC'
    if re.fullmatch(r'(mean )?(froc|froc score|cpm)( score)?', l): return 'FROC'
    return None

BENCH = {
 'PI-CAI':     (re.compile(r'PI-?CAI', re.I),      'AP',   'AUROC'),
 'CAMELYON16': (re.compile(r'CAMELYON\s*-?16|CAMELYON', re.I), 'FROC', 'AUROC'),
}

results = []
files = []
for d in (sys.argv[1:] or ['xml','xml2','xml3']):
    files += sorted(glob.glob(os.path.join(d,'*.xml')))
stat = dict(scanned=0, bench_papers=0, cand_tables=0)
for p in files:
    if os.path.getsize(p) < 2000: continue
    stat['scanned'] += 1
    raw = open(p, encoding='utf-8', errors='replace').read()
    which = [b for b,(rx,_,_) in BENCH.items() if rx.search(raw)]
    if not which: continue
    stat['bench_papers'] += 1
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
        mcols = {}
        for c, lab in enumerate(collab):
            m = col_metric_exact(lab)
            if m: mcols.setdefault(m, c)
        for b in which:
            _, fm, cm = BENCH[b]
            if fm not in mcols or cm not in mcols: continue
            stat['cand_tables'] += 1
            data = {}
            for row in B:
                if not row: continue
                name = row[0].strip()
                if not name or len(name) < 2: continue
                a = num(row[mcols[fm]]) if mcols[fm] < len(row) else None
                z = num(row[mcols[cm]]) if mcols[cm] < len(row) else None
                if a is not None and z is not None: data[name] = (a, z)
            if len(data) < 3: continue
            pairs = [(k, v[0], v[1]) for k, v in data.items()]
            fv=[x[1] for x in pairs]; cv=[x[2] for x in pairs]
            if len(set(fv)) < 2 or len(set(cv)) < 2: continue
            disc = sum(1 for x,y in itertools.combinations(pairs,2) if (x[1]-y[1])*(x[2]-y[2])<0)
            tot = len(pairs)*(len(pairs)-1)//2
            fb = max(pairs,key=lambda x:x[1]); cb = max(pairs,key=lambda x:x[2])
            results.append(dict(pmcid=pmcid, bench=b, table=tb['label'], caption=tb['caption'][:200],
                fine_metric=fm, coarse_metric=cm, n=len(pairs), disc=disc, tot=tot,
                tau=round((tot-2*disc)/tot,3) if tot else None,
                top_changed=fb[0]!=cb[0],
                top_changed_strict=(fb[0]!=cb[0] and fv.count(max(fv))==1 and cv.count(max(cv))==1),
                fine_best=(fb[0],fb[1]), coarse_best=(cb[0],cb[2]), pairs=pairs))
json.dump(results, open('scan_anchored_out.json','w'), indent=1)
print("@@@", stat)
print("@@@ rows:", len(results), "papers:", len({r['pmcid'] for r in results}))
print("@@@ with >=1 discordant pair:", len({r['pmcid'] for r in results if r['disc']>0}))
print("@@@ strict top-1 change:", len({r['pmcid'] for r in results if r['top_changed_strict']}))
for r in sorted(results, key=lambda x: (x['tau'] if x['tau'] is not None else 9)):
    flag = "TOP1" if r['top_changed_strict'] else "    "
    print("@@@ %s %-13s %-11s %-8s n=%2d disc=%2d/%2d tau=%6s | %s" % (
        flag, r['pmcid'], r['bench'], r['table'][:8], r['n'], r['disc'], r['tot'],
        r['tau'], r['caption'][:70]))

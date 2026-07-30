"""
Scan for rank inversions in tables that report BOTH units as COLUMN GROUPS of one table
(e.g. header row 1 = "Slice-level | Patient-level", header row 2 = "AUC Acc | AUC Acc").
The pass-1 scanner only handled the two-separate-tables layout, so this covers the
commoner layout. Handles colspan/rowspan properly.
"""
import re, html, os, json, glob, itertools, sys

FINE   = r'(slice|image|patch|lesion|frame|b-?scan|section|nodule|instance)'
COARSE = r'(patient|case|subject|exam|scan|study|volume|series|person|individual|participant)'
METRIC_NAMES = ['auroc','auc','accuracy','acc','f1-score','f1 score','f1','balanced accuracy',
                'sensitivity','specificity','precision','recall','average precision','ap','dice','iou']
MODELWORD = re.compile(r'(resnet|densenet|vgg|inception|efficientnet|convnext|vit\b|swin|mobilenet|'
    r'alexnet|googlenet|xception|nasnet|unet|u-net|nnunet|transformer|cnn|squeezenet|shufflenet|'
    r'regnet|deit|beit|clip|resnext|wideresnet|seresnet|capsule|lstm|gru|svm|random forest|xgboost|'
    r'logistic|proposed|ours|baseline|model|net\b|radiomics|ensemble|3d|2d)', re.I)

def cells_of(row_html):
    out = []
    for m in re.finditer(r'<t([dh])\b([^>]*)>(.*?)</t\1>', row_html, re.S):
        attrs, inner = m.group(2), m.group(3)
        txt = html.unescape(re.sub(r'<[^>]+>', ' ', inner))
        txt = re.sub(r'\s+', ' ', txt).strip()
        cs = re.search(r'colspan\s*=\s*"?(\d+)', attrs)
        rs = re.search(r'rowspan\s*=\s*"?(\d+)', attrs)
        out.append((txt, int(cs.group(1)) if cs else 1, int(rs.group(1)) if rs else 1))
    return out

def expand(rows_cells):
    """rows_cells: list of list of (txt,colspan,rowspan) -> dense matrix of strings"""
    grid = {}
    for ri, cells in enumerate(rows_cells):
        ci = 0
        for (txt, cs, rs) in cells:
            while (ri, ci) in grid:
                ci += 1
            for dr in range(rs):
                for dc in range(cs):
                    grid[(ri + dr, ci + dc)] = txt
            ci += cs
    if not grid:
        return []
    nr = max(r for r, _ in grid) + 1
    nc = max(c for _, c in grid) + 1
    return [[grid.get((r, c), '') for c in range(nc)] for r in range(nr)]

def get_tables(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    out = []
    for m in re.finditer(r'<table-wrap\b.*?</table-wrap>', t, re.S):
        blk = m.group(0)
        def grab(tag):
            g = re.search(r'<%s.*?</%s>' % (tag, tag), blk, re.S)
            return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', g.group(0)))).strip() if g else ''
        cap, lab = grab('caption'), grab('label')
        thead = re.search(r'<thead\b.*?</thead>', blk, re.S)
        tbody = re.search(r'<tbody\b.*?</tbody>', blk, re.S)
        hrows = [cells_of(r.group(0)) for r in re.finditer(r'<tr\b.*?</tr>', thead.group(0), re.S)] if thead else []
        brows = [cells_of(r.group(0)) for r in re.finditer(r'<tr\b.*?</tr>', tbody.group(0), re.S)] if tbody else []
        if not hrows and not brows:
            allr = [cells_of(r.group(0)) for r in re.finditer(r'<tr\b.*?</tr>', blk, re.S)]
            hrows, brows = allr[:2], allr[2:]
        out.append({'label': lab, 'caption': cap, 'H': expand(hrows), 'B': expand(brows)})
    return out

def num(s):
    s = s.replace('%', '').replace('−', '-').strip()
    s = re.sub(r'^[^0-9.\-]*', '', s)
    m = re.match(r'^(-?[0-9]*\.?[0-9]+)', s)
    return float(m.group(1)) if m else None

def unit_of(text):
    tl = text.lower()
    fl = bool(re.search(FINE + r'[\s-]?(level|wise|based)', tl)) or bool(re.search(r'per[\s-]?' + FINE, tl))
    cl = bool(re.search(COARSE + r'[\s-]?(level|wise|based)', tl)) or bool(re.search(r'per[\s-]?' + COARSE, tl))
    if fl and not cl: return 'fine'
    if cl and not fl: return 'coarse'
    return None

def metric_of(text):
    tl = re.sub(r'\s+', ' ', text.lower())
    for mn in METRIC_NAMES:
        if re.search(r'(^|[^a-z])' + re.escape(mn) + r'([^a-z]|$)', tl):
            return mn.replace(' score', '').replace('-score', '')
    return None

def norm_model(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

results = []
CORPORA = sys.argv[1:] or ['xml', 'xml2']
files = []
for d in CORPORA:
    files += sorted(glob.glob(os.path.join(d, '*.xml')))
stat = dict(scanned=0, has_both_units_in_cols=0, yielded=0)
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
        # column label = concatenation of all header cells in that column
        collab = []
        for c in range(ncol):
            parts = []
            for r in H:
                if c < len(r) and r[c] and (not parts or parts[-1] != r[c]):
                    parts.append(r[c])
            collab.append(' '.join(parts))
        cols = {}   # (unit, metric) -> col index
        for c, lab in enumerate(collab):
            u, mt = unit_of(lab), metric_of(lab)
            if u and mt:
                cols.setdefault((u, mt), c)
        units = {u for (u, _) in cols}
        if len(units) < 2: continue
        shared = {mt for (u, mt) in cols if (('fine', mt) in cols and ('coarse', mt) in cols)}
        if not shared: continue
        stat['has_both_units_in_cols'] += 1
        # model rows
        data = {}
        for row in B:
            if not row: continue
            name = row[0].strip()
            if not name or not MODELWORD.search(name): continue
            vals = {}
            for (u, mt), ci in cols.items():
                if ci < len(row):
                    v = num(row[ci])
                    if v is not None: vals[(u, mt)] = v
            if vals: data[name] = vals
        if len(data) < 2: continue
        for mt in shared:
            pairs = [(k, v[('fine', mt)], v[('coarse', mt)]) for k, v in data.items()
                     if ('fine', mt) in v and ('coarse', mt) in v]
            if len(pairs) < 2: continue
            fv = [x[1] for x in pairs]; cv = [x[2] for x in pairs]
            disc = sum(1 for a, b in itertools.combinations(pairs, 2) if (a[1]-b[1])*(a[2]-b[2]) < 0)
            tot = len(pairs)*(len(pairs)-1)//2
            fb = max(pairs, key=lambda x: x[1]); cb = max(pairs, key=lambda x: x[2])
            if disc > 0:
                stat['yielded'] += 1
                results.append(dict(pmcid=pmcid, table=tb['label'], caption=tb['caption'][:200],
                    metric=mt, n=len(pairs), disc=disc, tot=tot,
                    top_changed=fb[0] != cb[0],
                    top_changed_strict=(fb[0] != cb[0] and fv.count(max(fv)) == 1 and cv.count(max(cv)) == 1),
                    n_distinct_fine=len(set(fv)), n_distinct_coarse=len(set(cv)),
                    fine_best=(fb[0], fb[1]), coarse_best=(cb[0], cb[2]), pairs=pairs,
                    collab=[collab[cols[('fine', mt)]], collab[cols[('coarse', mt)]]]))
json.dump(results, open('scan_single_out.json', 'w'), indent=1)
print(stat)
print("rows:", len(results), "papers:", len({r['pmcid'] for r in results}),
      "TOP1-strict papers:", len({r['pmcid'] for r in results if r['top_changed_strict']}))

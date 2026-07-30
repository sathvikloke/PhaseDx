import re,html,os,json,glob,itertools

FINE = r'(slice|image|patch|lesion|frame|b-?scan|per-slice|per-image|section)'
COARSE = r'(patient|case|subject|exam|scan|study|volume|series|person|individual)'
METRICS = ['auc','auroc','accuracy','acc','f1','f1-score','sensitivity','specificity','precision','recall','bacc','balanced accuracy','ap','dice']

def tables(path):
    t=open(path,encoding='utf-8',errors='replace').read()
    out=[]
    for m in re.finditer(r'<table-wrap\b.*?</table-wrap>', t, re.S):
        blk=m.group(0)
        cap=re.sub(r'<[^>]+>','',re.search(r'<caption.*?</caption>',blk,re.S).group(0)) if re.search(r'<caption.*?</caption>',blk,re.S) else ''
        lab=re.sub(r'<[^>]+>','',re.search(r'<label.*?</label>',blk,re.S).group(0)) if re.search(r'<label.*?</label>',blk,re.S) else ''
        rows=[]
        for r in re.finditer(r'<tr\b.*?</tr>', blk, re.S):
            cells=[html.unescape(re.sub(r'<[^>]+>','',c)).strip() for c in re.findall(r'<t[dh]\b.*?</t[dh]>', r.group(0), re.S)]
            if cells: rows.append(cells)
        out.append({'label':html.unescape(lab).strip(),'caption':html.unescape(cap).strip(),'rows':rows})
    return out

def num(s):
    s=s.replace('%','').replace(',','.').strip()
    m=re.match(r'^([0-9]*\.?[0-9]+)',s)
    if not m: return None
    v=float(m.group(1))
    return v

def norm_model(s):
    s=s.lower()
    s=re.sub(r'[^a-z0-9]','',s)
    return s

MODELWORD=re.compile(r'(resnet|densenet|vgg|inception|efficientnet|convnext|vit|swin|mobilenet|alexnet|googlenet|xception|nasnet|unet|u-net|nnunet|transformer|cnn|squeezenet|shufflenet|regnet|deit|beit|clip|resnext|wideresnet|seresnet|capsule|lstm|gru|svm|random forest|xgboost|logistic|proposed|ours|baseline|model)',re.I)

def metric_cols(header):
    cols={}
    for i,h in enumerate(header):
        hl=h.lower().strip()
        for mname in ['auroc','auc','accuracy','f1-score','f1 score','f1','balanced accuracy','sensitivity','specificity','precision','recall','average precision']:
            if hl==mname or hl.startswith(mname+' ') or hl.startswith(mname+'(') or re.fullmatch(mname+r'\s*\(?%?\)?',hl):
                cols.setdefault(mname.replace(' score','').replace('-score',''),i); break
    return cols

def parse_model_table(tb):
    rows=[r for r in tb['rows'] if r]
    if len(rows)<3: return None
    # find header row = first row with >=2 cells where a metric name appears
    hdr=None;hi=0
    for i,r in enumerate(rows[:3]):
        if metric_cols(r): hdr=r;hi=i;break
    if hdr is None: return None
    cols=metric_cols(hdr)
    if not cols: return None
    data={}
    for r in rows[hi+1:]:
        if len(r)<2: continue
        name=r[0].strip()
        if not name or not MODELWORD.search(name): continue
        vals={}
        for mn,ci in cols.items():
            if ci < len(r):
                v=num(r[ci])
                if v is not None: vals[mn]=v
        if vals: data[name]=vals
    if len(data)<2: return None
    return cols,data

def unit_of(text):
    tl=text.lower()
    f=re.search(FINE,tl); c=re.search(COARSE,tl)
    fl = bool(re.search(FINE+r'[- ]?(level|wise|based|by-)',tl)) or bool(re.search(r'per-?'+FINE,tl))
    cl = bool(re.search(COARSE+r'[- ]?(level|wise|based|by-)',tl)) or bool(re.search(r'per-?'+COARSE,tl))
    if fl and not cl: return 'fine'
    if cl and not fl: return 'coarse'
    return None

results=[]
for p in sorted(glob.glob('xml/*.xml')):
    if os.path.getsize(p)<2000: continue
    pmcid=os.path.basename(p)[:-4]
    try: tbs=tables(p)
    except Exception: continue
    parsed=[]
    for tb in tbs:
        cap=(tb['label']+' '+tb['caption'])
        u=unit_of(cap)
        if u is None: continue
        pm=parse_model_table(tb)
        if pm is None: continue
        parsed.append((u,cap,pm))
    fines=[x for x in parsed if x[0]=='fine']; coarses=[x for x in parsed if x[0]=='coarse']
    if not fines or not coarses: continue
    for (uf,capf,(colf,dataf)) in fines:
        for (uc,capc,(colc,datac)) in coarses:
            shared_metrics=set(colf)&set(colc)
            mf={norm_model(k):k for k in dataf}; mc={norm_model(k):k for k in datac}
            shared_models=set(mf)&set(mc)
            if len(shared_models)<2 or not shared_metrics: continue
            for met in shared_metrics:
                pairs=[]
                for sm in shared_models:
                    a=dataf[mf[sm]].get(met); b=datac[mc[sm]].get(met)
                    if a is not None and b is not None: pairs.append((mf[sm],a,b))
                if len(pairs)<2: continue
                fbest=max(pairs,key=lambda x:x[1]); cbest=max(pairs,key=lambda x:x[2])
                disc=sum(1 for x,y in itertools.combinations(pairs,2) if (x[1]-y[1])*(x[2]-y[2])<0)
                tot=len(pairs)*(len(pairs)-1)//2
                if disc>0:
                    results.append(dict(pmcid=pmcid,metric=met,n=len(pairs),disc=disc,tot=tot,
                        top_changed=fbest[0]!=cbest[0], fine_cap=capf[:110], coarse_cap=capc[:110],
                        fine_best=(fbest[0],fbest[1]), coarse_best=(cbest[0],cbest[2]), pairs=pairs))
json.dump(results,open('scan_out.json','w'),indent=1)
print("candidate paper-table-metric combos with >=1 discordant pair:", len(results))
print("distinct papers:", len({r['pmcid'] for r in results}))
print("with TOP-1 change:", len({r['pmcid'] for r in results if r['top_changed']}))

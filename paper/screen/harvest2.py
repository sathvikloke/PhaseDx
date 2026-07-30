import json,subprocess,urllib.parse,time,os,sys
SP=os.path.dirname(os.path.abspath(__file__))
os.chdir(SP)

def search(q, maxpages=8, pagesize=100):
    out=[]; cursor='*'
    for _ in range(maxpages):
        url=("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
             +urllib.parse.quote(q)+f"&format=json&pageSize={pagesize}&cursorMark={urllib.parse.quote(cursor)}")
        r=subprocess.run(["curl","-sL","-m","90",url],capture_output=True,text=True).stdout
        try: d=json.loads(r)
        except Exception: break
        res=d.get('resultList',{}).get('result',[])
        out+=res
        nc=d.get('nextCursorMark')
        if not nc or nc==cursor or not res: break
        cursor=nc; time.sleep(0.15)
    return out

# Queries deliberately DISJOINT in phrasing from harvest pass 1 (which used 10 queries
# built on slice/patient/image/exam + ResNet/DenseNet/AUC).
QUERIES=[
 # --- explicit dual-unit reporting language ---
 '"at both the slice and patient level"',
 '"at both slice and patient levels"',
 '"both slice-level and patient-level"',
 '"both slice-level and scan-level"',
 '"both image-level and exam-level"',
 '"both image-level and patient-level"',
 '"slice-level and volume-level"',
 '"slice-level and subject-level"',
 '"frame-level and video-level" AND ("AUC" OR "accuracy")',
 '"lesion-level and patient-level" AND ("AUC" OR "sensitivity")',
 '"image-level and case-level" AND ("AUC" OR "accuracy")',
 '"nodule-level" AND "scan-level" AND ("sensitivity" OR "AUC")',
 '"per-lesion" AND "per-patient" AND ("AUC" OR "sensitivity")',
 '"per-image" AND "per-exam" AND ("AUC" OR "AUROC")',
 '"per-frame" AND "per-study" AND "AUC"',
 # --- aggregation language ---
 '"slice-level predictions" AND "aggregat" AND ("patient" OR "volume")',
 '"aggregating slice-level" AND ("AUC" OR "accuracy")',
 '"majority voting" AND "slice-level" AND "patient-level" AND "AUC"',
 '"max pooling" AND "slice-level" AND "patient-level" AND "AUC"',
 '"mean of slice" AND "patient-level" AND "AUC"',
 # --- unit-of-analysis / evaluation-granularity methodology ---
 '"unit of analysis" AND ("slice" OR "image") AND ("patient" OR "exam") AND ("AUC" OR "deep learning")',
 '"evaluation granularity" AND ("slice" OR "image" OR "lesion")',
 '"granularity of evaluation" AND ("slice" OR "patient")',
 '"choice of evaluation unit"',
 '"slice-level metrics overestimate"',
 '"overestimate" AND "slice-level" AND "patient-level" AND "performance"',
 '"inflated" AND "slice-level" AND "patient-level"',
 '"optimistic" AND "slice-level" AND "patient-level" AND "AUC"',
 # --- ranking / model-selection language, the actual claim ---
 '"ranking" AND "slice-level" AND "patient-level" AND ("model" OR "architecture")',
 '"model selection" AND "slice-level" AND "patient-level"',
 '"best-performing model" AND "slice-level" AND "patient-level"',
 '"differ" AND "slice-level" AND "patient-level" AND "ranking"',
 '"reorder" OR "re-order" AND "slice-level" AND "patient-level"',
 # --- domains named in the task ---
 '"lung nodule" AND "nodule-level" AND "patient-level" AND ("AUC" OR "sensitivity")',
 '"intracranial h*morrhage" AND "slice-level" AND "patient-level"',
 '"intracranial h*morrhage" AND "examination-level" AND "slice-level"',
 '"prostate" AND "slice-level" AND "patient-level" AND ("AUC" OR "csPCa")',
 '"prostate cancer" AND "lesion-level" AND "patient-level" AND "AUC" AND "deep learning"',
 '"liver lesion" AND ("lesion-level" OR "slice-level") AND "patient-level" AND "AUC"',
 '"hepatocellular" AND "slice-level" AND "patient-level" AND "AUC"',
 '"COVID-19" AND "slice-level" AND "patient-level" AND ("AUC" OR "accuracy") AND "CT"',
 '"COVID" AND "slice-wise" AND "patient-wise" AND "accuracy"',
 '"pulmonary embolism" AND "image-level" AND "exam-level"',
 '"breast" AND "slice-level" AND "patient-level" AND "AUC" AND "MRI"',
 '"knee" AND "slice-level" AND "patient-level" AND "AUC"',
 '"Alzheimer" AND "slice-level" AND "subject-level" AND "accuracy"',
 '"colonoscopy" AND "frame-level" AND "patient-level" AND "AUC"',
 '"OCT" AND "B-scan-level" AND "patient-level"',
 '"chest CT" AND "slice-level" AND "patient-level" AND "AUC"',
 # --- challenge / leaderboard arm ---
 '"leaderboard" AND ("slice-level" OR "image-level") AND ("patient-level" OR "exam-level")',
 '"challenge" AND "evaluation metric" AND "per-slice" AND "per-patient"',
 '"CAMELYON" AND ("FROC" AND "AUC") AND "ranking"',
 '"LUNA16" AND ("CPM" OR "FROC") AND "ranking"',
 '"rank" AND "metric" AND "changed" AND "challenge" AND ("segmentation" OR "classification") AND "medical"',
]
seen=set()
if os.path.exists('pmcids.txt'):
    seen={l.strip() for l in open('pmcids.txt') if l.strip()}
print("pass-1 pmcids on disk:", len(seen), flush=True)
new=set(); meta={}
for q in QUERIES:
    r=search(q)
    got=set()
    for x in r:
        p=x.get('pmcid')
        if not p: continue
        got.add(p)
        meta[p]={'title':x.get('title'),'year':x.get('pubYear'),'doi':x.get('doi'),
                 'journal':x.get('journalInfo',{}).get('journal',{}).get('title'),
                 'pmid':x.get('pmid'),'oa':x.get('isOpenAccess')}
    fresh=got-seen-new
    new|=fresh
    print(f"{len(r):5d} hits {len(got):4d} pmcid {len(fresh):4d} NEW | {q}", flush=True)
print("TOTAL NEW PMCIDs:", len(new), flush=True)
open('pmcids_new2.txt','w').write('\n'.join(sorted(new)))
json.dump(meta,open('meta2.json','w'),indent=1)

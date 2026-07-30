import json,subprocess,urllib.parse,time,sys,os
def search(q, maxpages=6, pagesize=100):
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
        cursor=nc; time.sleep(0.2)
    return out
QUERIES=[
 '"slice-level" AND "patient-level" AND "ResNet"',
 '"slice-level" AND "patient-level" AND ("AUC" OR "AUROC")',
 '"image-level" AND "patient-level" AND ("ResNet" OR "DenseNet") AND ("CT" OR "MRI")',
 '"slice level" AND "patient level" AND "accuracy"',
 '"per-slice" AND "per-patient" AND ("AUC" OR "sensitivity")',
 '"slice-wise" AND "patient-wise" AND "AUC"',
 '"image-level" AND "exam-level" AND ("AUC" OR "AUROC")',
 '"lesion-level" AND "patient-level" AND ("ranking" OR "AUC") AND ("ResNet" OR "nnU-Net" OR "DenseNet")',
 '"scan-level" AND "slice-level" AND "AUC"',
 '"subject-level" AND "slice-level" AND "accuracy"',
]
ids=set()
for q in QUERIES:
    r=search(q)
    n=0
    for x in r:
        p=x.get('pmcid')
        if p and x.get('isOpenAccess')=='Y' or (p and x.get('hasTextMinedTerms')):
            ids.add(p); n+=1
        elif p:
            ids.add(p); n+=1
    print(f"{len(r):4d} hits, {n:4d} pmcids | {q}")
ids={i for i in ids if i}
print("TOTAL UNIQUE PMCIDs:", len(ids))
open('pmcids.txt','w').write('\n'.join(sorted(ids)))

#!/usr/bin/env python3
"""Rebuild paper/screen_sample.json from the frozen permutation.

    python paper/screen/build_sample.py            # fetch metadata + rebuild
    python paper/screen/build_sample.py --check     # rebuild in memory, diff against the
                                                    # committed file, exit 1 on any difference

The SAMPLE ITSELF (which PMIDs, in which batch) is fully determined by
paper/screen/permutation.txt and the allocation table below; this script only attaches
bibliographic metadata. Re-running it can never change which papers are screened.

------------------------------------------------------------------------------------------
PARSING HAZARD -- read before touching the efetch section.

A PubMed record embeds its REFERENCE LIST inside the same <PubmedArticle> element, and every
reference carries its own <ArticleIdList> with DOIs and PMCIDs. Using
    art.iter("ArticleId")
therefore walks the references too and yields the LAST reference's identifiers, not the
article's. The first build of this file did exactly that and corrupted 229/400 DOIs and
276/400 PMCIDs -- e.g. PMID 40335658, a 2025 Eur Radiol paper, was given 10.2106/00004623-
198567020-00007, a 1985 JBJS DOI. Because oa_status is derived from the PMCID, the open-access
flags were wrong too.

Identifiers MUST be read from these two scoped paths only:
    PubmedData/ArticleIdList/ArticleId[@IdType='doi'|'pmc']    (direct child -- not a reference)
    MedlineCitation/Article/ELocationID[@EIdType='doi']        (fallback)
Never iter() over the whole record. The same applies to AuthorList and PublicationTypeList.
verify_dois() below is the regression test: it resolves a seeded 25-DOI subsample through
Crossref and asserts the titles agree.
------------------------------------------------------------------------------------------
"""
import argparse, datetime, hashlib, json, random, re, sys, time
import urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UA = {"User-Agent": "PhaseDx-screening-protocol/1.0 (academic literature screen)"}

# Allocation is frozen by paper/screen_protocol.md §3. Do not change.
SPANS = {"overlap": (1, 15), "batch_A": (16, 37), "batch_B": (38, 58),
         "batch_C": (59, 79), "batch_D": (80, 100)}
PILOT = (101, 110)
RESERVE = (111, 400)
SEED = 20260729


def _post(endpoint, params):
    body = urllib.parse.urlencode(params).encode()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(EUTILS + endpoint, data=body), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {e}", file=sys.stderr)
            time.sleep(5)
    sys.exit(f"{endpoint} failed after 6 attempts")


def fetch_metadata(pmids):
    """efetch with correctly SCOPED element access. See PARSING HAZARD above."""
    out = {}
    for i in range(0, len(pmids), 40):
        chunk = pmids[i:i + 40]
        root = ET.fromstring(_post("efetch.fcgi",
                                   {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}))
        for art in root.findall("PubmedArticle"):
            cit, pubdata = art.find("MedlineCitation"), art.find("PubmedData")
            article = cit.find("Article")
            journal = article.find("Journal")

            doi = pmc = None
            idlist = pubdata.find("ArticleIdList") if pubdata is not None else None
            if idlist is not None:                      # DIRECT child only, never a reference
                for aid in idlist.findall("ArticleId"):
                    if aid.get("IdType") == "doi" and not doi:
                        doi = (aid.text or "").strip()
                    elif aid.get("IdType") == "pmc" and not pmc:
                        pmc = (aid.text or "").strip()
            if not doi:
                for el in article.findall("ELocationID"):
                    if el.get("EIdType") == "doi":
                        doi = (el.text or "").strip()
                        break

            authors = []
            alist = article.find("AuthorList")
            if alist is not None:
                for a in alist.findall("Author"):
                    ln = a.findtext("LastName")
                    if ln:
                        authors.append((ln + " " + (a.findtext("Initials") or "")).strip())
                    elif a.findtext("CollectiveName"):
                        authors.append(a.findtext("CollectiveName"))

            pub = journal.find("JournalIssue/PubDate")
            year = ((pub.findtext("Year") or pub.findtext("MedlineDate") or "")[:4]
                    if pub is not None else "")
            title_el = article.find("ArticleTitle")

            out[cit.findtext("PMID")] = {
                "title": "".join(title_el.itertext()) if title_el is not None else "",
                "authors": authors, "year": year,
                "venue": journal.findtext("ISOAbbreviation") or journal.findtext("Title") or "",
                "doi": doi or None, "pmcid": pmc or None,
                "publication_types": [p.text for p in
                                      article.findall("PublicationTypeList/PublicationType")],
            }
        print(f"  metadata {i + len(chunk)}/{len(pmids)}")
        time.sleep(0.4)
    return out


def add_oa_status(recs):
    """PMC presence, else Crossref licence. Deliberately no Unpaywall: its API wants an email
    in the query string, and we do not put a personal address in a URL. Both signals
    UNDERSTATE open access; oa_status is a hint for screeners, never an analysis variable."""
    for pmid, r in recs.items():
        if r.get("pmcid"):
            r["oa_status"], r["oa_evidence"] = "oa_pmc", "PMCID " + r["pmcid"]
            continue
        if not r.get("doi"):
            r["oa_status"], r["oa_evidence"] = "unknown_no_doi", ""
            continue
        try:
            u = "https://api.crossref.org/works/" + urllib.parse.quote(r["doi"], safe="")
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30) as h:
                lic = [l.get("URL", "") for l in json.load(h)["message"].get("license", [])]
        except Exception as e:  # noqa: BLE001
            r["oa_status"], r["oa_evidence"] = "unknown_no_license_metadata", str(e)[:60]
            time.sleep(0.15)
            continue
        cc = sorted({l for l in lic if "creativecommons.org" in l})
        if cc:
            r["oa_status"], r["oa_evidence"] = "oa_cc_license", "; ".join(cc[:2])
        elif lic:
            r["oa_status"], r["oa_evidence"] = "closed_or_tdm_license_only", "; ".join(sorted(set(lic))[:2])
        else:
            r["oa_status"], r["oa_evidence"] = "unknown_no_license_metadata", ""
        time.sleep(0.15)
    return recs


def verify_dois(perm, recs, n=25):
    """Regression test for the PARSING HAZARD: a corrupted DOI comes from a reference and will
    resolve to a different paper. Seeded so the check is reproducible."""
    sample = random.Random(1).sample(perm[:100], n)
    norm = lambda s: set(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
    agree = mismatch = err = 0
    for pmid in sample:
        r = recs[pmid]
        try:
            u = "https://api.crossref.org/works/" + urllib.parse.quote(r["doi"], safe="")
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30) as h:
                ct = (json.load(h)["message"].get("title") or [""])[0]
            pt = norm(r["title"])
            if len(norm(ct) & pt) / max(1, len(pt)) > 0.6:
                agree += 1
            else:
                mismatch += 1
                print(f"  MISMATCH {pmid}\n    pubmed: {r['title'][:80]}\n    doi   : {ct[:80]}")
        except Exception as e:  # noqa: BLE001
            err += 1
            print(f"  lookup error {pmid}: {str(e)[:50]}")
        time.sleep(0.2)
    print(f"DOI verification: {agree}/{n} titles agree, {mismatch} mismatch, {err} errors")
    return mismatch == 0


def build(perm, recs):
    def entry(pos):
        pmid = perm[pos - 1]
        r = recs[pmid]
        return {"position": pos, "pmid": pmid, "title": r["title"],
                "authors_first3": r["authors"][:3], "n_authors": len(r["authors"]),
                "year": r["year"], "venue": r["venue"], "doi": r["doi"], "pmcid": r["pmcid"],
                "oa_status": r["oa_status"], "oa_evidence": r.get("oa_evidence", ""),
                "publication_types": r["publication_types"],
                "url_pubmed": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "url_fulltext_pmc": (f"https://www.ncbi.nlm.nih.gov/pmc/articles/{r['pmcid']}/"
                                     if r["pmcid"] else None),
                "url_doi": ("https://doi.org/" + r["doi"]) if r["doi"] else None}

    groups = {k: [entry(i) for i in range(a, b + 1)] for k, (a, b) in SPANS.items()}
    pilot = [entry(i) for i in range(*(PILOT[0], PILOT[1] + 1))]
    reserve = []
    for k, pos in enumerate(range(RESERVE[0], RESERVE[1] + 1)):
        e = entry(pos)
        e["reserve_assigned_to"] = "batch_" + "ABCD"[k % 4]
        reserve.append(e)

    fm = json.load(open(HERE / "frame_meta.json"))
    return {
        "_schema": "PhaseDx screening sample v1.1",
        "_do_not_edit": ("Generated by paper/screen/build_sample.py. Which PMIDs land in which "
                         "batch is fixed by paper/screen/permutation.txt; this file only "
                         "attaches metadata. Verify with build_sample.py --check."),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "paper/screen_protocol.md",
        "coding_frame": "paper/screen_frame.json",
        "frame": {"database": fm["database"], "query": fm["query"],
                  "date_run_utc": fm["date_run_utc"],
                  "esearch_hit_count": fm["esearch_count"],
                  "pmids_retrieved": fm["pmids_retrieved"],
                  "frame_file": "paper/screen/frame_pmids.txt",
                  "frame_sha256": fm["frame_sha256"],
                  "canonical_order": "ascending numeric PMID (de-duplicated) before permutation"},
        "sampling": {
            "seed": SEED,
            "seed_justification": ("the UTC date the frame was retrieved (2026-07-29); fixed "
                                   "before the permutation was drawn and never changed"),
            "rng": "CPython stdlib random.Random(20260729).shuffle on the canonically ordered frame",
            "permutation_file": "paper/screen/permutation.txt",
            "permutation_sha256": hashlib.sha256("\n".join(perm).encode()).hexdigest(),
            "allocation": ("permutation positions 1-15 overlap set; 16-100 split into four "
                           "disjoint batches; 101-110 pilot (SEEN by the protocol author, "
                           "permanently excluded from analysis); 111-400 pre-specified reserve "
                           "in permutation order"),
            "sampling_fraction": round(100 / fm["pmids_retrieved"], 6)},
        "counts": {"overlap": 15, "batch_A": 22, "batch_B": 21, "batch_C": 21, "batch_D": 21,
                   "analysis_sample_total": 100, "pilot": 10, "reserve": 290,
                   "records_per_screener": {"batch_A": 37, "batch_B": 36,
                                            "batch_C": 36, "batch_D": 36}},
        "overlap_set": groups["overlap"],
        "batch_A": groups["batch_A"], "batch_B": groups["batch_B"],
        "batch_C": groups["batch_C"], "batch_D": groups["batch_D"],
        "pilot_set_excluded_from_analysis": pilot,
        "reserve": reserve,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="rebuild and diff against the committed screen_sample.json")
    ap.add_argument("--skip-verify", action="store_true", help="skip the Crossref DOI check")
    a = ap.parse_args()

    perm = [l.strip() for l in open(HERE / "permutation.txt") if l.strip()]
    need = perm[:RESERVE[1]]
    print(f"fetching metadata for {len(need)} records")
    recs = add_oa_status(fetch_metadata(need))
    if not a.skip_verify and not verify_dois(perm, recs):
        sys.exit("DOI verification FAILED -- identifiers are being read from the reference "
                 "list. See PARSING HAZARD in this file's docstring.")

    built = build(perm, recs)
    target = PAPER / "screen_sample.json"
    if a.check:
        cur = json.load(open(target))
        drop = lambda d: {k: v for k, v in d.items() if k != "generated_utc"}
        same = drop(cur) == drop(built)
        print("MATCH" if same else "DIFFERS from committed screen_sample.json")
        sys.exit(0 if same else 1)
    json.dump(built, open(target, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

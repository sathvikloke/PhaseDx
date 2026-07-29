#!/usr/bin/env python3
"""Reproduce the sampling frame and the seeded draw for the PhaseDx prevalence screen.

Run:  python paper/screen/reproduce_frame.py --verify
      python paper/screen/reproduce_frame.py --refetch     # re-query PubMed today

--verify  recomputes the permutation from the frozen PMID list on disk and checks both
          SHA-256 digests. Requires no network. This is the check a reviewer runs.
--refetch re-runs the exact esearch query against PubMed live. The hit count WILL drift
          (PubMed keeps indexing retrospectively); the frozen list in frame_pmids.txt is
          the authoritative frame. Drift is reported, not silently absorbed.

Everything below is fixed by paper/screen_protocol.md. Do not edit the query or the seed.
"""
import argparse, hashlib, json, random, sys, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- FROZEN. Any change to either constant invalidates the pre-registration. -----------
QUERY = (
    '("deep learning"[tiab] OR "convolutional neural network"[tiab] OR '
    '"convolutional neural networks"[tiab] OR CNN[tiab] OR "neural network"[tiab] OR '
    '"neural networks"[tiab]) AND (MRI[tiab] OR "magnetic resonance imaging"[tiab] OR '
    '"computed tomography"[tiab] OR CT[tiab] OR PET[tiab] OR '
    '"optical coherence tomography"[tiab]) AND (classification[tiab] OR classifier[tiab] '
    'OR classify[tiab] OR "computer-aided diagnosis"[tiab] OR diagnosis[tiab] OR '
    'detection[tiab]) AND (AUC[tiab] OR AUROC[tiab] OR "area under the curve"[tiab] OR '
    '"area under the receiver"[tiab] OR accuracy[tiab] OR sensitivity[tiab] OR '
    'specificity[tiab] OR "F1"[tiab]) AND english[la] AND 2019/01/01:2026/12/31[dp] NOT '
    '(review[pt] OR "systematic review"[pt] OR meta-analysis[pt] OR editorial[pt] OR '
    'comment[pt] OR "case reports"[pt])'
)
SEED = 20260729
FRAME_SHA256 = "d611def0785f3a5e7b7489364959f1d3471b61651f98a3ed049252654264374b"
PERM_SHA256 = "dad12a30b77d1213ac5e8ced89cf3a6620977b5734b5076641bb8adb2db74a1a"
FRAME_HIT_COUNT = 9979
DATE_RUN_UTC = "2026-07-29"
# --------------------------------------------------------------------------------------

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def sha(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def canonical(pmids):
    """The one canonical ordering: de-duplicated, ascending numeric PMID.

    PubMed's own return order is not stable across calls, so the permutation is defined
    on this order and not on retrieval order.
    """
    return sorted(set(pmids), key=int)


def permute(pmids):
    perm = list(pmids)
    random.Random(SEED).shuffle(perm)
    return perm


def esearch():
    pmids, retstart = [], 0
    while True:
        body = urllib.parse.urlencode({
            "db": "pubmed", "term": QUERY, "retmode": "json",
            "retmax": "9999", "retstart": str(retstart), "sort": "pub_date",
        }).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(BASE + "esearch.fcgi", data=body)
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.load(r)["esearchresult"]
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt}: {e}", file=sys.stderr)
                time.sleep(4)
        else:
            sys.exit("esearch failed after 5 attempts")
        total, batch = int(j["count"]), j["idlist"]
        pmids += batch
        retstart += len(batch)
        print(f"  retrieved {retstart}/{total}")
        if retstart >= total or not batch:
            return pmids, total
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="offline digest check")
    ap.add_argument("--refetch", action="store_true", help="re-query PubMed live")
    a = ap.parse_args()
    if not (a.verify or a.refetch):
        ap.error("pass --verify or --refetch")

    if a.verify:
        frame = canonical(l.strip() for l in open(HERE / "frame_pmids.txt") if l.strip())
        perm_disk = [l.strip() for l in open(HERE / "permutation.txt") if l.strip()]
        ok = True
        for name, got, want in (("frame", sha(frame), FRAME_SHA256),
                                ("permutation-on-disk", sha(perm_disk), PERM_SHA256),
                                ("permutation-recomputed", sha(permute(frame)), PERM_SHA256)):
            flag = "OK " if got == want else "FAIL"
            ok &= got == want
            print(f"[{flag}] {name}: {got}")
        print(f"[{'OK ' if len(frame) == FRAME_HIT_COUNT else 'FAIL'}] "
              f"frame size: {len(frame)} (expected {FRAME_HIT_COUNT})")
        ok &= len(frame) == FRAME_HIT_COUNT
        sys.exit(0 if ok else 1)

    print(f"Re-running the frozen query live (original run {DATE_RUN_UTC}, "
          f"n={FRAME_HIT_COUNT})")
    live, total = esearch()
    live = canonical(live)
    frozen = canonical(l.strip() for l in open(HERE / "frame_pmids.txt") if l.strip())
    added, dropped = set(live) - set(frozen), set(frozen) - set(live)
    print(f"\nlive hit count : {total}")
    print(f"frozen frame   : {len(frozen)}  ({DATE_RUN_UTC})")
    print(f"records added since freeze  : {len(added)}")
    print(f"records dropped since freeze: {len(dropped)}")
    print("\nThe FROZEN list governs. Drift is expected and does not invalidate the draw; "
          "it is reported in the paper as the frame-stability check.")
    (HERE / "frame_pmids_refetched.txt").write_text("\n".join(live) + "\n")
    print(f"live list written to {HERE / 'frame_pmids_refetched.txt'} (not used for sampling)")


if __name__ == "__main__":
    main()

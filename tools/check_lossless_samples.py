#!/usr/bin/env python3
"""Bounded, label-blind decoder agreement on an explicitly frozen file list."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pydicom

from phasedx_external.dicom_decode import JPEG_LOSSLESS_SV1, decode_pixels
from phasedx_external.prospective_guard import require_registration


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_registration(__file__, a.plan)
    plan = json.loads(Path(a.plan).read_text())
    if sha(__file__) != plan["script_sha256"]:
        raise RuntimeError("script differs from prospective diagnostic plan")
    from phasedx_external import dicom_decode
    if sha(dicom_decode.__file__) != plan["decoder_sha256"]:
        raise RuntimeError("decoder differs from diagnostic plan")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    start = time.monotonic()
    for source in plan["files"]:
        if (Path(plan["run_root"]) / "STOP_REQUESTED").exists():
            break
        row = dict(source)
        try:
            path = Path(plan["image_root"]) / source["relpath"]
            actual = sha(path)
            if actual != source["sha256"]:
                raise RuntimeError("source byte mismatch")
            ds = pydicom.dcmread(path)
            uid = str(ds.file_meta.TransferSyntaxUID)
            row.update(transfer_syntax=uid, bits_allocated=int(ds.BitsAllocated),
                       bits_stored=int(ds.BitsStored), signed=int(ds.PixelRepresentation),
                       shape=[int(ds.Rows), int(ds.Columns)])
            if uid != JPEG_LOSSLESS_SV1:
                row["status"] = "not_SV1_not_tested"
            else:
                x = decode_pixels(ds, "lossless_sv1_pylibjpeg_v1")
                y = pydicom.pixels.pixel_array(ds, raw=True, decoding_plugin="gdcm")
                same = bool(np.array_equal(x, y))
                slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
                hu_same = bool(np.array_equal(x.astype(np.float64)*slope+intercept,
                                              y.astype(np.float64)*slope+intercept))
                row.update(status="pass" if same and hu_same else "mismatch",
                           exact_pixel_equality=same, exact_HU_equality=hu_same,
                           pixel_min=int(x.min()), pixel_max=int(x.max()),
                           decoded_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        except Exception as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
        records.append(row)
        with (out / "sample_checks.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
    report = {"scope": "Fixed label-blind existing-image decoder agreement, not cohort validation or clinical inference",
              "plan_sha256": sha(a.plan), "planned": len(plan["files"]), "completed": len(records),
              "passed": sum(r["status"] == "pass" for r in records),
              "failed": sum(r["status"] in ("failed", "mismatch") for r in records),
              "elapsed_seconds": time.monotonic()-start}
    (out / "SUMMARY.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))
    return 0 if not report["failed"] and report["completed"] == report["planned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

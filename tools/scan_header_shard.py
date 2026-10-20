#!/usr/bin/env python3
"""Resume one locked source-inventory shard, reading file bytes once for SHA and headers.

No pixel decompression, model, reference labels or scientific outcome calculation.
Each source remains represented, including parse failures and identity mismatches.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import time

import pydicom

from phasedx_external.prospective_guard import require_registration


def header(payload):
    ds = pydicom.dcmread(io.BytesIO(payload), stop_before_pixels=True, force=False)
    def g(key, default=""):
        v = ds.get(key)
        return default if v is None else v
    ipp = [float(x) for x in g("ImagePositionPatient", [])]
    iop = [float(x) for x in g("ImageOrientationPatient", [])]
    r = dict(StudyInstanceUID=str(g("StudyInstanceUID")), SeriesInstanceUID=str(g("SeriesInstanceUID")),
             SOPInstanceUID=str(g("SOPInstanceUID")), instance_number=str(g("InstanceNumber")),
             modality=str(g("Modality")), transfer_syntax=str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
             number_of_frames=int(g("NumberOfFrames", 1)), has_rescale="RescaleSlope" in ds and "RescaleIntercept" in ds,
             contrast_agent=str(g("ContrastBolusAgent")), convolution_kernel=str(g("ConvolutionKernel")),
             series_description=str(g("SeriesDescription")), image_type=str(g("ImageType")),
             body_part=str(g("BodyPartExamined")), manufacturer=str(g("Manufacturer")),
             slice_thickness=str(g("SliceThickness")), n_ipp=len(ipp), n_iop=len(iop),
             bits_allocated=str(g("BitsAllocated")), bits_stored=str(g("BitsStored")),
             pixel_representation=str(g("PixelRepresentation")), photometric=str(g("PhotometricInterpretation")),
             rows=str(g("Rows")), columns=str(g("Columns")), lossy=str(g("LossyImageCompression", "00")))
    r.update({f"ipp{i}": ipp[i] if len(ipp) == 3 else None for i in range(3)})
    r.update({f"iop{i}": iop[i] if len(iop) == 6 else None for i in range(6)})
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    require_registration(__file__, a.plan)
    plan_path, out = Path(a.plan), Path(a.out)
    plan = json.loads(plan_path.read_text())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["script_sha256"]:
        raise RuntimeError("source differs from frozen header plan")
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "INPUT.json"
    if lock_path.exists() and json.loads(lock_path.read_text())["plan_sha256"] != digest:
        raise RuntimeError("resume plan changed")
    if not lock_path.exists():
        lock_path.write_text(json.dumps({"plan_sha256": digest, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    records = out / "headers.jsonl"
    done = {}
    if records.exists():
        lines = records.read_bytes().splitlines(keepends=True)
        complete = []
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise RuntimeError("corrupt middle receipt; refusing resume")
                (out / ("interrupted_tail_" + str(time.time_ns()) + ".bin")).write_bytes(line)
                records.write_bytes(b"".join(complete))
                break
            done[row["relpath"]] = row
            complete.append(line)
    deadline = datetime.datetime.fromisoformat(plan["deadline_utc"].replace("Z", "+00:00")).timestamp()
    started = time.monotonic()
    newly_read = 0
    n_new = 0
    io_failure = None
    root = Path(plan["image_root"])
    # A disconnected source is not an unreadable examination. Leave it pending.
    try:
        if not root.is_dir():
            raise FileNotFoundError("source image root is unavailable")
    except OSError as e:
        io_failure = {"reason": f"{type(e).__name__}: {e}", "relpath": None}
    for src in plan["files"]:
        if io_failure:
            break
        if src["relpath"] in done:
            continue
        if time.time() >= deadline or (Path(plan["run_root"]) / "STOP_REQUESTED").exists():
            break
        row = {k: src[k] for k in ("relpath", "size", "sha256", "source_exam_id", "assignment")}
        row["size"] = int(row["size"])
        row.update(readable=False, unreadable_reason="", hash_verified=False)
        try:
            p = root / src["relpath"]
            payload = p.read_bytes()
            newly_read += len(payload)
            actual = hashlib.sha256(payload).hexdigest()
            if actual != src["sha256"] or len(payload) != int(src["size"]):
                row.update(actual_sha256=actual, actual_size=len(payload))
                raise RuntimeError("source changed since frozen inventory")
            row["hash_verified"] = True
            row.update(header(payload))
            row["readable"] = True
        except OSError as e:
            io_failure = {"reason": f"{type(e).__name__}: {e}", "relpath": src["relpath"]}
            # No source bytes/header were authenticated. Do not add it to done.
            break
        except Exception as e:
            row["unreadable_reason"] = f"{type(e).__name__}: {e}"
        with records.open("a") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
        done[src["relpath"]] = row
        n_new += 1
        if n_new % 100 == 0:
            (out / "PROGRESS.json").write_text(json.dumps({
                "completed": len(done), "planned": len(plan["files"]), "new_bytes_read": newly_read,
                "elapsed_seconds": time.monotonic()-started}))
    summary = {"planned": len(plan["files"]), "completed": len(done),
               "parse_failures": sum(r["hash_verified"] and not r["readable"] for r in done.values()),
               "identity_failures": sum(not r["hash_verified"] for r in done.values()),
               "plan_sha256": digest, "new_bytes_read": newly_read, "new_files": n_new,
               "elapsed_seconds": time.monotonic()-started,
               "io_failure": io_failure,
               "status": "blocked_io" if io_failure else
                         "identity_failure" if any(not r["hash_verified"] for r in done.values()) else
                         "complete" if len(done) == len(plan["files"]) else "checkpoint"}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))
    return 0 if summary["status"] == "complete" else 92


if __name__ == "__main__":
    raise SystemExit(main())

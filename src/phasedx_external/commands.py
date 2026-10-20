"""Thin command interfaces to the preserved scientific implementation."""
import argparse
import csv
import json
from pathlib import Path

from . import protocol28 as P


def _parser(name, fields):
    ap = argparse.ArgumentParser(prog="phasedx-external " + name)
    for field in fields:
        ap.add_argument("--" + field, required=True)
    return ap


def select(argv=None):
    ap = _parser("select", ["manifest", "review", "images", "out"])
    a = ap.parse_args(argv)
    from .build_manifest28 import main
    m = Path(a.manifest)
    args = ["--roster", str(m / "roster.csv"), "--root", a.images,
            "--policy", str(m / "policy.json"), "--selection", a.review, "--out", a.out]
    if (m / "HEADER_CACHE.json").exists():
        args += ["--header-cache", str(m / "HEADER_CACHE.json")]
    # Old manifests without complete cached headers still require a source scan.
    # Prediction admission independently checks selected source bytes.
    return main(args)


def lock(argv=None):
    ap = _parser("lock", ["design", "manifest", "out"])
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    require_registration(__file__)
    P.make_execution_lock(a.out, a.design, a.manifest)
    return 0


def receipt(argv=None):
    ap = _parser("receipt", ["execution-lock", "manifest", "runs", "out"])
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    require_registration(__file__)
    P.make_receipt(a.out, a.execution_lock, json.loads(Path(a.runs).read_text()), a.manifest)
    return 0


def score(argv=None):
    ap = _parser("score", ["design", "execution-lock", "receipt", "runs", "manifest",
                           "reference", "schema", "out"])
    ap.add_argument("--grouping")
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    require_registration(__file__)
    from .scorer29 import evaluate
    # No runtime endpoint, seed, resample-count or threshold options.
    result = evaluate(a.design, a.execution_lock, a.receipt,
                      json.loads(Path(a.runs).read_text()), a.manifest,
                      str(Path(a.manifest) / "candidates.csv"), a.reference, a.schema, a.grouping)
    P.write_once(a.out, result)
    return 0


def flow(argv=None):
    ap = _parser("flow", ["result", "out"])
    a = ap.parse_args(argv)
    result = json.loads(Path(a.result).read_text())
    keep = ("mode", "label", "n_candidates", "flow_partition", "eligible_labeled_denominator",
            "completion_bounds", "reference_missing_policy_counts", "unknown_keys", "grouping")
    P.write_once(a.out, {k: result[k] for k in keep if k in result})
    return 0


def reference(argv=None):
    ap = _parser("reference", ["design", "execution-lock", "receipt", "runs", "manifest",
                               "reads", "schema", "out"])
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    from . import reference28 as R
    require_registration(__file__)
    design = json.loads(Path(a.design).read_text())

    def opener():
        schema = json.loads(Path(a.schema).read_text())
        R.validate_schema(schema)
        if design["mode"] == "clinical_external" and P.sha_file(a.schema) != design["clinical_contract"]["reference_schema_sha256"]:
            raise P.AdmissionError("reference schema differs from frozen design")
        with open(a.reads, newline="") as f:
            return R.adapt(list(csv.DictReader(f)), schema)

    admission, rows = P.admit(a.design, a.execution_lock, a.receipt,
                             json.loads(Path(a.runs).read_text()), a.manifest,
                             str(Path(a.manifest) / "candidates.csv"), opener)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fields = ["source_key", "exam_key", "label", "n_reads_invalid_or_missing"]
    with (out / "reference.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    P.write_once(str(out / "REFERENCE_RECEIPT.json"), {
        "mode": design["mode"], "rows": len(rows), "source_sha256": P.sha_file(a.reads),
        "schema_sha256": P.sha_file(a.schema), "admission": admission, "rule_id": R.RULE_ID})
    return 0


def export(argv=None):
    ap = _parser("export", ["result", "out"])
    a = ap.parse_args(argv)
    result = json.loads(Path(a.result).read_text())
    rows = []
    for component, fields in {
        "primary": ("status", "auroc_image_seed42", "auroc_geometry_transported", "difference",
                    "n_pos", "n_neg", "paired_support", "eligible_labeled_denominator", "estimand"),
        "secondary_seed123": ("status", "auroc_seed123", "auroc_geometry_on_this_set", "difference",
                             "n_pos", "n_neg", "paired_support", "overlap_with_primary_set",
                             "secondary_only", "primary_only"),
        "bootstrap": ("status", "n_draws", "n_undefined_draws", "n_clusters", "seed", "lo", "hi"),
        "grouping": ("status", "caveat"),
    }.items():
        for field in fields:
            if field in result.get(component, {}):
                rows.append({"mode": result["mode"], "component": component, "field": field,
                             "value": result[component][field]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "aggregate_table.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, ["mode", "component", "field", "value"])
        writer.writeheader()
        writer.writerows(rows)
    P.write_once(str(out / "EXPORT.json"), {
        "source_result_sha256": P.sha_file(a.result), "rows": len(rows),
        "scope": "Explicit aggregate fields only; no case keys, source paths or draw arrays.",
        "release_status": "candidate; scientific and author approval remain required"})
    return 0

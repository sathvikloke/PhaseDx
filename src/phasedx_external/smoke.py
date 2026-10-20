"""Tiny synthetic study contract rehearsal: no images, models, RNG or network."""
import argparse
import csv
import json
from pathlib import Path

from . import protocol28 as P
from .reference28 import RULE_ID
from .scorer29 import evaluate


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prepare(root):
    root = Path(root)
    root.mkdir(exist_ok=False, parents=True)
    m = root / "manifest"
    m.mkdir()
    keys = ["FAKE_A", "FAKE_B", "FAKE_C", "FAKE_D", "FAKE_E", "FAKE_F"]
    for c in P.COMPONENTS:
        (m / c).write_text("fixture\n")
    write_csv(m / "roster.csv", ["exam_key"], [{"exam_key": k} for k in keys])
    write_csv(m / "candidates.csv", ["exam_key", "header_eligibility"],
              [{"exam_key": k, "header_eligibility": "eligible" if i < 5 else "unresolved"}
               for i, k in enumerate(keys)])
    write_csv(m / "inventory.csv", ["relpath", "sha256"], [])
    (m / "policy.json").write_text('{"fixture":true,"decoder_policy":"uncompressed_v1"}')
    rebind_manifest(m)
    design = {
        "protocol_id": "PUBLIC_SYNTHETIC_SMOKE_NO_CLINICAL_DATA",
        "mode": "synthetic_point_only",
        "models": {r: {"model_id": mid, "checkpoint_sha256": P.sha_obj({"fake": r})}
                   for r, mid in zip(P.ROLES_ALL, ("I1_seed42", "I1_seed123",
                                                  "G2_tree_transported_rank_reconstructed_frozen"))},
        "preprocessing": {"recipe_sha256": P.sha_obj("fake recipe"),
                          "precision_path": "recipe_f32->float16_roundtrip->f32_normalize"},
        "aggregation": "max_slice_logit",
        "representation": {"representation_id": "TRANSPORTED_PHYSICAL_RANK_FROZEN_RECONSTRUCTED_G2"},
        "label_rule_id": RULE_ID, "grouping_policy": "fallback_if_linkage_unavailable",
        "estimand": "examination-weighted paired AUROC difference",
    }
    dp, ep = root / "DESIGN.json", root / "EXECUTION.json"
    P.make_design_lock(str(dp), design)
    P.make_execution_lock(str(ep), str(dp), str(m))
    design, execl = json.loads(dp.read_text()), json.loads(ep.read_text())
    # Hand-written point scores: primary A/B beat C/D; geometry is the reverse.
    # E has an image failure and secondary B fails, exercising unequal support.
    scores = {"primary_image": [4, 3, 2, 1, None, None],
              "secondary_image": [4, None, 2, 1, 3, None],
              "geometry": [1, 2, 3, 4, 5, None]}
    runs = {}
    for role in P.ROLES_ALL:
        rd = root / role
        actual = P.intended_identity(role, design, execl)
        st, _ = P.start_run(str(rd), role, str(dp), str(ep), str(m), actual)
        rows = []
        for k, v in zip(keys, scores[role]):
            rows.append({"exam_key": k, "score": "" if v is None else v,
                         "status": "failed" if v is None else "scored_complete",
                         "run_id": st["run_id"], "execution_lock_sha256": st["execution_lock_sha256"],
                         "identity_sha256": st["actual_sha256"]})
        pp = rd / "predictions.csv"
        write_csv(pp, list(rows[0]), rows)
        P.end_run(str(rd), str(pp), len(rows))
        runs[role] = [str(rd), str(pp)]
    (root / "runs.json").write_text(json.dumps(runs))
    rc = root / "RECEIPT.json"
    P.make_receipt(str(rc), str(ep), runs, str(m))
    schema = {"provenance": {"locator": "FABRICATED PUBLIC TEST; not clinical evidence"},
              "authenticated": True, "exam_key_col": "name", "key_transform": "identity",
              "reader_cols": ["r1", "r2", "r3"], "n_readers": 3, "rule_id": RULE_ID,
              "missing_policy": "two_agreeing_valid_reads_else_missing"}
    sp = root / "schema.json"
    sp.write_text(json.dumps(schema))
    lp = root / "reference.csv"
    write_csv(lp, ["name", "r1", "r2", "r3"],
              [{"name": k, "r1": y, "r2": y, "r3": y}
               for k, y in zip(keys, [1, 1, 0, 0, 1, 0])])
    return [str(dp), str(ep), str(rc), runs, str(m), str(m / "candidates.csv"), str(lp), str(sp)]


def rebind_manifest(m):
    import hashlib
    m = Path(m)
    rows = list(csv.DictReader((m / "inventory.csv").open()))
    inv = hashlib.sha256("\n".join(f"{r['relpath']}\t{r['sha256']}" for r in rows).encode()).hexdigest()
    (m / "MANIFEST_IDENTITY.json").write_text(json.dumps({
        "components": {c: P.sha_file(m / c) for c in P.COMPONENTS}, "inventory_digest": inv}))


def run(root):
    args = prepare(root)
    result = evaluate(*args)
    assert result["primary"]["auroc_image_seed42"] == 1
    assert result["primary"]["auroc_geometry_transported"] == 0
    assert result["primary"]["difference"] == 1
    assert result["eligible_labeled_denominator"] == 5
    assert result["flow_partition"] == {
        "labeled_paired_evaluable": 4, "labeled_image_fail": 1, "unresolved_eligibility": 1}
    assert result["completion_bounds"]["bounds"]["difference"] == [1 / 3, 1]
    secondary = result["secondary_seed123"]
    assert (secondary["paired_support"], secondary["overlap_with_primary_set"],
            secondary["secondary_only"], secondary["primary_only"]) == (4, 3, 1, 1)
    assert secondary["difference"] == .5
    assert result["bootstrap"]["status"] == "not_applicable_point_only_mode"
    P.write_once(str(Path(root) / "RESULT.json"), result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="New directory for fabricated evidence")
    a = ap.parse_args(argv)
    run(a.out)
    print("PASS: fixed fabricated scores, full denominator, separate secondary support; no bootstrap or model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

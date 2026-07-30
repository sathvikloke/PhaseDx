#!/usr/bin/env python3
"""
Final pooled prevalence analysis for the PhaseDx zero-image-baseline screen.

Pools the re-coded analysis sample (permutation positions 1-100, codebook v1.2 +
v1.3 access-recovery overlay) with the reserve blocks drawn under the
pre-registered extension rule (screen_protocol.md sec.3.1) and reports:

  * the flow (frame -> sampled -> screened -> eligible -> reachable -> included)
  * post-adjudication agreement, carried through from adjudication_out.json
  * primary endpoint P1, Wilson 95%, plus the two unconditional bounding analyses
  * the censoring-free statement (positive codes anywhere, any denominator)
  * secondary endpoints S1-S9 and the pre-specified exploratory subgroups

Nothing here re-decides any paper.  Status is taken from each record's own
`final_inclusion`, which is the field the codebook's rule D1 governs, and a D1
conformance check is run over every block rather than assumed.

Inputs are read-only.  Writes paper/screen/analysis/pooled_final.json.

Usage:  python paper/screen/analysis/pool_final.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
P = lambda *a: os.path.join(ROOT, *a)

# ----------------------------------------------------------------------------
# intervals
# ----------------------------------------------------------------------------

Z = 1.959963984540054  # two-sided 95%


def wilson(k: int, n: int):
    """Wilson score interval, the interval method pre-specified for every
    proportion in this paper (screen_protocol.md sec.8)."""
    if n == 0:
        return None
    p = k / n
    d = 1.0 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    lo, hi = (c - h) / d, (c + h) / d
    return {"k": k, "n": n, "pct": 100.0 * p,
            "ci95": [100.0 * max(0.0, lo), 100.0 * min(1.0, hi)]}


def fmt(w) -> str:
    if w is None:
        return "n/a"
    return "%d/%d = %.1f%% [%.1f%%, %.1f%%]" % (
        w["k"], w["n"], w["pct"], w["ci95"][0], w["ci95"][1])


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------

P1_SUBFLAGS = ["constant_or_prevalence", "positional",
               "acquisition_metadata", "permuted_or_shuffled_label"]
ALL_SUBFLAGS = P1_SUBFLAGS + ["clinical_or_demographic_only", "other_non_imaging"]

STATUS = {
    "included": "included_reachable",
    "unreachable_eligibility_unresolved": "eligible_unreachable",
    "excluded": "excluded",
}


def sample_meta():
    d = json.load(open(P("paper", "screen_sample.json")))
    out = {}
    for key in ("overlap_set", "batch_A", "batch_B", "batch_C", "batch_D",
                "reserve", "pilot_set_excluded_from_analysis"):
        for r in d[key]:
            out[str(r["pmid"])] = r
    return out


META = sample_meta()


def load_main():
    """Analysis sample, positions 1-100.

    Paper-level codes come from screen_recoded.json `papers` (the post-
    adjudication, post-recode consensus).  Fields the consensus does not carry
    are lifted from the per-screener amended records; where the four overlap
    screeners still differ on such a field the disagreement is recorded rather
    than collapsed.
    """
    d = json.load(open(P("paper", "screen_recoded.json")))
    amended = collections.defaultdict(list)
    for rec in d["records"]:
        amended[str(rec["record_id"])].append(rec["amended"])

    papers = []
    for p in d["papers"]:
        pid = str(p["record_id"])
        recs = amended[pid]
        base = dict(recs[0]) if recs else {}
        # consensus fields override the per-screener record
        for f in ("final_inclusion", "fulltext_reachable", "exclusion_code",
                  "trivial_baseline", "evaluation_unit_reported",
                  "headline_unit", "split_unit",
                  "positional_distribution_reported"):
            if f in p:
                base[f] = p[f]
        resid = p.get("residual_between_screener_differences") or {}
        # modal value for fields the adjudication did not settle
        for f, per in resid.items():
            vals = [v for v in per.values()]
            base[f] = collections.Counter(map(json.dumps, vals)).most_common(1)[0][0]
            base[f] = json.loads(base[f])
        base["record_id"] = pid
        base["block"] = "main"
        base["batch"] = p["batch"]
        base["permutation_position"] = p["permutation_position"]
        base["n_screeners"] = p["n_screeners"]
        base["unsettled_fields"] = sorted(resid)
        base["consensus_P1"] = p["P1"]
        base["consensus_S1"] = p["S1"]
        base["fulltext_search"] = p.get("fulltext_search")
        papers.append(base)
    assert len(papers) == 100, len(papers)
    return papers, d


def load_reserve(block: str):
    d = json.load(open(P("paper", "screen_reserve_%s.json" % block)))
    out = []
    for r in d["records"]:
        r = dict(r)
        r["record_id"] = str(r["record_id"])
        r["block"] = block
        r["n_screeners"] = 1
        r["unsettled_fields"] = []
        out.append(r)
    return out, d


# ----------------------------------------------------------------------------
# normalisation and conformance
# ----------------------------------------------------------------------------

def status_of(r):
    return STATUS[r["final_inclusion"]]


def d1_violations(papers):
    """Codebook rule D1: unreachable dominates included; an unreachable record
    may be coded `excluded` only if its stage-1 decision was already exclude.
    Excluding a paper *at full text* that was never obtained is the error D1
    exists to catch, and it deflates the unreachable rate."""
    bad = []
    for r in papers:
        reach = str(r.get("fulltext_reachable") or "")
        if (r["final_inclusion"] == "excluded"
                and reach.startswith("unreachable")
                and r.get("stage1_decision") != "exclude"):
            bad.append(r["record_id"])
    return bad


def p1_of(r):
    """Primary flag for one paper.  True iff at least one of the four zero-image
    sub-flags is TRUE with a measured value.  Returns True / False /
    'not_assessable' (eligible but unreachable) / 'not_applicable' (excluded)."""
    st = status_of(r)
    if st == "excluded":
        return "not_applicable"
    if st == "eligible_unreachable":
        return "not_assessable"
    tb = r["trivial_baseline"]
    return any(tb.get(k) is True for k in P1_SUBFLAGS)


def s1_of(r):
    st = status_of(r)
    if st == "excluded":
        return "not_applicable"
    if st == "eligible_unreachable":
        return "not_assessable"
    tb = r["trivial_baseline"]
    return any(tb.get(k) is True for k in ALL_SUBFLAGS)


# ----------------------------------------------------------------------------
# build the pool
# ----------------------------------------------------------------------------

def main():
    main_papers, main_doc = load_main()
    blocks = {"main": main_papers}
    docs = {"main": main_doc}
    for b in ("R1", "R2", "R3", "R4"):
        blocks[b], docs[b] = load_reserve(b)

    # ---- de-duplication ----------------------------------------------------
    seen, dupes, pool = {}, [], []
    for b in ("main", "R1", "R2", "R3", "R4"):
        for r in blocks[b]:
            pid = r["record_id"]
            if pid in seen:
                dupes.append({"pmid": pid, "first_block": seen[pid],
                              "duplicate_block": b})
                continue
            seen[pid] = b
            pool.append(r)

    # ---- conformance -------------------------------------------------------
    viol = d1_violations(pool)

    # ---- extension rule ----------------------------------------------------
    inc_by_block = {b: sum(1 for r in blocks[b] if status_of(r) == "included_reachable")
                    for b in blocks}
    running, order, stop_after = 0, ["main", "R1", "R2", "R3", "R4"], None
    trace = []
    for b in order:
        authorised = (b == "main") or (stop_after is None)
        running += inc_by_block[b]
        trace.append({"block": b, "included_in_block": inc_by_block[b],
                      "running_included": running,
                      "authorised_by_extension_rule": authorised})
        if running >= 75 and stop_after is None:
            stop_after = b
    prereg = [t["block"] for t in trace if t["authorised_by_extension_rule"]]
    posthoc = [t["block"] for t in trace if not t["authorised_by_extension_rule"]]

    # ---- endpoint machinery ------------------------------------------------
    def subset(blocks_in):
        return [r for r in pool if r["block"] in blocks_in]

    def flow(rs):
        st = collections.Counter(status_of(r) for r in rs)
        inc, unre, exc = (st["included_reachable"], st["eligible_unreachable"],
                          st["excluded"])
        elig = inc + unre
        stage1_excl = sum(1 for r in rs if r.get("stage1_decision") == "exclude")
        excluded = [r for r in rs if status_of(r) == "excluded"]
        code = lambda rows: dict(collections.Counter(
            str(r.get("exclusion_code")) for r in rows).most_common())
        at1 = [r for r in excluded if r.get("stage1_decision") == "exclude"]
        at2 = [r for r in excluded if r.get("stage1_decision") != "exclude"]
        return {
            "records_screened": len(rs),
            "excluded_at_stage1_title_abstract": stage1_excl,
            "reports_sought_for_retrieval": len(rs) - stage1_excl,
            "reports_assessed_for_eligibility": len(rs) - stage1_excl - unre,
            "excluded_at_fulltext": len(at2),
            "eligible_looking": elig,
            "unreachable_eligibility_unresolved": unre,
            "included_and_reachable": inc,
            "excluded_total": exc,
            "excluded_by_code_total": code(excluded),
            "excluded_by_code_at_stage1": code(at1),
            "excluded_by_code_at_fulltext": code(at2),
            "unreachable_pct_of_eligible": (100.0 * unre / elig) if elig else None,
            "S6_unreachable": wilson(unre, elig),
        }

    def endpoints(rs, label):
        inc = [r for r in rs if status_of(r) == "included_reachable"]
        unre = [r for r in rs if status_of(r) == "eligible_unreachable"]
        elig = len(inc) + len(unre)
        p1 = [p1_of(r) for r in inc]
        k = sum(1 for v in p1 if v is True)
        # evidence-restricted: the codebook forbids an unevidenced negative, so
        # a complete-case negative only counts when the 14-term search over the
        # full text AND the supplement is on record.
        ev = [r for r in inc if str(r.get("searches_run") or "").strip()
              and _search_complete(r)]
        out = {
            "label": label,
            "n_eligible": elig,
            "n_included_reachable": len(inc),
            "n_unreachable": len(unre),
            "P1_complete_case": wilson(k, len(inc)),
            "P1_complete_case_evidence_restricted": wilson(
                sum(1 for r in ev if p1_of(r) is True), len(ev)),
            "P1_bound_lower_unreachable_all_negative": wilson(k, elig),
            "P1_bound_upper_unreachable_all_positive": wilson(k + len(unre), elig),
            "S6_unreachable": wilson(len(unre), elig),
            "unreachable_exceeds_15pct_threshold": (
                (100.0 * len(unre) / elig) > 15.0) if elig else None,
        }
        out["headline_per_protocol_section_7"] = (
            "BOUNDING INTERVAL [%.1f%%, %.1f%%] over %d eligible papers "
            "(complete-case point estimate %s is reported but is NOT the headline)"
            % (out["P1_bound_lower_unreachable_all_negative"]["pct"],
               out["P1_bound_upper_unreachable_all_positive"]["pct"], elig,
               fmt(out["P1_complete_case"]))
            if out["unreachable_exceeds_15pct_threshold"] else
            "complete-case point estimate %s" % fmt(out["P1_complete_case"]))
        out.update(secondaries(inc, unre))
        return out

    # Phrases that mark a supplement which EXISTS but could not be searched.
    # "none exists" is NOT one of them: there is nothing to search, so the
    # 14-term requirement is satisfied.
    _SUPP_UNSEARCHED = (
        "declared but not obtained",
        "no searchable text",
        "none retrievable",
        "not obtained",
        "could not be obtained",
        "could not be retrieved",
    )

    def _search_complete(r):
        """screen_frame.json -> reading_effort: a trivial_baseline coded
        all-false is only an EVIDENCED negative when the 14 named terms were run
        over the full text AND the supplement.  This returns False when the
        supplement exists but was not searched."""
        fs = r.get("fulltext_search")
        if fs is not None:              # analysis-sample consensus field
            return fs == "complete"
        s = r.get("searches_run")
        if not s:
            return False
        blob = (s if isinstance(s, str) else json.dumps(s)).lower()
        return not any(t in blob for t in _SUPP_UNSEARCHED)

    def secondaries(inc, unre):
        """Endpoint definitions are taken verbatim from screen_frame.json ->
        endpoints, including the exact enum level each one names."""
        n = len(inc)
        # S2: headline_unit='slice', OR evaluation_unit_reported='slice' with
        # only one unit reported.
        s2 = [r for r in inc
              if r.get("headline_unit") == "slice"
              or (r.get("evaluation_unit_reported") == "slice"
                  and r.get("headline_unit") == "na_only_one_unit_reported")]
        # S3: denominator = evaluation_unit_reported in {slice, both}
        any_slice = [r for r in inc
                     if r.get("evaluation_unit_reported") in ("slice", "both")]
        also_patient = [r for r in any_slice
                        if r.get("evaluation_unit_reported") == "both"]
        subj_split = [r for r in inc if r.get("split_unit") == "patient_subject"]
        # S5: {figure_or_table, text_with_numbers}. 'text_qualitative_only' is a
        # separate level and is NOT in the endpoint's numerator; it is reported
        # on its own line so the reader can see the looser reading too.
        posdist = [r for r in inc if r.get("positional_distribution_reported")
                   in ("figure_or_table", "text_with_numbers")]
        posdist_loose = [r for r in inc
                         if r.get("positional_distribution_reported")
                         in ("figure_or_table", "text_with_numbers",
                             "text_qualitative_only")]
        clustered = [r for r in inc if r.get("uncertainty_interval_reported")
                     == "ci_clustered_by_subject"]
        # S9 as the codebook pins it: BOTH counts reported.
        pos_both = [r for r in inc
                    if r.get("n_positive_reported") == "patients_and_slices"]
        # the looser reading of the same sentence in protocol sec.8
        pos_any_patient = [r for r in inc if r.get("n_positive_reported")
                           in ("patients_only", "patients_and_slices")]
        s7 = collections.Counter(
            (str(r.get("headline_unit")), str(p1_of(r))) for r in inc)
        return {
            "S1_any_non_imaging_baseline": wilson(
                sum(1 for r in inc if s1_of(r) is True), n),
            "S2_headline_unit_is_slice": wilson(len(s2), n),
            "S3_slice_reporters_also_reporting_patient": wilson(
                len(also_patient), len(any_slice)),
            "S4_explicit_subject_level_split": wilson(len(subj_split), n),
            "S5_positional_distribution_reported": wilson(len(posdist), n),
            "S5_positional_distribution_incl_qualitative_text": wilson(
                len(posdist_loose), n),
            "S7_headline_unit_x_P1": {"%s|P1=%s" % kk: vv for kk, vv in
                                      sorted(s7.items())},
            "S8_subject_clustered_interval": wilson(len(clustered), n),
            "S9_n_positive_patients_and_slices": wilson(len(pos_both), n),
            "S9_alt_any_patient_level_positive_count": wilson(
                len(pos_any_patient), n),
            "_distributions": {
                f: dict(collections.Counter(str(r.get(f)) for r in inc).most_common())
                for f in ("evaluation_unit_reported", "headline_unit", "split_unit",
                          "split_disjointness_verified",
                          "positional_distribution_reported",
                          "uncertainty_interval_reported", "n_positive_reported",
                          "input_representation", "label_broadcast_to_slices",
                          "dataset_public", "code_availability")
            },
        }

    # ---- censoring-free statement -----------------------------------------
    all_records = []
    for rec in main_doc["records"]:
        all_records.append(("main/%s" % rec["screener_id"], rec["amended"]))
    for b in ("R1", "R2", "R3", "R4"):
        for r in docs[b]["records"]:
            all_records.append((b, r))
    subflag_true = collections.defaultdict(list)
    for src, r in all_records:
        tb = r.get("trivial_baseline") or {}
        for f in ALL_SUBFLAGS:
            if tb.get(f) is True:
                subflag_true[f].append(str(r.get("record_id")))
    censoring_free = {
        "n_coded_records": len(all_records),
        "n_distinct_papers": len(pool) + len(dupes),
        "P1_family_subflags_true_anywhere": {f: sorted(set(subflag_true[f]))
                                             for f in P1_SUBFLAGS},
        "P1_family_true_count": sum(len(set(subflag_true[f])) for f in P1_SUBFLAGS),
        "S1_only_subflags_true_anywhere": {
            f: sorted(set(subflag_true[f]))
            for f in ("clinical_or_demographic_only", "other_non_imaging")},
    }
    censoring_free["statement"] = (
        "Across all %d independently coded records covering %d distinct sampled "
        "papers, the four zero-image sub-flags -- constant/prevalence, positional, "
        "acquisition-metadata, permuted-label -- are TRUE in %d records. This "
        "statement needs no denominator and is unaffected by unreachability."
        % (censoring_free["n_coded_records"], len(pool),
           censoring_free["P1_family_true_count"]))

    # ---- exploratory subgroups --------------------------------------------
    def subgroups(rs):
        inc = [r for r in rs if status_of(r) == "included_reachable"]
        unre = [r for r in rs if status_of(r) == "eligible_unreachable"]
        out = {}

        def cut(name, keyfn, rows_inc, rows_unre):
            """rows_unre is passed only for strata derivable from the record's
            PubMed metadata (year, venue).  Strata read out of the extraction
            form -- modality, dataset_public, evaluation_unit -- cannot be coded
            for a paper nobody could open, so those cuts are included-only and
            carry no S6 column rather than a spurious 'not_applicable' stratum."""
            g = collections.defaultdict(lambda: {"inc": [], "unre": []})
            for r in rows_inc:
                g[keyfn(r)]["inc"].append(r)
            for r in rows_unre:
                g[keyfn(r)]["unre"].append(r)
            out[name] = {}
            for k in sorted(g, key=str):
                i, u = g[k]["inc"], g[k]["unre"]
                cell = {"n_included": len(i),
                        "P1_complete_case": wilson(
                            sum(1 for r in i if p1_of(r) is True), len(i))}
                if rows_unre:
                    cell["n_unreachable"] = len(u)
                    cell["P1_upper_bound"] = wilson(
                        sum(1 for r in i if p1_of(r) is True) + len(u),
                        len(i) + len(u))
                    cell["S6_unreachable"] = wilson(len(u), len(i) + len(u))
                out[name][str(k)] = cell

        def year(r):
            m = META.get(r["record_id"], {})
            y = m.get("year") or r.get("year")
            try:
                y = int(y)
            except (TypeError, ValueError):
                return "year_unknown"
            return "2019-2022" if y <= 2022 else "2023-2026"

        def venue(r):
            m = META.get(r["record_id"], {})
            v = (m.get("venue") or r.get("venue") or "").lower()
            eng = ("comput", "ieee", "sensors", "electron", "inform", "eng",
                   "sci rep", "scientific reports", "math", "signal", "phys",
                   "entropy", "plos one", "expert syst", "neural", "bioeng",
                   "technol", "j imaging", "diagnostics")
            clin = ("radiol", "eur j", "clin", "med", "neuro", "oncol", "cancer",
                    "acad", "j nucl", "ajnr", "surg", "hepat", "cardio",
                    "ultraso", "chest", "thorac", "abdom", "spine", "arthritis",
                    "ophthal", "retina", "urol", "gastro", "endocr", "diabet",
                    "psychiatr", "dement", "alzheim", "stroke", "insights")
            if any(t in v for t in clin):
                return "clinical_or_radiology"
            if any(t in v for t in eng):
                return "engineering_or_computing"
            return "unclassified"

        cut("year_of_publication", year, inc, unre)
        cut("venue_class_heuristic", venue, inc, unre)
        cut("modality", lambda r: r.get("modality") or "unknown", inc, [])
        cut("dataset_public", lambda r: str(r.get("dataset_public")), inc, [])
        cut("evaluation_unit_reported",
            lambda r: str(r.get("evaluation_unit_reported")), inc, [])
        out["_venue_class_caveat"] = (
            "The venue classifier here is a keyword heuristic run over the "
            "journal name, NOT the pre-specified reading of each journal's scope "
            "statement. It is reported as exploratory-and-provisional; the "
            "protocol's own classification remains an outstanding action."
        )
        return out

    # ---- assemble ----------------------------------------------------------
    prereg_pool = subset(prereg)
    full_pool = subset(order)

    result = {
        "_generated_by": "paper/screen/analysis/pool_final.py",
        "_inputs": ["paper/screen_recoded.json",
                    "paper/screen_reserve_R1.json", "paper/screen_reserve_R2.json",
                    "paper/screen_reserve_R3.json", "paper/screen_reserve_R4.json",
                    "paper/screen/analysis/adjudication_out.json",
                    "paper/screen_sample.json", "paper/screen_frame.json"],
        "_interval_method": "Wilson score 95% two-sided, pre-specified in "
                            "screen_protocol.md sec.8 for every proportion.",
        "deduplication": {"n_records_pooled": len(pool),
                          "duplicates_dropped": dupes,
                          "n_duplicates": len(dupes),
                          "blocks": {b: len(blocks[b]) for b in order}},
        "codebook_conformance": {
            "D1_violations_unreachable_coded_excluded": viol,
            "D1_note": "Empty list means no paper was excluded at full text "
                       "without the full text having been obtained.",
        },
        "extension_rule": {
            "rule": "screen_protocol.md sec.3.1 -- screen positions 1-100; if "
                    "included papers < 75, continue into the reserve in "
                    "permutation order in blocks of 50 until 75 included papers "
                    "or position 400, whichever first; started blocks are never "
                    "truncated.",
            "trace": trace,
            "stopped_after_block": stop_after,
            "pre_registered_blocks": prereg,
            "post_hoc_blocks": posthoc,
            "verdict": ("The rule fired and stopped after block %s. Blocks %s "
                        "were screened beyond the stopping point and are a "
                        "POST-HOC extension; they are reported separately and "
                        "never silently pooled into the pre-registered "
                        "denominator." % (stop_after, posthoc or "(none)")),
        },
        "flow_pre_registered": flow(prereg_pool),
        "flow_with_post_hoc_R4": flow(full_pool),
        "flow_by_block": {b: flow(blocks[b]) for b in order},
        "primary_pre_registered": endpoints(prereg_pool, "PRE-REGISTERED (main + R1 + R2 + R3)"),
        "primary_with_post_hoc_R4": endpoints(full_pool, "POST-HOC EXTENDED (+ R4)"),
        "endpoints_by_block": {b: endpoints(blocks[b], b) for b in order},
        "censoring_free": censoring_free,
        "subgroups_exploratory_pre_registered": subgroups(prereg_pool),
        "subgroups_exploratory_with_R4": subgroups(full_pool),
        "unsettled_fields_in_overlap_papers": {
            r["record_id"]: r["unsettled_fields"] for r in pool
            if r.get("unsettled_fields")},
    }

    # agreement, carried through unmodified
    adj = json.load(open(P("paper", "screen", "analysis", "adjudication_out.json")))
    result["agreement"] = {
        "_source": "paper/screen/analysis/adjudication_out.json",
        "_n_items": 15, "_n_raters": 4, "_bootstrap": "percentile, 2000 resamples, seed 20260729",
        "pre_reconciliation_v1_0": adj["pre_reconciliation_v10"],
        "counterfactual_v1_2_encoding": adj["counterfactual_v12_encoding"],
        "restricted_to_core_six": adj["restricted_to_core"],
        "floor": adj["floor"],
        "what_the_floor_is_assessed_against":
            "The counterfactual v1.2 encoding of the SAME four sealed files. It "
            "is a re-expression, not an independent re-rating. A genuine "
            "post-amendment reliability estimate requires a fresh independent "
            "four-screener re-coding under v1.2, which has NOT been run.",
        "reserve_block_screeners": {b: docs[b].get("screener_id") for b in
                                    ("R1", "R2", "R3", "R4")},
        "reserve_blocks_are_single_coded":
            "Every reserve record was coded by ONE screener, so no reserve "
            "record contributes any agreement information. All 4-rater "
            "reliability in this paper rests on the 15 overlap papers of the "
            "original analysis sample.",
    }
    out = P("paper", "screen", "analysis", "pooled_final.json")
    json.dump(result, open(out, "w"), indent=1, default=str)

    # ---- console report ---------------------------------------------------
    W = sys.stdout.write
    W("\n=== DEDUP =========================================================\n")
    W("pooled records: %d  duplicates dropped: %d\n" % (len(pool), len(dupes)))
    for d in dupes:
        W("   dup %s (first in %s, again in %s)\n"
          % (d["pmid"], d["first_block"], d["duplicate_block"]))
    W("D1 violations: %s\n" % (viol or "none"))

    W("\n=== EXTENSION RULE ================================================\n")
    for t in trace:
        W("  %-5s +%-3d running=%-4d %s\n" % (
            t["block"], t["included_in_block"], t["running_included"],
            "authorised" if t["authorised_by_extension_rule"] else "POST-HOC"))
    W("  stopped after: %s\n" % stop_after)

    for name, key in (("PRE-REGISTERED", "flow_pre_registered"),
                      ("WITH POST-HOC R4", "flow_with_post_hoc_R4")):
        f = result[key]
        W("\n=== FLOW %s ====================================\n" % name)
        for k in ("records_screened", "excluded_at_stage1_title_abstract",
                  "reports_sought_for_retrieval",
                  "unreachable_eligibility_unresolved",
                  "reports_assessed_for_eligibility", "excluded_at_fulltext",
                  "included_and_reachable", "eligible_looking",
                  "excluded_total"):
            W("  %-38s %s\n" % (k, f[k]))
        W("  %-38s %.1f%%\n" % ("unreachable_pct_of_eligible",
                                f["unreachable_pct_of_eligible"]))
        W("  codes at stage 1 : %s\n" % f["excluded_by_code_at_stage1"])
        W("  codes at full text: %s\n" % f["excluded_by_code_at_fulltext"])

    for key in ("primary_pre_registered", "primary_with_post_hoc_R4"):
        e = result[key]
        W("\n=== %s ==========================\n" % e["label"])
        for k in ("P1_complete_case", "P1_complete_case_evidence_restricted",
                  "P1_bound_lower_unreachable_all_negative",
                  "P1_bound_upper_unreachable_all_positive", "S6_unreachable",
                  "S1_any_non_imaging_baseline", "S2_headline_unit_is_slice",
                  "S3_slice_reporters_also_reporting_patient",
                  "S4_explicit_subject_level_split",
                  "S5_positional_distribution_reported",
                  "S5_positional_distribution_incl_qualitative_text",
                  "S8_subject_clustered_interval",
                  "S9_n_positive_patients_and_slices",
                  "S9_alt_any_patient_level_positive_count"):
            W("  %-52s %s\n" % (k, fmt(e[k])))
        W("  15%% threshold breached: %s\n" % e["unreachable_exceeds_15pct_threshold"])
        W("  HEADLINE: %s\n" % e["headline_per_protocol_section_7"])
        W("  S7 headline_unit x P1: %s\n" % e["S7_headline_unit_x_P1"])

    W("\n=== PER BLOCK =====================================================\n")
    for b in order:
        f, e = result["flow_by_block"][b], result["endpoints_by_block"][b]
        W("  %-5s screened %3d  incl %2d  unre %2d  excl %2d  unre%% %5.1f  P1 %s\n"
          % (b, f["records_screened"], f["included_and_reachable"],
             f["unreachable_eligibility_unresolved"], f["excluded_total"],
             f["unreachable_pct_of_eligible"], fmt(e["P1_complete_case"])))

    W("\n=== CENSORING-FREE ================================================\n")
    W("  " + censoring_free["statement"] + "\n")
    for f, ids in censoring_free["S1_only_subflags_true_anywhere"].items():
        W("  %s TRUE in: %s\n" % (f, ids or "none"))

    W("\n=== AGREEMENT (15 overlap papers, 4 screeners) ====================\n")
    for tag, sec in (("pre v1.0", "pre_reconciliation_v10"),
                     ("counterfactual v1.2", "counterfactual_v12_encoding")):
        p = adj[sec]["P1_flag"]
        ci = p["bootstrap_ci95"]
        W("  %-20s raw %.3f %s | Fleiss k %.3f %s | AC1 %.3f %s | unan %d/15\n" % (
            tag, p["pa"], [round(x, 3) for x in ci["pa"][:2]],
            p["fleiss_kappa"], [round(x, 3) for x in ci["fleiss_kappa"][:2]],
            p["gwet_ac1"], [round(x, 3) for x in ci["gwet_ac1"][:2]],
            p["unanimous_items"]))
    W("  floor met post-amendment: kappa %.3f >= 0.60 -> %s ; raw %.3f >= 0.90 -> %s\n"
      % (adj["floor"]["post_kappa"], adj["floor"]["post_kappa"] >= 0.60,
         adj["floor"]["post_raw"], adj["floor"]["post_raw"] >= 0.90))

    W("\nwrote %s\n" % out)
    return result


if __name__ == "__main__":
    main()

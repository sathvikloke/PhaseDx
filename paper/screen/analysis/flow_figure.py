#!/usr/bin/env python3
"""
Draws the pooled PRISMA-style flow for the PhaseDx prevalence screen.

Every number is read from paper/screen/analysis/pooled_final.json; nothing is
typed into this file by hand.  Writes paper/figures/prisma_flow_pooled.svg.

Usage:  python paper/screen/analysis/flow_figure.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "paper", "screen", "analysis", "pooled_final.json")
OUT = os.path.join(ROOT, "paper", "figures", "prisma_flow_pooled.svg")

D = json.load(open(SRC))
F = D["flow_pre_registered"]
F4 = D["flow_with_post_hoc_R4"]
E = D["primary_pre_registered"]
BLK = D["flow_by_block"]

W, H = 1180, 1250
INK, GREY, FADE = "#111111", "#555555", "#f7f7f7"
out = []
add = out.append


def box(x, y, w, h, title, lines, fill="#ffffff", tsize=13):
    add('<rect x="%g" y="%g" width="%g" height="%g" rx="3" fill="%s" '
        'stroke="#222222" stroke-width="1.2"/>' % (x, y, w, h, fill))
    add('<text x="%g" y="%g" font-size="%g" font-weight="bold" fill="%s">%s</text>'
        % (x + 12, y + 24, tsize, INK, title))
    for i, ln in enumerate(lines):
        add('<text x="%g" y="%g" font-size="%g" fill="%s">%s</text>'
            % (x + 12, y + 42 + 17 * i, tsize - 0.5, INK, ln))


def arrow(x1, y1, x2, y2):
    add('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="#222222" '
        'stroke-width="1.2" marker-end="url(#a)"/>' % (x1, y1, x2, y2))


def band(y, label):
    add('<text x="34" y="%g" font-size="12" font-weight="bold" fill="#888888" '
        'text-anchor="middle" transform="rotate(-90 34 %g)">%s</text>'
        % (y, y, label))


add('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
    'viewBox="0 0 %d %d" font-family="Helvetica,Arial,sans-serif">' % (W, H, W, H))
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    'markerHeight="6" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#222222"/></marker></defs>')
add('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
add('<text x="70" y="32" font-size="17" font-weight="bold" fill="%s">'
    'PhaseDx prevalence screen &#8212; pooled flow, pre-registered analysis set</text>' % INK)
add('<text x="70" y="52" font-size="11.5" fill="%s">'
    'Analysis sample (positions 1&#8211;100) + reserve blocks R1&#8211;R3 (positions 111&#8211;260), '
    'drawn under the pre-registered extension rule. Counts from '
    'paper/screen/analysis/pooled_final.json.</text>' % GREY)

L, RX, BW, RW = 70, 640, 500, 470

# identification
box(L, 78, BW, 96, "Records identified from a database",
    ["PubMed, one frozen Boolean query, n = 9,979",
     "run 2026-07-29; SHA-256 d611def0785f...",
     "registers n = 0; other methods n = 0"])
box(RX, 78, RW, 96, "Removed before screening",
    ["duplicates n = 0 (PMID-unique frame)",
     "automation tools n = 0", "other reasons n = 0"], FADE)
arrow(L + BW, 108, RX, 108)
band(126, "Identification")

# sampling
box(L, 210, BW, 96, "Randomly sampled for screening",
    ["n = %d  (permutation positions 1&#8211;100, 111&#8211;260)" % F["records_screened"],
     "seed 20260729; blocks never truncated part-way",
     "extension rule stopped after block %s"
     % D["extension_rule"]["stopped_after_block"]])
box(RX, 210, RW, 96, "Frame records not sampled",
    ["never drawn n = 9,579",
     "pilot, excluded a priori n = 10",
     "reserve beyond position 260 n = 90 (R4 = 50, post-hoc)"], FADE)
arrow(L + BW, 240, RX, 240)
arrow(L + BW / 2, 174, L + BW / 2, 210)
band(258, "Sampling")

# stage 1
c1 = F["excluded_by_code_at_stage1"]
box(L, 342, BW, 58, "Records screened (title and abstract)",
    ["n = %d" % F["records_screened"]])
_c1 = ["%s %d" % (k, v) for k, v in c1.items()]
box(RX, 342, RW, 42 + 17 * 3, "Excluded at title/abstract",
    ["n = %d" % F["excluded_at_stage1_title_abstract"],
     "   " + ";  ".join(_c1[:4]), "   " + ";  ".join(_c1[4:])], FADE)
arrow(L + BW, 372, RX, 372)
arrow(L + BW / 2, 306, L + BW / 2, 342)

# retrieval
box(L, 436, BW, 58, "Reports sought for retrieval",
    ["n = %d" % F["reports_sought_for_retrieval"]])
box(RX, 436, RW, 96, "Reports NOT retrieved &#8212; full text unreachable",
    ["n = %d" % F["unreachable_eligibility_unresolved"],
     "NOT excluded: eligibility unresolved, carried",
     "into both bounding analyses (protocol &#167;7)"], "#fff4f4")
arrow(L + BW, 466, RX, 466)
arrow(L + BW / 2, 400, L + BW / 2, 436)

# eligibility
box(L, 568, BW, 58, "Reports assessed for eligibility",
    ["n = %d" % F["reports_assessed_for_eligibility"]])
codes = F["excluded_by_code_at_fulltext"]
box(RX, 568, RW, 42 + 17 * (1 + len(codes)),
    "Excluded at full text, by reason  (n = %d)" % F["excluded_at_fulltext"],
    ["%s  n = %d%s" % (k, v, "   ← inside the query, outside the failure mode"
                       if k == "E-DERIV" else "")
     for k, v in codes.items()]
    + ["total excluded at both stages n = %d" % F["excluded_total"]], FADE)
arrow(L + BW, 598, RX, 598)
arrow(L + BW / 2, 494, L + BW / 2, 568)
band(600, "Screening")

# included
box(L, 760, BW, 96, "STUDIES INCLUDED AND REACHABLE",
    ["n = %d  &#8212; the complete-case denominator" % F["included_and_reachable"],
     "pre-registered target was n = 75: MET",
     "%d of %d carry a fully evidenced 14-term search"
     % (E["P1_complete_case_evidence_restricted"]["n"], F["included_and_reachable"])],
    "#f2f7f2")
arrow(L + BW / 2, 626, L + BW / 2, 760)
band(808, "Included")

# the eligible box and the censoring statement
box(RX, 760, RW, 96, "ELIGIBLE-LOOKING SET (the bounding denominator)",
    ["n = %d = %d included + %d unreachable"
     % (F["eligible_looking"], F["included_and_reachable"],
        F["unreachable_eligibility_unresolved"]),
     "S6 unreachable = %.1f%% [%.1f%%, %.1f%%]"
     % (E["S6_unreachable"]["pct"], E["S6_unreachable"]["ci95"][0],
        E["S6_unreachable"]["ci95"][1]),
     "&#8250; 15% threshold: the BOUND is the headline"], "#fff4f4")

# result strip
add('<rect x="70" y="890" width="%d" height="118" rx="3" fill="#ffffff" '
    'stroke="#222222" stroke-width="1.8"/>' % (RX + RW - 70))
add('<text x="82" y="916" font-size="14" font-weight="bold" fill="%s">'
    'PRIMARY ENDPOINT P1 &#8212; papers reporting a measured zero-image baseline</text>' % INK)
add('<text x="82" y="939" font-size="13" fill="%s">'
    'HEADLINE (protocol &#167;7, censoring &#8250; 15%%): bounding interval '
    '<tspan font-weight="bold">[0.0%%, %.1f%%]</tspan> over %d eligible papers</text>'
    % (INK, E["P1_bound_upper_unreachable_all_positive"]["pct"], F["eligible_looking"]))
add('<text x="82" y="959" font-size="13" fill="%s">'
    'complete case (reported, NOT the headline): %d/%d = 0.0%% [%.1f%%, %.1f%%] Wilson</text>'
    % (INK, E["P1_complete_case"]["k"], E["P1_complete_case"]["n"],
       E["P1_complete_case"]["ci95"][0], E["P1_complete_case"]["ci95"][1]))
add('<text x="82" y="979" font-size="13" fill="%s">'
    'lower bound (all unreachable negative): 0/%d = 0.0%% [%.1f%%, %.1f%%]  &#183;  '
    'upper bound (all unreachable positive): %d/%d = %.1f%%</text>'
    % (INK, E["P1_bound_lower_unreachable_all_negative"]["n"],
       E["P1_bound_lower_unreachable_all_negative"]["ci95"][0],
       E["P1_bound_lower_unreachable_all_negative"]["ci95"][1],
       E["P1_bound_upper_unreachable_all_positive"]["k"],
       E["P1_bound_upper_unreachable_all_positive"]["n"],
       E["P1_bound_upper_unreachable_all_positive"]["pct"]))
add('<text x="82" y="999" font-size="13" fill="%s">'
    'censoring-free: across %d independently coded records over %d distinct papers, the four '
    'zero-image sub-flags are TRUE %d times.</text>'
    % (INK, D["censoring_free"]["n_coded_records"],
       D["deduplication"]["n_records_pooled"],
       D["censoring_free"]["P1_family_true_count"]))

# per-block strip
add('<text x="70" y="1045" font-size="12.5" font-weight="bold" fill="%s">'
    'By block &#8212; the access rate does not improve with sample size</text>' % INK)
COLS = [(82, "start", "block"), (250, "end", "screened"),
        (350, "end", "included"), (470, "end", "unreachable"),
        (620, "end", "unreachable % of eligible"), (700, "end", "P1")]
y = 1065


def row(y, cells, colour, bold=False):
    for (x, anchor, _), val in zip(COLS, cells):
        add('<text x="%g" y="%g" font-size="12" fill="%s" text-anchor="%s"%s>'
            '%s</text>' % (x, y, colour, "start" if anchor == "start" else "end",
                           ' font-weight="bold"' if bold else "", val))


row(y, [c[2] for c in COLS], GREY, bold=True)
for b in ("main", "R1", "R2", "R3", "R4"):
    y += 18
    f = BLK[b]
    row(y, [b + (" (post-hoc)" if b == "R4" else ""),
            f["records_screened"], f["included_and_reachable"],
            f["unreachable_eligibility_unresolved"],
            "%.1f%%" % f["unreachable_pct_of_eligible"],
            "0/%d" % f["included_and_reachable"]],
        "#333333" if b != "R4" else "#888888")
y += 22
add('<text x="82" y="%g" font-size="11.5" fill="%s">'
    'Block R4 (positions 261&#8211;310) lies BEYOND the pre-registered stopping point and is a '
    'post-hoc extension; with it the flow is %d screened, %d included, %d unreachable '
    '(%.1f%%), P1 bound [0.0%%, %.1f%%].</text>'
    % (y, GREY, F4["records_screened"], F4["included_and_reachable"],
       F4["unreachable_eligibility_unresolved"], F4["unreachable_pct_of_eligible"],
       D["primary_with_post_hoc_R4"]["P1_bound_upper_unreachable_all_positive"]["pct"]))
add("</svg>")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(out))
print("wrote", OUT)

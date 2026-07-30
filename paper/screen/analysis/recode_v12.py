#!/usr/bin/env python3
"""RE-CODE every already-coded paper under codebook v1.2 (+ the v1.3 access-recovery overlay).

This is the SECOND HALF of the pre-registered agreement remedy
(screen_frame.json -> agreement.threshold_and_remedy): "a documented adjudication round is
held, the codebook is amended in the changelog, and EVERY already-coded paper is re-coded
under the amended codebook."

The four sealed batch files are READ-ONLY. Every amended code is produced here, keyed on
(screener_id, record_id), and every change carries the old code, the new code and the
amendment id that forced it.

    python paper/screen/analysis/recode_v12.py

writes paper/screen_recoded.json and paper/screen_recoded.md
"""
import json, math, os, collections

HERE  = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))

UNREACHABLE = ('unreachable_paywalled', 'unreachable_not_found')
SCREENERS   = ('S1', 'S2', 'S3', 'S4')
P1_KEYS     = ('constant_or_prevalence', 'positional',
               'acquisition_metadata', 'permuted_or_shuffled_label')
S1_KEYS     = P1_KEYS + ('clinical_or_demographic_only', 'other_non_imaging')
SUBFLAGS    = S1_KEYS

# D3: the fields that become 'not_applicable' where final_inclusion != 'included'.
# Verbatim from screen_frame.json fields[] -> the D3 _note block.
D3_FIELDS = ['evaluation_unit_reported', 'headline_unit', 'split_unit',
             'split_disjointness_verified', 'positional_distribution_reported',
             'dataset_public', 'modality', 'n_positive_reported', 'headline_metric',
             'headline_test_set', 'headline_value_scope', 'headline_selection_rule',
             'uncertainty_interval_reported', 'input_representation',
             'label_broadcast_to_slices', 'code_availability']
QUOTE_OF = {'evaluation_unit_reported': 'evaluation_unit_quote',
            'headline_unit': 'headline_unit_quote',
            'split_unit': 'split_quote',
            'positional_distribution_reported': 'positional_distribution_quote'}

# ----------------------------------------------------------------- manual rule decisions
# Every entry below is a judgement made by re-reading the sealed record's own quote against
# an amended rule. Nothing here is mechanical, so each carries its reasoning verbatim.
# Keyed (screener_id, record_id) -> list of (field, new_value, rule, basis)
MANUAL = {
 ('S1', '34003996'): [('exclusion_code', 'E-NOCLF', 'D10',
    "E-SEG's text is qualified 'with NO categorical class decision evaluated'. A categorical "
    "class decision IS evaluated here: the screener's own quote records a per-B-scan "
    "presence/absence decision scored by AUC (0.97/0.95/0.99). That decision is a threshold "
    "on the segmentation network's predicted fluid volume, i.e. it is not produced by a "
    "fitted classifier, so E-SEG's qualifier fails and the defining failure is I2 -> E-NOCLF. "
    "E-DERIV does not intervene because E-DERIV presupposes a fitted classifier whose input "
    "is a derived representation, and no classifier is fitted at all. The screener asked for "
    "adjudication on exactly this point in its own notes. Exclusion is unchanged.")],
 ('S1', '41274092'): [('exclusion_code', 'E-NOCLF', 'D10',
    "Deep-learning iterative reconstruction plus a 46-observer visual grading analysis; the "
    "only categorical decision is the observers' image-quality rating, scored with AUC 0.75. "
    "This is D10's own trigger verbatim -- a non-classification DL task plus a categorical "
    "decision made by HUMAN READERS -- so E-NOCLF, not E-SEG. The screener flagged the "
    "ordering question in its notes and said E-NOCLF was arguably the more informative code. "
    "E-NONMED (phantom-only) also applies but is later in the fixed order. Exclusion unchanged.")],
 ('S3', '34603980'): [('exclusion_code', 'E-NOCLF', 'D10',
    "A categorical class decision IS evaluated -- 'an analysis of classification change "
    "(dilatation versus no dilatation) ... An aneurysm was misclassified in 34/399 cases "
    "(8.5%)' -- but it is a threshold on a DL-measured diameter, not a fitted classifier. "
    "E-SEG's qualifier therefore fails; E-NOCLF is the code. The screener had already "
    "recorded the reasoning and flagged the record BORDERLINE. Exclusion unchanged.")],
 ('S3', '40093990'): [('split_unit', 'random_unit_not_stated', 'D6',
    "The splitting sentence is 'The matched data was then randomly split into a training set "
    "(75%) and an internal testing set (25%)' -- no patient-naming noun. The only other split "
    "sentence divides 'five groups (61 cases per group)', and 'case' is undefined in that "
    "sentence. The screener's own justification is 'the units divided are propensity-matched "
    "patient cases (Table 1 counts patients)', which is precisely the upgrade D6 forbids and "
    "precisely the move overturned on PMID 42130124 in the adjudication. DIRECTION: this "
    "REMOVES a paper from endpoint S4's numerator, i.e. it makes the literature look worse.")],
 ('S4', '40232605'): [('split_unit', 'lesion_or_roi', 'D8(a)',
    "The split is over 64x64x64 candidate patches: '1200 volumes (3D patches) that included "
    "600 nodules and 600 non-nodules' and '400 nodules and 400 non-nodules were randomly "
    "selected'. Under v1.0 there was no level for this and the screener was forced into "
    "slice_or_image; D8(a) adds lesion_or_roi for exactly this case. Same correction the "
    "adjudication applied to PMID 36776294. No endpoint moves: neither level is S4's numerator.")],
 ('S4', '40883444'): [('headline_unit', 'na_only_one_unit_reported', 'D14',
    "The screener's own quote is 'Only one, undefined, unit is reported.' D14: 'unclear' is "
    "available only where two or more units are reported; where exactly one is reported, "
    "'na_only_one_unit_reported' applies even if that unit is itself 'unclear' in "
    "evaluation_unit_reported. Same correction the adjudication applied to PMID 41068276.")],
}

# Records whose all-false trivial_baseline rests on a MAIN-TEXT-ONLY search because a
# supplement exists and could not be retrieved. Verbatim from each record's own searches_run.
SUPPLEMENT_NOT_SEARCHED = {
 ('S2', '40768653'): "supplement: Multimedia Appendices 1-3 exist but could not be downloaded (PMC supplementary endpoint blocked); main text only",
 ('S2', '38337016'): "supplement: Supplementary Information exists but could not be downloaded (PMC supplementary endpoint blocked); main text only",
 ('S2', '34003056'): "supplement: Appendix E1 exists but is paywalled and was not retrieved; main article searched in full",
 ('S3', '31093705'): "supplement: none available -- NOTE: 'Supplement 1' is referenced in the text and was not obtainable, so the negative is evidenced on the main text only.",
 ('S3', '32714766'): "supplement: none available -- Supporting Information (Figures S1-S5, Table S1) referenced but not obtainable; negative evidenced on the main text.",
 ('S4', '31093705'): "supplement: EXISTS and NOT retrieved -- 'Supplement 1' is referenced 6 times in the body. The negative on trivial_baseline is therefore main-text-only",
 ('S4', '41068276'): "supplement: referenced 3 times but not retrieved (PMC body only)",
 ('S4', '40232605'): "supplement: referenced 7 times but not retrieved (PMC body only)",
 ('S4', '35401411'): "supplement: referenced 9 times but not retrieved (PMC body only)",
 ('S4', '41568076'): "supplement: referenced 13 times and NOT retrieved - the montage construction and the aggregation rule are both deferred to the Supplementary Material, so the negative on trivial_baseline is main-text-only",
 ('S3', '38591974'): "supplement_status (recovery record): NOT OBTAINED. Appendix S1 and Tables S1-S3 are cited throughout the main text; the supplement URL 403s and has no archive snapshot. The 14-term search covers the complete main text and NOT the supplement.",
}
# Records read only as a preprint / accepted manuscript that itself carries no supplement.
# The negative is evidenced for the version read; flagged for the version-of-record analysis.
VERSION_CAVEAT = {
 ('S1', '35247336'): "read as the preprint version; 'supplement: none in the preprint'. Negative evidenced for the version read; carried into the version-of-record sensitivity analysis.",
 ('S4', '36200353'): "recovered as an Authorea PREPRINT, which carries no supplement. Negative evidenced for the version read; carried into the version-of-record sensitivity analysis.",
 ('S4', '39846055'): "recovered as the author's accepted manuscript; 'No supplement exists for this article.' Negative fully evidenced.",
}

# The adjudication round's consensus codes for the 15 overlap papers (paper/screen_adjudication.md,
# paper/screen/analysis/adjudication_out.json). These ARE the v1.2 answer for those papers.
ADJ_EXTRA = {  # fields the adjudication settled that adjudicated_codes does not carry as columns
 '40335658': {'exclusion_code': ('E-NOCLF', 'D10'), 'modality': ('MRI', 'D12'),
              'stage1_decision': ('exclude', 'D4/D11 (implied by the adjudicated fulltext code)')},
 '40194851': {'stage1_decision': ('exclude', 'D4/D11 (implied by the adjudicated fulltext code)'),
              'exclusion_code': ('E-DERIV', 'D4 (E-DERIV was unanimous; the code is restored '
                                 'with the record, because the adjudicated stage1_decision is '
                                 "'exclude' and D1's withdrawal therefore does not apply)")},
 '40239684': {'stage1_decision': ('exclude', 'D4/D11 (implied by the adjudicated fulltext code)')},
 '42489954': {'exclusion_code': (None, 'D1 + the existing stage-1 rule')},
}
# Sub-flag decisions the adjudication made explicitly, where they differ from "carry the
# screener's own TRUE across". Only 41617832 is affected: S1 and S4 coded a TRUE there and
# the adjudication REFUSED it, because the abstract names a clinical arm and reports no
# number for it -- the codebook's existing rule that an assertion with no number is FALSE
# for every flag, which D2 turns into 'not_assessable' rather than false.
ADJ_SUBFLAG = {
 ('41617832', 'clinical_or_demographic_only'): ('not_assessable', 'adjudication (D2)',
   "S1 and S4 coded TRUE; the adjudication overturned both. The abstract names a clinical "
   "logistic-regression arm but reports NO value for it, and D2 says an abstract that NAMES a "
   "pixel-free comparator without a number is 'not_assessable', not TRUE. Contrast PMID "
   "37222638, whose abstract carries three measured AUCs (0.741/0.772/0.675) and whose TRUE "
   "therefore stands."),
}
# Fields carried by the recovery overlay that are provenance, not codes. Logged separately so
# the code-change counts are not inflated by retrieval bookkeeping.
PROVENANCE_FIELDS = {
 'recoded_utc', 'citation', 'access_rung_succeeded', 'access_channel_note',
 'access_rungs_tried_and_failed', 'supplement_status', 'version_note',
 'version_divergence_warning', '_retrieval_urls', 'notes', 'screener_confidence',
 'flag_for_adjudication', 'headline_test_set_note', 'n_patients_note',
 'exclusion_secondary_note', 'split_observed_note', 'coded_utc', 'dataset_name',
 'organ_or_region',
}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------------------ load
PS = {}
for b in 'ABCD':
    d = json.load(open(f'{PAPER}/screen_batch_{b}.json'))
    PS[d['screener_id']] = d
    for r in d['records']:
        r['_source_file'] = f'paper/screen_batch_{b}.json'

REC = json.load(open(f'{PAPER}/screen/access_recovery.json'))
RECOVERY = {r['record_id']: r for r in REC['records']}
ADJ = json.load(open(f'{PAPER}/screen/analysis/adjudication_out.json'))['adjudicated_codes']

sealed = []
for s in SCREENERS:
    for r in PS[s]['records']:
        sealed.append(r)
assert len(sealed) == 145

OVERLAP = sorted({r['record_id'] for r in sealed if r['batch'] == 'overlap'})
assert len(OVERLAP) == 15


# ------------------------------------------------------------------------- recode one
def recode(r):
    s = r['screener_id']
    pmid = r['record_id']
    key = (s, pmid)
    a = json.loads(json.dumps(r))          # deep copy; sealed file untouched
    a.pop('_source_file', None)
    ch, prov = [], []
    preserved = {}

    def setf(field, new, rule, basis):
        old = a.get(field)
        if isinstance(old, dict) or isinstance(new, dict):
            same = json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
        else:
            same = old == new
        if same:
            return
        entry = dict(field=field, old=old, new=new, rule=rule, basis=basis)
        (prov if field in PROVENANCE_FIELDS or field.endswith('_quote') else ch).append(entry)
        a[field] = new

    # --- v1.3 ACCESS-RECOVERY OVERLAY (no codebook rule changed; new evidence) ----------
    recovered = False
    if pmid in RECOVERY and RECOVERY[pmid]['original_screener_id'] == s:
        rc = RECOVERY[pmid]
        recovered = True
        for field, v in rc.items():
            if field.startswith('_') or field in ('record_id', 'permutation_position',
                                                  'batch', 'original_screener_id'):
                continue
            setf(field, v, 'v1.3 access recovery',
                 f"full text recovered at {rc['access_rung_succeeded'].split(' -- ')[0]}; "
                 f"record re-coded on the full text, including the mandatory 14-term search")

    # --- D4: the access ladder is not climbed for a stage-1 exclusion ------------------
    if a['stage1_decision'] == 'exclude' and a['fulltext_reachable'] in UNREACHABLE:
        setf('fulltext_reachable', 'not_attempted_excluded_at_stage1', 'D4',
             "stage1_decision='exclude', so the ladder is not climbed and 'unreachable_*' is "
             "not available; 'unreachable_*' is reserved for records that reached stage 2 so "
             "that endpoint S6 measures the reachability of the ELIGIBLE-LOOKING literature.")

    # --- D1: unreachable dominates included -------------------------------------------
    if a['fulltext_reachable'] in UNREACHABLE:
        if a['final_inclusion'] == 'included':
            setf('final_inclusion', 'unreachable_eligibility_unresolved', 'D1',
                 "full text unreachable, so the mandatory 14-term search could not be run; an "
                 "'included' record must carry an EVIDENCED trivial_baseline code, so coding "
                 "this 'included' would admit an unevidenced negative into the P1 denominator.")
        elif a['final_inclusion'] == 'excluded':
            setf('final_inclusion', 'unreachable_eligibility_unresolved', 'D1',
                 "an unreachable record may be 'excluded' only where stage1_decision='exclude', "
                 "i.e. only where the exclusion code was declared unambiguous from the abstract "
                 "BEFORE the access attempt. Here stage1_decision='%s', so the exclusion rests "
                 "on evidence the screener had already declared insufficient. Same reading the "
                 "adjudication applied to PMID 42489954." % a['stage1_decision'])
            if a.get('exclusion_code') is not None:
                preserved['exclusion_code_withdrawn'] = a.get('exclusion_code')
                preserved['exclusion_quote_withdrawn'] = a.get('exclusion_quote')
                setf('exclusion_code', None, 'D1',
                     "provisional exclusion withdrawn with the record; the code and its quote "
                     "are preserved in preserved_evidence and are re-usable if the full text "
                     "is ever obtained.")

    # --- manual, quote-driven rule applications ---------------------------------------
    for field, new, rule, basis in MANUAL.get(key, []):
        setf(field, new, rule, basis)

    # --- the adjudicated consensus, for the 15 overlap papers -------------------------
    if pmid in ADJ:
        A = ADJ[pmid]
        setf('final_inclusion', A['final_inclusion'], 'adjudication (%s)' % A['rule'],
             A['note'])
        setf('fulltext_reachable', A['fulltext'], 'adjudication (%s)' % A['rule'], A['note'])
        for f, v in ADJ_EXTRA.get(pmid, {}).items():
            setf(f, v[0], 'adjudication (%s)' % v[1], A['note'])
        if A['final_inclusion'] == 'included':
            for f, k in (('evaluation_unit_reported', 'eval_unit'),
                         ('headline_unit', 'headline_unit'),
                         ('split_unit', 'split_unit'),
                         ('positional_distribution_reported', 'pos_dist')):
                setf(f, A[k], 'adjudication (%s)' % A['rule'], A['note'])

    incl = a['final_inclusion']

    # --- D2: three-valued trivial_baseline, positive/negative asymmetry ---------------
    tb = dict(a.get('trivial_baseline') or {})
    got_fulltext = a['fulltext_reachable'] not in UNREACHABLE and \
                   a['fulltext_reachable'] != 'not_attempted_excluded_at_stage1'
    supp_gap = key in SUPPLEMENT_NOT_SEARCHED
    search_state = ('not_run' if not got_fulltext else
                    ('main_text_only' if supp_gap else 'complete'))

    new_tb, tb_reason = {}, {}
    for k in SUBFLAGS:
        v = tb.get(k)
        if (pmid, k) in ADJ_SUBFLAG:
            new_tb[k], _rule, tb_reason[k] = ADJ_SUBFLAG[(pmid, k)]
        elif incl == 'excluded':
            new_tb[k] = 'not_applicable'
            tb_reason[k] = "D3: final_inclusion != 'included'; the record enters no P1 or S1 denominator."
        elif v is True:
            new_tb[k] = True
            tb_reason[k] = ("D2(a): TRUE stands. The quote carries a MEASURED value, which is "
                            "admissible evidence even on a record whose full text was never obtained.")
        elif incl == 'unreachable_eligibility_unresolved':
            new_tb[k] = 'not_assessable'
            tb_reason[k] = ("D2(c): the 14-term full-text search could not be run, so a FALSE is "
                            "not available; 'not_assessable', never false.")
        elif search_state == 'complete':
            new_tb[k] = False
            tb_reason[k] = "D2(b): the 14-term search was run over the full text and the supplement was resolved."
        elif search_state == 'main_text_only':
            new_tb[k] = 'not_assessable'
            tb_reason[k] = ("D2(b): a FALSE requires the 14-term search INCLUDING THE SUPPLEMENT. "
                            "A supplement exists and could not be retrieved (" +
                            SUPPLEMENT_NOT_SEARCHED[key] + "), so the negative is main-text-only "
                            "and is not admissible as a FALSE.")
        else:
            new_tb[k] = 'not_assessable'
            tb_reason[k] = "D2(c): no full-text search on record."
    if json.dumps(tb, sort_keys=True) != json.dumps(new_tb, sort_keys=True):
        ch.append(dict(field='trivial_baseline', old=tb, new=new_tb,
                       rule='D2' if incl != 'excluded' else 'D3',
                       basis=' | '.join(sorted(set(tb_reason.values())))))
    a['trivial_baseline'] = new_tb
    a['trivial_baseline_evidence'] = dict(fulltext_search=search_state,
                                          supplement_note=SUPPLEMENT_NOT_SEARCHED.get(key)
                                          or VERSION_CAVEAT.get(key) or 'resolved',
                                          per_subflag_reason=tb_reason)

    # --- D3: descriptive fields on a record that is not 'included' --------------------
    if incl != 'included':
        why = ("D3: final_inclusion='%s', so this field describes something that either does "
               "not exist or could not be observed. The record's own evidence is preserved "
               "verbatim in preserved_evidence." % incl)
        for f in D3_FIELDS:
            if a.get(f) != 'not_applicable':
                preserved[f] = a.get(f)
                setf(f, 'not_applicable', 'D3', why)
            qf = QUOTE_OF.get(f)
            if qf and a.get(qf):
                preserved[qf] = a.get(qf)
                a[qf] = why
        for f in ('mixed_modality', 'headline_value', 'headline_metric_qualifier'):
            if a.get(f) not in (None, False):
                preserved[f] = a.get(f)

    # --- D13: 'none' now means the paper explicitly declines to share code ------------
    if incl == 'included' and a.get('code_availability') == 'none':
        setf('code_availability', 'not_stated', 'D13',
             "v1.0 offered no 'not_stated' level, so a paper that simply says nothing about "
             "code had to be coded 'none'. Every record's own note describes silence (no code "
             "link found), not an explicit refusal; none records a declination. D13 makes "
             "'not_stated' the code for silence and reserves 'none' for an explicit refusal.")

    # Collapse the pipeline's intermediate steps into NET changes, one per field: the sealed
    # code, the final code, and the chain of rules that got there. A field that the pipeline
    # moved and then moved back (D1 withdrawing an exclusion the adjudication restores) is a
    # no-op and is dropped rather than reported as two changes.
    def collapse(entries):
        first, last, rules, basis = {}, {}, {}, {}
        order = []
        for c in entries:
            f = c['field']
            if f not in first:
                first[f] = c['old']
                rules[f] = []
                basis[f] = []
                order.append(f)
            last[f] = c['new']
            if c['rule'] not in rules[f]:
                rules[f].append(c['rule'])
                basis[f].append(c['basis'])
        outl = []
        for f in order:
            if json.dumps(first[f], sort_keys=True) == json.dumps(last[f], sort_keys=True):
                continue
            outl.append(dict(field=f, old=first[f], new=last[f],
                             rule=' + '.join(rules[f]), basis=' || '.join(basis[f])))
        return outl

    return a, collapse(ch), collapse(prov), preserved, search_state, recovered


# ------------------------------------------------------------------------------- run
records, all_changes = [], []
for r in sorted(sealed, key=lambda x: (x['permutation_position'], x['screener_id'])):
    a, ch, prov, preserved, search_state, recovered = recode(r)
    substantive = [c for c in ch if not (c['rule'] == 'D3' and c['new'] == 'not_applicable')]
    rec = dict(record_id=r['record_id'], permutation_position=r['permutation_position'],
               batch=r['batch'], screener_id=r['screener_id'],
               source_file=r['_source_file'],
               recoded_under='codebook v1.2 + v1.3 access-recovery overlay',
               n_changes=len(ch), n_substantive_changes=len(substantive),
               changes=ch, evidence_and_provenance_changes=prov, amended=a)
    if preserved:
        rec['preserved_evidence'] = preserved
    records.append(rec)
    for c in ch:
        all_changes.append(dict(record_id=r['record_id'], screener_id=r['screener_id'],
                                batch=r['batch'],
                                kind='D3_not_applicable_fill'
                                     if (c['rule'] == 'D3' and c['new'] == 'not_applicable')
                                     else 'substantive', **c))

# ------------------------------------------------------------------- paper-level view
by_pmid = collections.defaultdict(dict)
for rec in records:
    by_pmid[rec['record_id']][rec['screener_id']] = rec


def flag(tb, keys):
    vs = [tb.get(k) for k in keys]
    if any(v is True for v in vs):
        return True
    if all(v is False for v in vs):
        return False
    if all(v == 'not_applicable' for v in vs):
        return 'not_applicable'
    return 'not_assessable'


def status(a):
    if a['final_inclusion'] == 'excluded':
        return 'excluded'
    if a['fulltext_reachable'] in UNREACHABLE:
        return 'eligible_unreachable'
    return 'included_reachable'


AGREE_FIELDS = ['final_inclusion', 'evaluation_unit_reported', 'headline_unit',
                'split_unit', 'positional_distribution_reported']
papers = []
for pmid, per in by_pmid.items():
    ss = sorted(per)
    amended = [per[s]['amended'] for s in ss]
    row = dict(record_id=pmid, permutation_position=amended[0]['permutation_position'],
               batch=amended[0]['batch'], screeners=ss, n_screeners=len(ss),
               citation=amended[0].get('citation'))
    st = {status(a) for a in amended}
    assert len(st) == 1, (pmid, st)      # every rule above is deterministic per paper
    row['status_for_analysis'] = st.pop()
    row['final_inclusion'] = amended[0]['final_inclusion']
    row['fulltext_reachable'] = amended[0]['fulltext_reachable']
    row['exclusion_code'] = amended[0].get('exclusion_code')

    # Paper-level trivial_baseline. Screeners can still differ on the EVIDENCE (whether a
    # supplement exists and was searched), so the paper-level sub-flag takes:
    #   TRUE  if any screener has an admissible TRUE (D2(a): a positive needs one quote);
    #   FALSE only if EVERY screener's negative is fully evidenced (D2(b): a negative needs
    #         the 14-term search including the supplement, so one screener documenting an
    #         unsearched supplement is enough to defeat the paper-level FALSE);
    #   not_applicable / not_assessable otherwise.
    tb_paper, tb_src = {}, {}
    for k in SUBFLAGS:
        vals = [a['trivial_baseline'][k] for a in amended]
        if any(v is True for v in vals):
            tb_paper[k] = True
        elif all(v == 'not_applicable' for v in vals):
            tb_paper[k] = 'not_applicable'
        elif all(v is False for v in vals):
            tb_paper[k] = False
        else:
            tb_paper[k] = 'not_assessable'
        if len(set(map(str, vals))) > 1:
            tb_src[k] = {s: per[s]['amended']['trivial_baseline'][k] for s in ss}
    row['trivial_baseline'] = tb_paper
    if tb_src:
        row['subflag_evidence_split_between_screeners'] = tb_src
    row['P1'] = flag(tb_paper, P1_KEYS)
    row['S1'] = flag(tb_paper, S1_KEYS)
    searches = {a['trivial_baseline_evidence']['fulltext_search'] for a in amended}
    row['fulltext_search'] = ('not_run' if 'not_run' in searches else
                              'main_text_only' if 'main_text_only' in searches else 'complete')
    for f in AGREE_FIELDS[1:]:
        vals = {json.dumps(a.get(f)) for a in amended}
        row[f] = json.loads(vals.pop()) if len(vals) == 1 else 'residual_disagreement'
    # non-adjudicated descriptive fields can still differ between screeners on overlap papers
    resid = {}
    for f in ('split_disjointness_verified', 'dataset_public', 'modality',
              'n_positive_reported', 'headline_metric', 'headline_test_set',
              'headline_value_scope', 'headline_selection_rule',
              'uncertainty_interval_reported', 'input_representation',
              'label_broadcast_to_slices', 'code_availability'):
        vals = {json.dumps(a.get(f)) for a in amended}
        if len(vals) > 1:
            resid[f] = {s: per[s]['amended'].get(f) for s in ss}
    if resid:
        row['residual_between_screener_differences'] = resid
        row['residual_note'] = ("These fields were NOT agreement fields and were NOT settled by "
                                "the adjudication round. They are reported per screener rather "
                                "than collapsed, because collapsing them would invent a code.")
    row['n_changes'] = sum(per[s]['n_changes'] for s in ss)
    papers.append(row)
papers.sort(key=lambda x: x['permutation_position'])
assert len(papers) == 100

# ------------------------------------------------------------------------- endpoints
CC   = [p for p in papers if p['status_for_analysis'] == 'included_reachable']
UNRE = [p for p in papers if p['status_for_analysis'] == 'eligible_unreachable']
EXCL = [p for p in papers if p['status_for_analysis'] == 'excluded']
ELIG = CC + UNRE
CC_EV = [p for p in CC if p['P1'] is False]          # evidenced negative
CC_NA = [p for p in CC if p['P1'] == 'not_assessable']


def ep(k, n):
    p, lo, hi = wilson(k, n)
    return dict(k=k, n=n, pct=None if p is None else 100 * p,
                ci=None if p is None else [100 * lo, 100 * hi])


k_p1 = sum(1 for p in CC if p['P1'] is True)
k_s1 = sum(1 for p in CC if p['S1'] is True)
endpoints = dict(
    n_screened=len(papers), n_included_reachable=len(CC),
    n_eligible_unreachable=len(UNRE), n_excluded=len(EXCL), n_eligible=len(ELIG),
    n_complete_case_with_evidenced_P1_negative=len(CC_EV),
    n_complete_case_P1_not_assessable=len(CC_NA),
    S6_unreachable=ep(len(UNRE), len(ELIG)),
    P1_complete_case_all_reachable=ep(k_p1, len(CC)),
    P1_complete_case_evidence_restricted=ep(k_p1, len(CC_EV)),
    P1_lower=ep(k_p1, len(ELIG)),
    P1_upper=ep(k_p1 + len(UNRE), len(ELIG)),
    P1_upper_incl_not_assessable=ep(k_p1 + len(UNRE) + len(CC_NA), len(ELIG)),
    S1_complete_case=ep(k_s1, len(CC)),
    S4_subject_split=ep(sum(1 for p in CC if p['split_unit'] == 'patient_subject'), len(CC)),
    S5_positional=ep(sum(1 for p in CC if p['positional_distribution_reported']
                         in ('figure_or_table', 'text_with_numbers')), len(CC)),
    threshold_breached=len(UNRE) / len(ELIG) > 0.15,
    P1_TRUE_records=[p['record_id'] for p in papers if p['P1'] is True],
    S1_TRUE_records=[p['record_id'] for p in papers if p['S1'] is True],
)

# ------------------------------------------------- transitions against the sealed state
# The as-sealed paper-level state, computed with the SAME logic the published analysis used
# (paper/screen/analysis/recompute_with_recovery.py): majority status over the four overlap
# codes with ties broken toward eligible_unreachable, P1 true if any true / false if all false.
def sealed_status(r):
    if r['final_inclusion'] == 'excluded':
        return 'excluded'
    return 'eligible_unreachable' if r['fulltext_reachable'] in UNREACHABLE else 'included_reachable'


def sealed_flag(r, keys):
    vs = [(r.get('trivial_baseline') or {}).get(k) for k in keys]
    if any(v is True for v in vs):
        return True
    if all(v is False for v in vs):
        return False
    return 'not_assessable'


sealed_by_pmid = collections.defaultdict(list)
for r in sealed:
    sealed_by_pmid[r['record_id']].append(r)

transitions = []
for p in papers:
    srs = sealed_by_pmid[p['record_id']]
    c = collections.Counter(sealed_status(r) for r in srs)
    top = max(c.values())
    tied = [k for k, v in c.items() if v == top]
    for pref in ('eligible_unreachable', 'included_reachable', 'excluded'):
        if pref in tied:
            s_st = pref
            break
    vals = [sealed_flag(r, P1_KEYS) for r in srs]
    s_p1 = True if any(v is True for v in vals) else (
           False if all(v is False for v in vals) else 'not_assessable')
    if s_st != p['status_for_analysis'] or str(s_p1) != str(p['P1']):
        transitions.append(dict(record_id=p['record_id'],
                                permutation_position=p['permutation_position'],
                                batch=p['batch'],
                                status_sealed=s_st, status_amended=p['status_for_analysis'],
                                P1_sealed=s_p1, P1_amended=p['P1'],
                                fulltext_search=p['fulltext_search']))
transitions.sort(key=lambda x: x['permutation_position'])

# ---------------------------------------------------------------------------- summary
chg_by_field  = collections.Counter(c['field'] for c in all_changes)
chg_by_rule   = collections.Counter()
for c in all_changes:
    for part in c['rule'].split(' + '):
        chg_by_rule[part.split(' (')[0]] += 1
recs_changed  = sum(1 for r in records if r['n_changes'])
papers_changed = sum(1 for p in papers if p['n_changes'])

out = dict(
 _schema='phasedx.screen.recoded.v1',
 _what_this_is=(
   "Every already-coded record re-coded under codebook v1.2 (amendments D1-D14) plus the "
   "v1.3 access-recovery overlay. This is the second half of the pre-registered agreement "
   "remedy: a codebook amendment applied only to new papers would make old and new records "
   "incomparable, so screen_frame.json -> agreement.threshold_and_remedy requires EVERY "
   "already-coded paper to be re-coded."),
 _sealed_files_modified=False,
 _inputs=['paper/screen_batch_A.json', 'paper/screen_batch_B.json',
          'paper/screen_batch_C.json', 'paper/screen_batch_D.json',
          'paper/screen_frame.json (v1.2/v1.3)', 'paper/screen_adjudication.md',
          'paper/screen/analysis/adjudication_out.json', 'paper/screen/access_recovery.json'],
 _generated_by='paper/screen/analysis/recode_v12.py',
 _units=("145 coded RECORDS over 100 sampled PAPERS: 15 overlap papers coded by all four "
         "screeners, 85 papers coded once. Both views are given: 'records' is the "
         "record-level re-code, 'papers' the paper-level code that enters the endpoints."),
 _no_number_was_improved_by_a_rule_change=(
   "Direction audit. D1 moves 5 records OUT of 'excluded' and INTO the eligible-unreachable "
   "set, which RAISES the unreachability rate and WIDENS the bounding interval -- against us. "
   "D2's supplement clause moves complete-case records to 'not_assessable', which SHRINKS the "
   "evidenced denominator -- against us. D6 removes a paper from S4 -- against the literature. "
   "D9 (adjudication) adds a paper to S5 -- in FAVOUR of the literature, on the very endpoint "
   "this paper accuses the literature of ignoring. No rule was applied selectively."),
 summary=dict(
   n_records=len(records), n_records_changed=recs_changed,
   n_records_with_substantive_change=sum(1 for r in records if r['n_substantive_changes']),
   n_papers=len(papers), n_papers_changed=papers_changed,
   n_field_changes=len(all_changes),
   n_substantive_field_changes=sum(1 for c in all_changes if c['kind'] == 'substantive'),
   n_D3_not_applicable_fills=sum(1 for c in all_changes if c['kind'] != 'substantive'),
   changes_by_field=dict(chg_by_field.most_common()),
   changes_by_rule=dict(chg_by_rule.most_common()),
   moved_into_reports_a_zero_image_baseline=[t['record_id'] for t in transitions
                                             if t['P1_amended'] is True],
   moved_out_of_reports_a_zero_image_baseline=[t['record_id'] for t in transitions
                                               if t['P1_sealed'] is True],
   headline=("NO paper moved INTO 'reports a zero-image baseline'. The P1 numerator is 0 "
             "before the amendment, 0 after it, and 0 in every one of the 145 re-coded "
             "records: not one of the four P1 sub-flags (constant/prevalence, positional, "
             "acquisition-metadata, permuted-label) is TRUE anywhere in the sample.")),
 residual_gaps_found_during_the_recode=[
  dict(id='G1', severity='material',
       case="D2(b) requires the 14-term search to cover the SUPPLEMENT before a sub-flag may "
            "be coded FALSE. Ten complete-case papers were searched over the main text only, "
            "because a supplement exists and could not be retrieved (publisher paywall or a "
            "blocked PMC supplementary endpoint). Their negatives are therefore inadmissible "
            "and are re-coded 'not_assessable'.",
       records=['31093705', '41068276', '40768653', '38337016', '34003056', '32714766',
                '38591974', '40232605', '35401411', '41568076'],
       consequence="The evidenced complete-case P1 denominator falls from 38 to 28. This is a "
                   "state the protocol does not name: eligible, reachable, but the primary "
                   "flag is not assessable. missing_and_unreachable.rule_3 defines the bounding "
                   "analysis over UNREACHABLE papers only.",
       recommendation="Amend missing_and_unreachable so the bounding analysis is defined over "
                      "'papers whose P1 code is not an evidenced negative', which is the set "
                      "the rule was plainly meant to cover, and report both denominators."),
  dict(id='G2', severity='minor',
       case="code_availability has no level for a paper that PROMISES code on publication and "
            "gives no URL. D13 splits 'none' (explicit refusal) from 'not_stated' (silence); "
            "this record is neither.",
       records=['40232605'],
       quote="\"The code used in this work will be publicly available on GitHub upon "
             "publication of the paper\" -- no URL (screener S4's own note).",
       consequence="Coded 'not_stated' as the least misleading available level, with the "
                   "promise recorded here. No endpoint uses this field.",
       recommendation="Add a level 'stated_no_link' in a future amendment."),
  dict(id='G3', severity='minor',
       case="D6's word list is satisfied LITERALLY but not in spirit: the splitting sentence "
            "is 'We divided our dataset into three folds depending on the number of dead and "
            "censored patients', where the divided unit is 'our dataset' and the patient noun "
            "appears only as the stratification count.",
       records=['38298725'],
       consequence="The literal word-list test is what makes D6 decidable, so the sealed code "
                   "'patient_subject' is KEPT. Applying the word list literally is also what "
                   "moved PMID 40093990 the other way, so the test was not applied selectively.",
       recommendation="A future amendment should say the patient-naming noun must denote the "
                      "unit being divided, not a quantity the division is balanced on."),
  dict(id='G4', severity='minor',
       case="Two split sentences name different units and there is NO factual contradiction "
            "for D5 to resolve: Abstract '385 subjects ... randomly separated into training "
            "set (n=308) and test set (n=77)', Methods 'All NCCT scans were randomly separated "
            "into a training set (n=308) and a test set (n=77)'. One scan per subject, so both "
            "describe the same partition.",
       records=['35401411'],
       consequence="Sealed code 'patient_subject' KEPT; D5 does not fire because the two "
                   "statements agree on every number.",
       recommendation="State whether the Methods noun governs even when the two statements "
                      "are consistent."),
  dict(id='G5', severity='minor',
       case="D10's stated trigger is a categorical decision made by HUMAN READERS. Two "
            "reader-study records sit just outside it: the readers produce an image-quality "
            "Likert score or a lesion marking, and NO classification metric is reported on "
            "the reader decisions.",
       records=['34324463', '41547664'],
       consequence="E-SEG is KEPT on both, on the test actually applied here -- E-SEG's "
                   "qualifier fails only where a categorical class decision is scored with a "
                   "metric from the I4 list. Both records stay excluded either way, so no "
                   "endpoint moves.",
       recommendation="Say in the codebook whether an image-quality rating is a 'categorical "
                      "class decision' for the purposes of E-SEG's qualifier."),
 ],
 _operational_test_used_for_D10=(
   "E-SEG's qualifier ('with NO categorical class decision evaluated') FAILS, and E-NOCLF "
   "applies, where a categorical class decision on an imaging unit IS evaluated with a metric "
   "from the I4 list but is NOT produced by a fitted classifier -- whether the decider is a "
   "human reader (D10's own wording, PMIDs 41274092, 40335658, 37962500) or a threshold on a "
   "continuous quantity the network produced (PMIDs 34003996, 34603980). E-DERIV does not "
   "intervene, because E-DERIV presupposes a fitted classifier whose INPUT is a derived "
   "representation; where no classifier is fitted at all, the defining failure is I2."),
 changelog_entry_for_screen_frame=dict(
   date='2026-07-29', version='1.3.1',
   trigger="The second half of agreement.threshold_and_remedy: 'EVERY already-coded paper is "
           "re-coded under the amended codebook'. The adjudication amended the codebook to "
           "v1.2 and the access-recovery pass added four full texts at v1.3; until this pass "
           "ran, the amendment had been applied only to the 15 overlap papers, which would "
           "have left the 85 singly-coded records coded under v1.0 and non-comparable.",
   event="RE-CODE EXECUTED. NO CODEBOOK RULE WAS CHANGED and no sealed batch file was edited. "
         "All 145 coded records over all 100 sampled papers re-coded under v1.2 + the v1.3 "
         "overlay. Output: paper/screen_recoded.json, paper/screen_recoded.md, generated by "
         "paper/screen/analysis/recode_v12.py.",
   consequence="Primary endpoint UNCHANGED and still exactly zero: no record in the sample "
               "carries a TRUE on any of the four P1 sub-flags, and none moved INTO that state "
               "under the amendment. D1 moved 5 papers out of 'excluded' and into the "
               "eligible-unreachable set, so the eligible denominator rises 54 -> 59, "
               "unreachable 16 -> 21, S6 29.6% -> 35.6% [24.6%, 48.3%] and the P1 bound "
               "[0.0%, 29.6%] -> [0.0%, 35.6%]. The 15% threshold is still breached, so "
               "rule_4 still binds and the bounding interval remains the headline. D2's "
               "supplement clause moves 10 complete-case papers from an evidenced FALSE to "
               "'not_assessable', so the evidenced complete-case denominator is 28, not 38; "
               "both are reported. S4 12/38, S5 1/38.",
   honesty_note="Both material changes move numbers AGAINST this paper: D1 widens the bound "
                "the paper wants narrow, and D2's supplement clause shrinks the evidenced "
                "denominator behind the paper's central negative. They are adopted because "
                "the rules say so. Five residual gaps in v1.2 found during the re-code are "
                "logged in screen_recoded.json -> residual_gaps_found_during_the_recode and "
                "are NOT patched here."),
 status_and_P1_transitions=transitions,
 endpoints_after_recoding=endpoints,
 papers=papers,
 records=records,
 changes=all_changes,
)
json.dump(out, open(f'{PAPER}/screen_recoded.json', 'w'), indent=1)


# ------------------------------------------------------------------------------ markdown
def cell(v):
    if v is None:
        return '`null`'
    if isinstance(v, dict) and set(v) == set(SUBFLAGS):
        p1 = {json.dumps(v[k]) for k in P1_KEYS}
        p1s = p1.pop() if len(p1) == 1 else '/'.join(sorted(p1))
        return (f"P1 four = `{p1s.strip(chr(34))}`; clinical = "
                f"`{json.dumps(v['clinical_or_demographic_only']).strip(chr(34))}`; other = "
                f"`{json.dumps(v['other_non_imaging']).strip(chr(34))}`")
    if isinstance(v, dict):
        return '`' + ', '.join(f'{k}={json.dumps(x)}' for k, x in v.items()) + '`'
    s = str(v).replace('|', '\\|').replace('\n', ' ')
    if len(s) > 200:
        s = s[:200] + ' … [full text in screen_recoded.json]'
    return '`' + s + '`'


RULE_IDS = ['D14', 'D13', 'D12', 'D11', 'D10', 'D9', 'D8', 'D7', 'D6', 'D5', 'D4', 'D3',
            'D2', 'D1']


def rule_label(rule):
    if 'adjudication' not in rule:
        return rule
    ids, seen = [], rule
    for t in RULE_IDS:
        if t in seen:
            ids.append(t)
            seen = seen.replace(t, '')
    pre = [x for x in rule.split(' + ') if 'adjudication' not in x]
    lab = 'adjudication: ' + ', '.join(sorted(ids, key=lambda z: int(z[1:]))) if ids else 'adjudication'
    return (' + '.join(pre) + ' + ' + lab) if pre else lab


def pct(e):
    return (f"{e['k']}/{e['n']} = {e['pct']:.1f}% [{e['ci'][0]:.1f}%, {e['ci'][1]:.1f}%]"
            if e['pct'] is not None else f"{e['k']}/{e['n']}")


REV = json.load(open(f'{HERE}/recovery_out.json'))
L = []
w = L.append
w('# Re-coding of every screened paper under codebook v1.2')
w('')
w('**Status: the second half of the pre-registered agreement remedy, executed.**  ')
w('`screen_frame.json` &rarr; `agreement.threshold_and_remedy` requires that when the '
  'agreement floor fails, *"a documented adjudication round is held, the codebook is amended '
  'in the changelog, and EVERY already-coded paper is re-coded under the amended codebook."* '
  'The floor failed, the adjudication round ran (`paper/screen_adjudication.md`, rules D1-D14), '
  'and this document is the re-coding. A codebook amendment applied only to new papers would '
  'make old and new records incomparable.')
w('')
w(f"| | |\n|---|---|\n| codebook | `paper/screen_frame.json` v1.2, plus the v1.3 "
  f"access-recovery overlay |\n| inputs | the four sealed batch files, "
  f"`screen_adjudication.md`, `screen/access_recovery.json` |\n| sealed files modified | "
  f"**no** |\n| generated by | `paper/screen/analysis/recode_v12.py` |\n| records re-coded | "
  f"{len(records)} records over {len(papers)} papers |")
w('')
w('---')
w('')
w('## 1. The headline, stated first')
w('')
w('> **No paper moved INTO "reports a zero-image baseline." Not one.**')
w('>')
w('> Across all 145 re-coded records and all 100 papers, not a single one of the four primary '
  'sub-flags — `constant_or_prevalence`, `positional`, `acquisition_metadata`, '
  '`permuted_or_shuffled_label` — is coded TRUE. The P1 numerator was 0 before the amendment '
  'and is 0 after it. The amendment could have moved a paper in: D2(a) was written precisely '
  'to *admit* a positive on abstract-only evidence, and it was applied to every record, '
  'including the 21 whose full text was never obtained. It found none.')
w('')
w('The one flag that does fire anywhere is `clinical_or_demographic_only`, which the codebook '
  '**deliberately excludes from the primary endpoint** (a clinical-variables nomogram does not '
  'test the pixel shortcut). It is TRUE on four papers: '
  + ', '.join(f'`{x}`' for x in endpoints['S1_TRUE_records']) +
  '. Two further papers carry the flag on evidence that is preserved but not counted, because '
  'the record is excluded (`39513126`, `39200968`).')
w('')
w('## 2. What changed, in one table')
w('')
w(f"| | count |\n|---|---|\n"
  f"| records re-coded | {len(records)} |\n"
  f"| records with at least one changed code | **{recs_changed}** |\n"
  f"| papers with at least one changed code | **{papers_changed}** of {len(papers)} |\n"
  f"| field-level changes, total | {len(all_changes)} |\n"
  f"| &nbsp;&nbsp;of which substantive | **{sum(1 for c in all_changes if c['kind'] == 'substantive')}** |\n"
  f"| &nbsp;&nbsp;of which D3 `not_applicable` fills on non-included records | "
  f"{sum(1 for c in all_changes if c['kind'] != 'substantive')} |")
w('')
w('Changes by amendment:')
w('')
w('| rule | field changes | what it does |')
w('|---|---|---|')
RULE_GLOSS = {
 'D3': "descriptive fields on a record that is not `included` become `not_applicable`",
 'D13': "`none` (explicit refusal) split from `not_stated` (silence)",
 'v1.3 access recovery': "four full texts recovered and fully re-coded; **no rule changed**",
 'D2': "`trivial_baseline` becomes three-valued; a FALSE now requires the 14-term search",
 'adjudication': "the 15 overlap papers take the adjudicated consensus code",
 'D1': "unreachable dominates included",
 'D4': "the access ladder is not climbed for a stage-1 exclusion",
 'D10': "E-SEG's *no categorical class decision* qualifier binds &rarr; E-NOCLF",
 'D6': "`patient_subject` needs a patient-naming noun in the splitting sentence",
 'D14': "`headline_unit='unclear'` only where two or more units are reported",
 'D8(a)': "new `split_unit` level `lesion_or_roi`",
}
for r_, n_ in chg_by_rule.most_common():
    w(f"| **{r_}** | {n_} | {RULE_GLOSS.get(r_, '')} |")
w('')
w('## 3. Every substantive code change')
w('')
w('Every row is one field on one record: the sealed code, the final code, and the amendment '
  f"that forced it. Intermediate steps are collapsed, so a field the pipeline moved and moved "
  f"back is not counted twice. The "
  f"{sum(1 for c in all_changes if c['kind'] != 'substantive'):,} mechanical D3 "
  '`not_applicable` fills are summarised in §4 and listed in full in `screen_recoded.json` '
  '&rarr; `changes`.')
w('')
w('| pos | PMID | screener | field | old | new | rule |')
w('|---|---|---|---|---|---|---|')
pos_of = {p['record_id']: p['permutation_position'] for p in papers}
for c in sorted([c for c in all_changes if c['kind'] == 'substantive'],
                key=lambda x: (pos_of[x['record_id']], x['screener_id'], x['field'])):
    w(f"| {pos_of[c['record_id']]} | `{c['record_id']}` | {c['screener_id']} | "
      f"`{c['field']}` | {cell(c['old'])} | {cell(c['new'])} | {rule_label(c['rule'])} |")
w('')
w('## 4. The D3 fills, by paper')
w('')
w('D3 adds `not_applicable` to every descriptive field and makes it available **only** where '
  '`final_inclusion` is not `included`. On such a record those fields describe something that '
  'either does not exist or was never observed. Each fill preserves the record\'s original '
  'value and quote verbatim under `preserved_evidence`, so nothing is discarded.')
w('')
w('| pos | PMID | status | fields filled |')
w('|---|---|---|---|')
fills = collections.Counter()
for c in all_changes:
    if c['kind'] != 'substantive':
        fills[(c['record_id'], c['screener_id'])] += 1
byp = collections.defaultdict(int)
for (pm, s), n_ in fills.items():
    byp[pm] += n_
st_of = {p['record_id']: p['status_for_analysis'] for p in papers}
for pm in sorted(byp, key=lambda x: pos_of[x]):
    w(f"| {pos_of[pm]} | `{pm}` | {st_of[pm]} | {byp[pm]} |")
w('')
movers = [t for t in transitions if t['status_sealed'] != t['status_amended']]
w('## 5. Papers whose analysis status moved')
w('')
w(f'These {len(movers)} are the changes that move an endpoint. Everything else in §3 changes '
  'a code without changing which denominator the paper sits in.')
w('')
w('| pos | PMID | sealed &rarr; amended | why |')
w('|---|---|---|---|')
WHY = {
 '42162744': 'D1 — E-DERIV on an abstract the screener had already sent to stage 2; its own '
             'note opens "JUDGEMENT CALL on which exclusion code"',
 '33937792': 'D1 — E-SEG on an abstract; the screener flagged that a full text "might reveal a '
             'per-nodule benign/malignant classification arm"',
 '41874622': 'D1 — E-SEG; the screener recorded LOW CONFIDENCE and wrote that the row "must '
             'flip to included" if the full text carries a classification arm',
 '35787928': 'D1 — E-DERIV; LOW CONFIDENCE, and the derived-feature reading is explicitly '
             'conditional in the screener\'s own note',
 '35641181': 'D1 — E-DERIV; LOW CONFIDENCE, and the screener wrote "if so it must flip to '
             'included"',
 '38591974': 'v1.3 access recovery — full text obtained, eligibility confirmed',
 '36200353': 'v1.3 access recovery — preprint of the same work obtained',
 '39846055': 'v1.3 access recovery — accepted manuscript obtained from the author\'s repository',
 '36170844': 'v1.3 access recovery — full text obtained and the paper proved INELIGIBLE (E-NOCLF)',
}
for t in transitions:
    if t['status_sealed'] == t['status_amended']:
        continue
    w(f"| {t['permutation_position']} | `{t['record_id']}` | {t['status_sealed']} &rarr; "
      f"**{t['status_amended']}** | {WHY.get(t['record_id'], '')} |")
w('')
w('The five D1 movers all share one shape: the screener coded `stage1_decision='
  "='go_to_fulltext'`, i.e. declared on the record that the abstract alone did not settle "
  'eligibility, then could not obtain the full text, then excluded the paper anyway. D1 '
  'refuses that, and the adjudication had already refused it once on PMID `42489954`. The '
  'exclusion code and its quote are preserved on each record and are re-usable the moment a '
  'full text is obtained.')
w('')
w('### 5.1 Papers whose primary flag changed category without changing status')
w('')
w('| pos | PMID | status | P1: sealed &rarr; amended | search |')
w('|---|---|---|---|---|')
for t in transitions:
    if t['status_sealed'] != t['status_amended']:
        continue
    w(f"| {t['permutation_position']} | `{t['record_id']}` | {t['status_amended']} | "
      f"`{t['P1_sealed']}` &rarr; `{t['P1_amended']}` | {t['fulltext_search']} |")
w('')
w('Two mechanisms, both D2/D3. On an **excluded** record the sub-flags become '
  '`not_applicable`: the record enters no P1 denominator, so an unevidenced `false` sitting '
  'there was pure noise. On an **included or unreachable** record they become '
  '`not_assessable`: a `false` now requires the 14-term search, and either the full text or '
  'the supplement was missing.')
w('')
w('## 6. Effect on the endpoints')
w('')
b, af = REV['before'], REV['after']
e = endpoints
w('| | as sealed (v1.0) | after access recovery (v1.3) | **after this re-code (v1.2)** |')
w('|---|---|---|---|')
w(f"| eligible-looking | {b['n_eligible']} | {af['n_eligible']} | **{e['n_eligible']}** |")
w(f"| included + reachable | {b['n_included_reachable']} | {af['n_included_reachable']} | "
  f"**{e['n_included_reachable']}** |")
w(f"| eligible but unreachable | {b['n_eligible_unreachable']} | {af['n_eligible_unreachable']} | "
  f"**{e['n_eligible_unreachable']}** |")
w(f"| excluded | {b['n_excluded']} | {af['n_excluded']} | **{e['n_excluded']}** |")
w(f"| S6 unreachable | {pct(b['S6_unreachable'])} | {pct(af['S6_unreachable'])} | "
  f"**{pct(e['S6_unreachable'])}** |")
w(f"| P1 complete case | {pct(b['P1_complete_case'])} | {pct(af['P1_complete_case'])} | "
  f"**{pct(e['P1_complete_case_all_reachable'])}** |")
w(f"| P1 complete case, restricted to EVIDENCED negatives | n/a | n/a | "
  f"**{pct(e['P1_complete_case_evidence_restricted'])}** |")
w(f"| P1 bounding interval | [{b['P1_lower']['pct']:.1f}%, {b['P1_upper']['pct']:.1f}%] | "
  f"[{af['P1_lower']['pct']:.1f}%, {af['P1_upper']['pct']:.1f}%] | "
  f"**[{e['P1_lower']['pct']:.1f}%, {e['P1_upper']['pct']:.1f}%]** |")
w(f"| >15% threshold breached | {b['threshold_breached']} | {af['threshold_breached']} | "
  f"**{e['threshold_breached']}** |")
w('')
w(f"**The bounding interval remains the headline number.** Unreachability is "
  f"{e['S6_unreachable']['pct']:.1f}%, still far above the pre-registered 15% threshold at "
  f"which `missing_and_unreachable.rule_4` puts the bound in place of the point estimate. The "
  f"re-code moved that rate *up*, from {af['S6_unreachable']['pct']:.1f}% to "
  f"{e['S6_unreachable']['pct']:.1f}%, because D1 refuses five stage-2 exclusions that rested "
  f"on abstracts the screeners had themselves declared insufficient.")
w('')
w('Secondary endpoints on the complete-case set:')
w('')
w(f"- **S1** (any non-imaging baseline): {pct(e['S1_complete_case'])}")
w(f"- **S4** (explicit subject-level split): {pct(e['S4_subject_split'])}")
w(f"- **S5** (positional distribution reported): {pct(e['S5_positional'])}")
w('')
w('## 7. Direction audit')
w('')
w('Every amendment that moved a number is listed here with the direction it moved it. Three '
  'of the four move against us.')
w('')
w('| rule | what moved | direction |')
w('|---|---|---|')
w('| **D1** | 5 papers out of `excluded` and into the eligible-unreachable set | **against us** '
  '— raises unreachability and widens the bound |')
w('| **D2** (supplement clause) | 10 complete-case papers from an evidenced `false` to '
  '`not_assessable` | **against us** — shrinks the evidenced denominator from 38 to 28 |')
w('| **D6** | PMID `40093990` out of S4\'s numerator | **against the literature** — one fewer '
  'paper with a subject-level split |')
w('| **D9** (adjudication) | PMID `42130124` into S5\'s numerator | **in favour of the '
  'literature**, on the very endpoint this paper accuses it of ignoring |')
w('')
w('No rule was applied where it helped and withheld where it hurt. The clearest test is D6: '
  'the same literal word-list test that removed `40093990` from S4 is the test that *kept* '
  '`38298725` and `35401411` in it (§8, G3 and G4).')
w('')
w('## 8. Gaps the re-code found in v1.2')
w('')
w('These are logged, not silently patched. No rule was invented here to dispose of them.')
w('')
for g in out['residual_gaps_found_during_the_recode']:
    w(f"**{g['id']} ({g['severity']}).** {g['case']}")
    w('')
    w(f"- records: {', '.join('`' + x + '`' for x in g['records'])}")
    w(f"- consequence: {g['consequence']}")
    w(f"- recommendation: {g['recommendation']}")
    w('')
w('## 9. What this re-code is NOT')
w('')
w('It is **not** an independent re-rating and it produces **no reliability statistic**. '
  '`screen_frame.json` &rarr; `agreement.what_a_post_remedy_number_may_and_may_not_claim` is '
  'explicit: agreement measured on one consensus code per paper is 1.000 by construction and '
  'is evidence of nothing. A genuine post-amendment reliability estimate requires a fresh '
  'independent re-coding by four screeners under v1.2, and remains an outstanding action.')
w('')
w('What this re-code *does* establish is that the amended codebook, applied to every record '
  'the screen produced, leaves the primary result exactly where the sealed files left it: '
  '**zero papers reporting a zero-image baseline, and a bounding interval that only recovering '
  'more full texts can narrow.**')
w('')
w('## 10. Changelog entry to be added to `screen_frame.json`')
w('')
w('This pass changed no rule, so it is logged as an execution record, not an amendment. The '
  'entry is carried in `screen_recoded.json` &rarr; `changelog_entry_for_screen_frame` rather '
  'than written into the codebook here, because other passes were editing that file '
  'concurrently.')
w('')
w('```json')
w(json.dumps(out['changelog_entry_for_screen_frame'], indent=1))
w('```')
w('')
open(f'{PAPER}/screen_recoded.md', 'w').write('\n'.join(L) + '\n')
print(f"wrote {PAPER}/screen_recoded.md")

print(f"records {len(records)}  changed {recs_changed}   field-changes {len(all_changes)}")
print('by rule :', dict(chg_by_rule.most_common()))
print('by field:', dict(chg_by_field.most_common()))
print()
for k in ('n_included_reachable', 'n_eligible_unreachable', 'n_excluded', 'n_eligible',
          'n_complete_case_with_evidenced_P1_negative', 'n_complete_case_P1_not_assessable',
          'threshold_breached', 'P1_TRUE_records', 'S1_TRUE_records'):
    print(f'  {k:48s} {endpoints[k]}')
for k in ('S6_unreachable', 'P1_complete_case_all_reachable',
          'P1_complete_case_evidence_restricted', 'P1_lower', 'P1_upper',
          'P1_upper_incl_not_assessable', 'S1_complete_case', 'S4_subject_split',
          'S5_positional'):
    e = endpoints[k]
    print(f"  {k:48s} {e['k']}/{e['n']}" +
          (f" = {e['pct']:.1f}%  [{e['ci'][0]:.1f}%, {e['ci'][1]:.1f}%]" if e['pct'] is not None else ''))
print(f"\nwrote {PAPER}/screen_recoded.json")

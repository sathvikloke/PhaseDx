"""Lane C: outcome-blind clinical series selection. Clinical fields true/false/unknown; unknown != false and blocks selection; tie-break only after clinical status is resolved:
largest superior-inferior coverage, then |median gap - 5 mm| smallest, then lexical SeriesInstanceUID. Technical unusability is separate from clinical eligibility and from model failure."""
import csv
TRI = ('true', 'false', 'unknown')
CLIN = ('noncontrast', 'axial', 'soft_reconstruction', 'brain_coverage')
COLS = ['source_exam_id', 'study_uid', 'series_uid', *CLIN, 'evidence_locator', 'selector', 'review_status', 'reason']

class SelectionError(RuntimeError):
    pass

def load_selection(path):
    out = {}
    for r in csv.DictReader(open(path)):
        for c in COLS:
            if c not in r:
                raise SelectionError('series_selection.csv missing column ' + c)
        for c in CLIN:
            if r[c] not in TRI:
                raise SelectionError('field %s must be true/false/unknown, got %r' % (c, r[c]))
        k = (r['source_exam_id'], r['series_uid'])
        if k in out:
            raise SelectionError('duplicate selection row %r' % (k,))
        out[k] = r
    return out

def clinical_state(row):
    if row is None:
        return ('unresolved', ['no_selection_row'])
    v = [row[c] for c in CLIN]
    if 'false' in v:
        return ('negative', [c for c in CLIN if row[c] == 'false'])
    if 'unknown' in v:
        return ('unresolved', [c for c in CLIN if row[c] == 'unknown'])
    return ('qualifying', [])

def select(exam, series, sel, study_uid_ok=True):
    """series: list of dicts with series_key, technical_usable(bool), technical_reason, span_mm, median_gap_mm. Returns (series_key, status, reason)."""
    if not study_uid_ok:
        return ('', 'study_uid_unresolved', 'expected/resolved StudyUID conflict: no series chosen')
    usable = [s for s in series if s['technical_usable']]
    if not usable:
        return ('', 'no_technically_usable_series', ';'.join(sorted({s['technical_reason'] for s in series if s['technical_reason']})) or 'no series')
    q, unres, neg = ([], [], [])
    for s in usable:
        st, why = clinical_state(sel.get((exam, s['series_key'])))
        (q if st == 'qualifying' else unres if st == 'unresolved' else neg).append((s, why))
    if unres:
        return ('', 'clinical_unresolved', 'unresolved clinical fields: ' + '|'.join(('%s:%s' % (s['series_key'], ','.join(w)) for s, w in unres)))
    if not q:
        return ('', 'no_clinically_qualifying_series', '|'.join(('%s:%s' % (s['series_key'], ','.join(w)) for s, w in neg)))
    best = sorted(q, key=lambda t: (-float(t[0]['span_mm'] or 0), abs(float(t[0]['median_gap_mm'] if t[0]['median_gap_mm'] not in (None, '') else 1000000000.0) - 5.0), t[0]['series_key']))[0][0]
    return (best['series_key'], 'selected', 'qualifying=%d; tie-break coverage>gap-to-5mm>lexical uid' % len(q))

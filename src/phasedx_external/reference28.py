"""Lane E: reference adapter. Rule: reader-level any-ICH finding, majority of the three readers (chronic hemorrhage positive); two agreeing valid reads decide, otherwise MISSING (analysis rule of this protocol, not a claim about the source).
Schema must be authenticated (column names + key rule from primary schema documentation); three DISTINCT reader columns."""
import re
RULE_ID = 'majority_of_reader_ich_field_two_agreeing_else_missing'

class ReferenceBlocked(RuntimeError):
    pass

def validate_schema(s):
    for k in ('provenance', 'authenticated', 'exam_key_col', 'key_transform', 'reader_cols', 'n_readers', 'rule_id', 'missing_policy'):
        if k not in s:
            raise ReferenceBlocked('schema missing field ' + k)
    if s['authenticated'] is not True or not isinstance(s['provenance'], dict) or (not s['provenance'].get('locator')):
        raise ReferenceBlocked('schema not authenticated with a source locator')
    cols = s['reader_cols']
    if len(cols) != 3 or len(set(cols)) != 3 or s['n_readers'] != 3:
        raise ReferenceBlocked('need exactly three distinct reader columns')
    if s['exam_key_col'] in cols:
        raise ReferenceBlocked('exam key column cannot be a reader column')
    if s['rule_id'] != RULE_ID or s['missing_policy'] != 'two_agreeing_valid_reads_else_missing':
        raise ReferenceBlocked('unsupported rule/missing policy')
    return s

def key_transform(rule, v):
    if rule == 'identity':
        return v
    if rule.startswith('regex_extract:'):
        m = re.search(rule.split(':', 1)[1], v)
        if not m:
            return None
        return m.group(1) if m.groups() else m.group(0)
    raise ReferenceBlocked('unknown key transform ' + rule)

def majority(vals):
    p = sum((v == 1 for v in vals))
    n = sum((v == 0 for v in vals))
    return 1 if p >= 2 else 0 if n >= 2 else None

def adapt(rows, schema):
    validate_schema(schema)
    out = []
    for r in rows:
        for c in schema['reader_cols'] + [schema['exam_key_col']]:
            if c not in r:
                raise ReferenceBlocked('label file lacks column ' + c)
        vals = [int(r[c]) if r[c] in ('0', '1') else None for c in schema['reader_cols']]
        k = key_transform(schema['key_transform'], r[schema['exam_key_col']])
        out.append({'source_key': r[schema['exam_key_col']], 'exam_key': k, 'label': '' if majority(vals) is None else str(majority(vals)), 'n_reads_invalid_or_missing': sum((v is None for v in vals))})
    return out

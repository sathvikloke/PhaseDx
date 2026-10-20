"""Lane A/B: design lock (outcome-blind) -> execution lock (binds manifest COMPONENT digests) -> run start/events/end -> receipt -> admission BEFORE labels.
Manifest identity is always RECOMPUTED from the consumed files; a claimed MANIFEST_IDENTITY.json digest is never trusted. Local records only; no external notarization."""
import os, sys, json, hashlib, datetime, uuid, csv

class AdmissionError(RuntimeError):
    pass
ROLES_REQUIRED = ('primary_image', 'geometry')
ROLES_ALL = ('primary_image', 'secondary_image', 'geometry')
COMPONENTS = ('roster.csv', 'policy.json', 'candidates.csv', 'inventory.csv', 'series.csv', 'series_selection.csv', 'selected_series.csv', 'slices.csv', 'geometry_headers.csv')
MODES = ('clinical_external', 'synthetic_point_only')
CLIN_RECIPE = {'n_draws': 2000, 'seed': 2026101327, 'percentile_method': 'linear', 'percentiles': [2.5, 97.5]}

def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def parse_utc(s):
    try:
        d = datetime.datetime.fromisoformat(s)
    except Exception:
        raise AdmissionError('unparseable timestamp %r' % (s,))
    if d.tzinfo is None:
        raise AdmissionError('timestamp without timezone %r' % (s,))
    return d

def write_once(path, obj):
    try:
        with open(path, 'x') as f:
            json.dump(obj, f, indent=1, sort_keys=True)
    except FileExistsError:
        raise AdmissionError('write-once file exists: ' + path)
    return sha_file(path)

def verify_manifest_dir(mdir):
    """Recompute component digests from the actual files; compare with MANIFEST_IDENTITY.json components; return recomputed identity. Missing files are explicit."""
    ip = os.path.join(mdir, 'MANIFEST_IDENTITY.json')
    if not os.path.exists(ip):
        raise AdmissionError('MANIFEST_IDENTITY.json missing')
    claimed = json.load(open(ip))
    comp = {}
    missing = []
    for c in COMPONENTS:
        p = os.path.join(mdir, c)
        if os.path.exists(p):
            comp[c] = sha_file(p)
        else:
            missing.append(c)
    if missing:
        raise AdmissionError('manifest component files missing: ' + ','.join(missing))
    bad = sorted((c for c in COMPONENTS if claimed.get('components', {}).get(c) != comp[c]))
    if bad:
        raise AdmissionError('manifest files differ from recorded component digests: ' + ','.join(bad))
    rows = list(csv.DictReader(open(os.path.join(mdir, 'inventory.csv'))))
    inv = hashlib.sha256('\n'.join(('%s\t%s' % (r['relpath'], r['sha256']) for r in rows)).encode()).hexdigest()
    if inv != claimed.get('inventory_digest'):
        raise AdmissionError('inventory digest differs from inventory.csv content')
    return {'components': comp, 'manifest_sha256': sha_obj({'components': comp, 'inventory_digest': inv}), 'inventory_digest': inv, 'roster_sha256': comp['roster.csv']}

def make_design_lock(path, d):
    for k in ('protocol_id', 'mode', 'models', 'preprocessing', 'aggregation', 'representation', 'label_rule_id', 'grouping_policy', 'estimand'):
        if k not in d:
            raise AdmissionError('design lock missing ' + k)
    if d['mode'] not in MODES:
        raise AdmissionError('unknown mode')
    for r in ROLES_ALL:
        m = d['models'].get(r)
        if not m or not m.get('checkpoint_sha256') or set(m['checkpoint_sha256']) == {'0'}:
            raise AdmissionError('real checkpoint hash required for ' + r)
    if d['mode'] == 'clinical_external':
        cc = d.get('clinical_contract') or {}
        need = {'primary_model_id': 'I1_seed42', 'geometry_model_id': 'G2_tree_transported_rank_reconstructed_frozen', 'bootstrap': CLIN_RECIPE}
        for k, v in need.items():
            if cc.get(k) != v:
                raise AdmissionError('clinical contract field %s must equal %r' % (k, v))
        for k in ('target', 'precision_path', 'aggregation', 'reference_schema_sha256', 'source_key_rule', 'grouping_rule'):
            if not cc.get(k):
                raise AdmissionError('clinical contract missing ' + k)
        if cc['grouping_rule'] not in ('verified_linkage_required', 'fallback_if_linkage_unavailable'):
            raise AdmissionError('bad grouping rule')
        if d['models']['primary_image']['model_id'] != 'I1_seed42' or d['models']['geometry']['model_id'] != cc['geometry_model_id']:
            raise AdmissionError('model ids differ from contract')
    d = dict(d, kind='design_lock', created_utc=now_utc(), statement='outcome-blind local record; not external pre-registration')
    return write_once(path, d)

def make_execution_lock(path, design_path, mdir):
    ident = verify_manifest_dir(mdir)
    e = {'kind': 'execution_lock', 'created_utc': now_utc(), 'design_lock_sha256': sha_file(design_path), 'manifest_sha256': ident['manifest_sha256'], 'inventory_digest': ident['inventory_digest'], 'roster_sha256': ident['roster_sha256'], 'components': ident['components']}
    return write_once(path, e)

def intended_identity(role, design, execl):
    m = design['models'][role]
    g = role == 'geometry'
    identity = {'model_id': m['model_id'], 'checkpoint_sha256': m['checkpoint_sha256'], 'recipe_sha256': 'n/a' if g else design['preprocessing']['recipe_sha256'], 'precision_path': 'n/a' if g else design['preprocessing']['precision_path'], 'aggregation': 'n/a' if g else design['aggregation'], 'series_policy_sha256': execl['components']['policy.json'], 'manifest_sha256': execl['manifest_sha256'], 'inventory_digest': execl['inventory_digest'], 'representation_id': design['representation']['representation_id'] if g else 'n/a'}
    if not g and 'decoder_policy' in design['preprocessing']:
        for key in ('decoder_policy', 'decoder_sha256', 'adapter_sha256', 'decoder_packages'):
            if not design['preprocessing'].get(key):
                raise AdmissionError('decoder identity missing ' + key)
            identity[key] = design['preprocessing'][key]
    return identity

def start_run(run_dir, role, design_path, exec_path, mdir, actual):
    """Before model load: recompute manifest identity from consumed files and compare with the execution lock."""
    design, execl = (json.load(open(design_path)), json.load(open(exec_path)))
    if execl.get('design_lock_sha256') != sha_file(design_path):
        raise AdmissionError('execution lock does not bind this design lock')
    ident = verify_manifest_dir(mdir)
    if ident['manifest_sha256'] != execl['manifest_sha256']:
        raise AdmissionError('consumed manifest files differ from the execution lock (recomputed identity)')
    actual = dict(actual, manifest_sha256=ident['manifest_sha256'], inventory_digest=ident['inventory_digest'], series_policy_sha256=ident['components']['policy.json'])
    want = intended_identity(role, design, execl)
    bad = sorted((k for k in want if want[k] != actual.get(k)))
    if bad:
        raise AdmissionError('actual run identity differs from execution lock before model load: ' + ','.join(bad))
    os.makedirs(run_dir, exist_ok=True)
    sp, ev = (run_dir + '/RUN_START.json', run_dir + '/RUN_EVENTS.jsonl')
    rec = {'kind': 'run_start', 'run_id': str(uuid.uuid4()), 'role': role, 'execution_lock_sha256': sha_file(exec_path), 'design_lock_sha256': sha_file(design_path), 'actual': actual, 'actual_sha256': sha_obj(actual), 'start_utc': now_utc()}
    try:
        write_once(sp, rec)
        open(ev, 'a').write(json.dumps({'event': 'start', 'utc': rec['start_utc']}) + '\n')
        return (rec, False)
    except AdmissionError:
        old = json.load(open(sp))
        if old['actual'] != actual or old['execution_lock_sha256'] != rec['execution_lock_sha256']:
            raise AdmissionError('resume with a different identity than the original run start')
        open(ev, 'a').write(json.dumps({'event': 'resume', 'utc': now_utc()}) + '\n')
        return (old, True)

def end_run(run_dir, pred_csv, n_rows):
    ep = run_dir + '/RUN_END.json'
    if os.path.exists(ep):
        return json.load(open(ep))
    e = {'kind': 'run_end', 'run_start_sha256': sha_file(run_dir + '/RUN_START.json'), 'predictions_sha256': sha_file(pred_csv), 'n_rows': n_rows, 'end_utc': now_utc()}
    write_once(ep, e)
    open(run_dir + '/RUN_EVENTS.jsonl', 'a').write(json.dumps({'event': 'end', 'utc': e['end_utc']}) + '\n')
    return e

def make_receipt(path, exec_path, runs, mdir):
    ident = verify_manifest_dir(mdir)
    body = {'kind': 'prediction_receipt', 'execution_lock_sha256': sha_file(exec_path), 'manifest_sha256': ident['manifest_sha256'], 'candidates_sha256': ident['components']['candidates.csv'], 'runs': {}}
    ends = []
    for r, (d, p) in runs.items():
        e = json.load(open(d + '/RUN_END.json'))
        ends.append(parse_utc(e['end_utc']))
        body['runs'][r] = {'run_start_sha256': sha_file(d + '/RUN_START.json'), 'run_end_sha256': sha_file(d + '/RUN_END.json'), 'predictions_sha256': sha_file(p)}
    body['receipt_utc'] = now_utc()
    if parse_utc(body['receipt_utc']) < max(ends):
        raise AdmissionError('receipt earlier than a run end')
    body['body_sha256'] = sha_obj({k: v for k, v in body.items() if k != 'body_sha256'})
    write_once(path, body)
    return body

def _check_run(role, d, p, design, execl, exec_path, t_exec):
    if not os.path.exists(d + '/RUN_START.json'):
        raise AdmissionError('missing run start record for ' + role)
    if not os.path.exists(d + '/RUN_END.json'):
        raise AdmissionError('run not ended for ' + role)
    st, en = (json.load(open(d + '/RUN_START.json')), json.load(open(d + '/RUN_END.json')))
    if st['execution_lock_sha256'] != sha_file(exec_path):
        raise AdmissionError('run %s loaded a different execution lock' % role)
    if st['role'] != role:
        raise AdmissionError('run role mismatch')
    want = intended_identity(role, design, execl)
    bad = sorted((k for k in want if want[k] != st['actual'].get(k)))
    if bad:
        raise AdmissionError('run %s actual identity differs from lock: %s' % (role, ','.join(bad)))
    if st['actual_sha256'] != sha_obj(st['actual']):
        raise AdmissionError('run start record altered')
    ts, te = (parse_utc(st['start_utc']), parse_utc(en['end_utc']))
    if ts < t_exec:
        raise AdmissionError('run %s started before the execution lock existed' % role)
    if te < ts:
        raise AdmissionError('run end before start')
    if en['run_start_sha256'] != sha_file(d + '/RUN_START.json'):
        raise AdmissionError('run end does not bind this start record')
    if en['predictions_sha256'] != sha_file(p):
        raise AdmissionError('predictions changed since run end (%s)' % role)
    ev = [json.loads(l) for l in open(d + '/RUN_EVENTS.jsonl') if l.strip()]
    if not ev or ev[0]['event'] != 'start' or sum((e['event'] == 'start' for e in ev)) != 1 or (ev[0]['utc'] != st['start_utc']):
        raise AdmissionError('run event log inconsistent')
    for r in csv.DictReader(open(p)):
        if r.get('run_id') != st['run_id'] or r.get('execution_lock_sha256') != st['execution_lock_sha256'] or r.get('identity_sha256') != st['actual_sha256']:
            raise AdmissionError('prediction row identity does not match run (%s)' % role)
    return te

def admit(design_path, exec_path, receipt_path, runs, mdir, candidates_csv, label_opener):
    """All checks BEFORE label_opener(). Candidate table and manifest are recomputed/compared with the execution lock; a new receipt cannot bless a changed cohort."""
    design, execl, rc = (json.load(open(design_path)), json.load(open(exec_path)), json.load(open(receipt_path)))
    if execl['design_lock_sha256'] != sha_file(design_path):
        raise AdmissionError('execution lock not bound to design lock')
    ident = verify_manifest_dir(mdir)
    if ident['manifest_sha256'] != execl['manifest_sha256']:
        raise AdmissionError('manifest differs from the execution lock (cohort/inputs changed after locking)')
    if sha_file(candidates_csv) != execl['components']['candidates.csv']:
        raise AdmissionError('candidate table differs from the one fixed by the execution lock')
    if rc['body_sha256'] != sha_obj({k: v for k, v in rc.items() if k != 'body_sha256'}):
        raise AdmissionError('receipt body modified')
    if rc['execution_lock_sha256'] != sha_file(exec_path) or rc['manifest_sha256'] != execl['manifest_sha256']:
        raise AdmissionError('receipt not bound to this execution lock/manifest')
    t_exec = parse_utc(execl['created_utc'])
    if t_exec < parse_utc(design['created_utc']):
        raise AdmissionError('execution lock older than design lock')
    ends, secondary = ([], {'status': 'unavailable', 'reason': 'no secondary run supplied'})
    for role in ROLES_REQUIRED:
        if role not in runs:
            raise AdmissionError('missing run for required role ' + role)
        d, p = runs[role]
        ends.append(_check_run(role, d, p, design, execl, exec_path, t_exec))
        rr = rc['runs'].get(role) or {}
        if rr.get('run_start_sha256') != sha_file(d + '/RUN_START.json') or rr.get('run_end_sha256') != sha_file(d + '/RUN_END.json') or rr.get('predictions_sha256') != sha_file(p):
            raise AdmissionError('receipt does not bind run files (%s)' % role)
    if 'secondary_image' in runs:
        d, p = runs['secondary_image']
        try:
            te = _check_run('secondary_image', d, p, design, execl, exec_path, t_exec)
            rr = rc['runs'].get('secondary_image') or {}
            if rr.get('predictions_sha256') != sha_file(p) or rr.get('run_end_sha256') != sha_file(d + '/RUN_END.json'):
                raise AdmissionError('receipt does not bind secondary run')
            ends.append(te)
            secondary = {'status': 'admitted', 'reason': ''}
        except (AdmissionError, FileNotFoundError, KeyError) as e:
            secondary = {'status': 'unavailable', 'reason': 'secondary failed admission/incomplete: ' + str(e)}
    if parse_utc(rc['receipt_utc']) < max(ends):
        raise AdmissionError('receipt predates a run end')
    return ({'admitted': True, 'secondary': secondary, 'manifest_sha256': ident['manifest_sha256'], 'limits': 'local records; no trusted external timestamp'}, label_opener())

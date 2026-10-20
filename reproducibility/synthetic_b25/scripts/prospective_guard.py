"""Prospective entry guard. Call require_registration(__file__, config_path) BEFORE any real-data read.
Admits only when RECEIPTS_ALL.jsonl (path from the launcher, inside this batch root) holds a 'register' event, written before the child was spawned, for the current attempt whose
token hash, launcher pid (== our parent), script sha256 and config sha256 match, with no 'end' event yet. Anything else exits 97 without touching data."""
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
def _sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()
def require_registration(script_file, config_path=None, root=None):
    root = os.path.abspath(root or ROOT)
    why = None
    try:
        att, tok, rp = os.environ.get('PHASEDX_ATTEMPT_ID'), os.environ.get('PHASEDX_TOKEN'), os.environ.get('PHASEDX_RECEIPTS')
        if not (att and tok and rp): why = 'no registration environment (direct invocation)'
        elif os.path.commonpath([os.path.abspath(rp), root]) != root or os.path.basename(rp) != 'RECEIPTS_ALL.jsonl': why = 'receipts file outside expected root'
        else:
            recs = [json.loads(l) for l in open(rp) if l.strip()]
            reg = [r for r in recs if r.get('event') == 'register' and r.get('attempt_id') == att]
            ended = [r for r in recs if r.get('event') == 'end' and r.get('attempt_id') == att]
            if not reg: why = 'no register receipt for attempt'
            elif ended: why = 'attempt already ended'
            else:
                r = reg[-1]
                if r.get('token_sha256') != hashlib.sha256(tok.encode()).hexdigest(): why = 'token mismatch'
                elif r.get('launcher_pid') != os.getppid(): why = 'parent is not the registering launcher'
                elif r.get('script_sha256') != _sha(os.path.abspath(script_file)): why = 'script identity mismatch'
                elif (config_path is not None) and r.get('config_sha256') != _sha(os.path.abspath(config_path)): why = 'config identity mismatch'
                elif (config_path is None) and r.get('config_sha256'): why = 'registered with a config but script declared none'
    except Exception as e:
        why = 'guard error: ' + repr(e)[:80]
    if why:
        sys.stderr.write('PROSPECTIVE_GUARD_REJECTED: ' + why + '\n'); sys.exit(97)
    return {'attempt_id': att, 'registered': True}

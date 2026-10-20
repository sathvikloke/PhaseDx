"""acquire31.py — image-only CQ500 operation (HTTPS, no peer sharing). One owner per destination; explicit deadline; append-only byte ledger; content-valid reuse; bounded probes/extraction.
Counted bytes = application payload received from the server (headers/TLS overhead NOT counted). Crash uncertainty: every transfer first appends a 'reserve' (remaining expected bytes + the 1-byte anomaly probe); an unsettled reserve counts in full.
Subcommands: plan | fetch | extract. Exit: 0 ok, 3 refused/partial, 4 stopped (resume preserved), 6 baseline unresolved.
Batch31 changes vs batch30/acquire30.py (sha256 1d50a04b...): (1) post-read/pre-write resource refresh in fetch and extraction loops; (2) reservation and cumulative admission include the 1-byte anomaly probe;
(3) a fully received, identity-bound .part is content-verified and promoted without a Range request beyond EOF; (4) accounting-root storage is a cached tree measurement plus this process's own writes, re-measured every
TREE_REFRESH_S seconds (a full os.walk per 1 MiB chunk was infeasible for ~29 GB / ~10^5 extracted files); (5) `extract` CLI subcommand for safe_extract; (6) network-level errors (timeout/reset/incomplete read/URLError) become Transient, retried per file with bounded
backoff (STOP/deadline honoured) and Range resume of the identity-bound partial, instead of crashing the whole run (real S3 read timeout at 14:34Z, 2026-10-15);
(7) a missing accounting root (volume ejected) refuses before any write at start and pauses cleanly mid-run instead of crashing in the ledger write (observed ~18:19Z); (8) opt-in --workers N parallel fetch in one owner process, ledger/admission serialized by a lock
(single-stream S3 throughput fell to ~1 MB/s at 23:01Z); (9) ZIP_TOTAL_FACTOR 3.0 -> 4.0 (CQ500-CT-43 declares 3.0105x) and
run_extract records an unreadable archive (BadZipFile, e.g. source-truncated CQ500-CT-120) as REFUSED instead of crashing; (10) opt-in --workers N for extract (archives in parallel); (11) extraction state flushed every 64 members/10 s (IOPS); (15) resume tmp cleanup skips AppleDouble sidecars and tolerates already-removed files; (14) case-insensitive path collisions get '~caseN' paths with a recorded map (4 CQ500 archives; verification caught the
overwrites); (13) storage re-measurement adaptive (>=20x walk time, timed from walk end, walk outside the lock, no forced walk once walks
are slow); (12) parallel stop cancels queued jobs and safe_extract checks STOP before
hashing/creating state (a parallel extraction STOP had drained through every queued archive at 00:53Z); a lock left by a dead/ended owner is moved aside
(owner.lock.stale.N, preserved) with a ledger event, while a live owner still refuses."""
import threading, concurrent.futures
import sys, os, csv, json, hashlib, argparse, urllib.request, urllib.parse, urllib.error, shutil, zipfile, stat, re, time, datetime, uuid
FORBIDDEN_NAMES = ('prediction_probabilities.csv',)

class Refuse(RuntimeError):

    def __init__(self, msg, code=3, decision=None):
        super().__init__(msg)
        self.code = code
        self.decision = decision

class Stopped(RuntimeError):
    pass

class Transient(RuntimeError):
    """Network-level failure (timeout, reset, incomplete read, connection error). The identity-bound partial is preserved; run_fetch retries with a Range resume."""
import http.client, ssl
NET_ERRORS = (TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException, urllib.error.URLError)

def _net(fn, *a, **k):
    try:
        return fn(*a, **k)
    except urllib.error.HTTPError:
        raise
    except NET_ERRORS as e:
        raise Transient('%s: %s' % (type(e).__name__, str(e)[:120]))

class NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, *a, **k):
        return None
_ENDED_INVOCATIONS = set()

class OpContext:

    def __init__(self, dest, deadline_utc, stop_files=(), clock=time.time, storage_cap=400 * 10 ** 9, cumulative_cap=200 * 10 ** 9, min_free=50 * 10 ** 9, baseline_bytes=None, invocation=None, accounting_root=None, free_fn=None, tree_fn=None):
        if deadline_utc is None:
            raise Refuse('explicit --deadline-utc is required (no default, never the expired cycle deadline)')
        d = datetime.datetime.fromisoformat(deadline_utc)
        if d.tzinfo is None:
            raise Refuse('deadline needs a timezone')
        self.deadline = d.timestamp()
        self.clock = clock
        self.stop_files = list(stop_files)
        self.dest = dest
        if self.clock() >= self.deadline:
            raise Refuse('deadline already passed')
        self.storage_cap, self.cumulative_cap, self.min_free, self.baseline = (storage_cap, cumulative_cap, min_free, baseline_bytes)
        self.accounting_root = accounting_root or dest
        self.free_fn = free_fn or free_bytes
        self.tree_fn = tree_fn or tree_bytes
        self._tree_at = None
        self._tree_val = 0
        self._written_since = 0
        self._walk_s = 0.0
        self._walking = False
        self._lk = threading.RLock()
        self._stop_evt = threading.Event()
        self.ledger_dir = os.path.join(dest, '_ledger')
        self.invocation = invocation or str(uuid.uuid4())
        self.lock_path = os.path.join(self.ledger_dir, 'owner.lock')
        self.locked = False

    def _lock_is_stale(self):
        try:
            old = json.load(open(self.lock_path))
        except Exception:
            return (False, None)
        pid = old.get('pid')
        if pid == os.getpid():
            return (old.get('invocation') in _ENDED_INVOCATIONS, old)
        try:
            os.kill(int(pid), 0)
            return (False, old)
        except ProcessLookupError:
            return (True, old)
        except Exception:
            return (False, old)

    def acquire(self):
        os.makedirs(self.ledger_dir, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            stale, old = self._lock_is_stale()
            if not stale:
                raise Refuse('destination owned by another invocation (lock exists: %s)' % self.lock_path)
            n = 1
            while os.path.exists('%s.stale.%d' % (self.lock_path, n)):
                n += 1
            os.replace(self.lock_path, '%s.stale.%d' % (self.lock_path, n))
            self.append({'kind': 'stale_lock_recovered', 'old_lock': old, 'preserved_as': '%s.stale.%d' % (self.lock_path, n)})
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({'invocation': self.invocation, 'pid': os.getpid(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}).encode())
        os.close(fd)
        self.locked = True

    def release(self):
        _ENDED_INVOCATIONS.add(self.invocation)
        if self.locked:
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            self.locked = False

    def check(self):
        if self._stop_evt.is_set():
            raise Stopped('peer_worker_stopped')
        if not os.path.isdir(self.accounting_root):
            raise Stopped('accounting_root_missing')
        if self.clock() >= self.deadline:
            raise Stopped('deadline')
        if any((os.path.exists(s) for s in self.stop_files)):
            raise Stopped('stop_file')

    def _ledger(self):
        return os.path.join(self.ledger_dir, 'bytes.jsonl')

    def events(self):
        p = self._ledger()
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

    def append(self, ev):
        with self._lk:
            if not os.path.isdir(self.accounting_root):
                raise Stopped('accounting_root_missing: ledger %s not written; any open reserve stays unsettled and counts in full' % ev.get('kind'))
            os.makedirs(self.ledger_dir, exist_ok=True)
            open(self._ledger(), 'a').write(json.dumps(dict(ev, invocation=self.invocation, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())) + '\n')

    def cumulative_used(self):
        with self._lk:
            if self.baseline is None:
                raise Refuse('cumulative baseline unresolved: no recorded approved baseline; refusing (no guessed zero)', code=6)
            ev = self.events()
            settled = {e['reserve_id'] for e in ev if e['kind'] == 'settle'}
            received = sum((e['bytes'] for e in ev if e['kind'] == 'settle'))
            open_res = sum((e['bytes'] for e in ev if e['kind'] == 'reserve' and e['reserve_id'] not in settled))
            return self.baseline + received + open_res
    TREE_REFRESH_S = 30.0

    def stored(self, force=False):
        """Accounting-root usage = last full tree measurement + bytes this process wrote since (writes during a walk stay counted: conservative).
        Re-measured when older than max(TREE_REFRESH_S, 20 x last walk duration), interval timed from walk END; force refreshes only while walks are cheap (<1 s).
        The walk runs outside the lock so other workers continue on the cached value. External writers between measurements remain a documented uncertainty.
        (B31: one walk of the real root took 281.5 s at ~170k files; the old per-30-s / forced-per-archive walk stalled extraction ~02:50-03:15Z.)"""
        with self._lk:
            now = self.clock()
            interval = max(self.TREE_REFRESH_S, 20.0 * self._walk_s)
            due = self._tree_at is None or now - self._tree_at >= interval or (force and self._walk_s < 1.0)
            if not due or (self._walking and self._tree_at is not None):
                return self._tree_val + self._written_since
            self._walking = True
            w0 = self._written_since
        t0 = time.time()
        try:
            v = self.tree_fn(self.accounting_root)
        finally:
            with self._lk:
                self._walking = False
        with self._lk:
            self._walk_s = time.time() - t0
            self._tree_val = v
            self._written_since -= w0
            self._tree_at = self.clock()
            return self._tree_val + self._written_since

    def note_written(self, n):
        with self._lk:
            self._written_since += n

    def check_resources(self, need_write):
        """Runtime refresh before a bounded read AND again after it returns, immediately before the write: headroom and project-wide storage (accounting root incl. sibling allocations). Pauses safely (Stopped); partial + ledger intact."""
        if self.free_fn(self.dest) < need_write + self.min_free:
            raise Stopped('insufficient_space_runtime')
        if self.stored() + need_write > self.storage_cap:
            raise Stopped('storage_cap_runtime')

    def admit_bytes(self, need, network_extra=0):
        """need = bytes that may be stored; network_extra = bounded additional network bytes that may be received but never stored (the 1-byte anomaly probe)."""
        with self._lk:
            used = self.cumulative_used()
            if used + need + network_extra > self.cumulative_cap:
                raise Refuse('cumulative cap: used %d + need %d (+%d probe) > %d' % (used, need, network_extra, self.cumulative_cap))
            st = self.stored(force=True)
            if st + need > self.storage_cap:
                raise Refuse('storage cap: stored %d + need %d > %d' % (st, need, self.storage_cap))
            if self.free_fn(self.dest) < need + self.min_free:
                raise Refuse('free space %d < need %d + headroom %d' % (self.free_fn(self.dest), need, self.min_free))
    PROBE_BYTES = 1

    def reserve(self, name, nbytes):
        """Reserve the remaining body plus the 1-byte anomaly probe, so an exactly exhausted cumulative budget refuses instead of exceeding the hard cap by one byte. Admission + reserve record are one atomic step across workers."""
        with self._lk:
            self.admit_bytes(nbytes, self.PROBE_BYTES)
            rid = str(uuid.uuid4())
            self.append({'kind': 'reserve', 'reserve_id': rid, 'name': name, 'bytes': nbytes + self.PROBE_BYTES, 'body_bytes': nbytes, 'probe_bytes': self.PROBE_BYTES})
            return rid

    def settle(self, rid, name, received):
        self.append({'kind': 'settle', 'reserve_id': rid, 'name': name, 'bytes': received})

def free_bytes(path):
    while not os.path.exists(path):
        path = os.path.dirname(path.rstrip('/')) or '/'
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize

def tree_bytes(path):
    t = 0
    if not os.path.exists(path):
        return 0
    for dp, _, fn in os.walk(path):
        if '_ledger' in dp.split(os.sep):
            continue
        for f in fn:
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t

def sha_file(p, algo='sha256'):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def read_list(p, stage_permissions=()):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r['size_bytes'] = int(r['size_bytes'])
        if r['name'] in FORBIDDEN_NAMES:
            raise Refuse('forbidden artifact in source list: ' + r['name'])
        if '/' in r['name'] or r['name'].startswith('.'):
            raise Refuse('unsafe name ' + r['name'])
    if len({r['name'] for r in rows}) != len(rows):
        raise Refuse('duplicate names')
    return rows

def select(rows, roles=('images',), stage_permissions=()):
    for ro in roles:
        if ro != 'images' and 'STAGE:' + ro not in stage_permissions:
            raise Refuse('role %s requires an explicit stage permission token' % ro)
    return [r for r in rows if r['role'] in roles]

def partial_meta_path(final):
    return final + '.part.json'

def expected_digests(r):
    return {k: r[k] for k in ('expected_md5_from_object_metadata', 'expected_sha256') if r.get(k)}

def verify_complete(final, r):
    """size AND every available expected digest; always returns local receipt (local sha256 is NOT source-authenticated)."""
    if os.path.getsize(final) != r['size_bytes']:
        return (False, {'reason': 'size', 'have': os.path.getsize(final)})
    rec = {'local_sha256_receipt': sha_file(final, 'sha256'), 'receipt_note': 'locally computed; not source-authenticated'}
    if r.get('expected_md5_from_object_metadata'):
        m = sha_file(final, 'md5')
        rec['local_md5'] = m
        if m != r['expected_md5_from_object_metadata']:
            return (False, dict(rec, reason='md5_mismatch'))
    if r.get('expected_sha256') and rec['local_sha256_receipt'] != r['expected_sha256']:
        return (False, dict(rec, reason='sha256_mismatch'))
    return (True, rec)

def fetch_one(r, ctx, source_list_sha, allow_hosts, opener=None, restart_policy='refuse', allow_http=False, chunk=1 << 20, timeout=60):
    u = urllib.parse.urlparse(r['url'])
    if u.scheme != 'https' and (not (allow_http and u.scheme == 'http')):
        raise Refuse('scheme not allowed')
    if u.hostname not in allow_hosts:
        raise Refuse('host not allowlisted')
    d = os.path.join(ctx.dest, 'archives')
    os.makedirs(d, exist_ok=True)
    final = os.path.join(d, r['name'])
    part = final + '.part'
    pm = partial_meta_path(final)
    ctx.check()
    if os.path.exists(final):
        ok, rec = verify_complete(final, r)
        if ok:
            json.dump(rec, open(final + '.receipt.json', 'w'))
            return 'complete_verified'
        raise Refuse('existing file fails content validation; preserved untouched (operator decision required)', decision=dict(action='operator_decide', path=final, **rec))
    ident = {'source_list_sha256': source_list_sha, 'url': r['url'], 'size_bytes': r['size_bytes'], **expected_digests(r), 'etag_listed': r.get('etag', '')}
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have:
        old = json.load(open(pm)) if os.path.exists(pm) else None
        if old is None or {k: old.get(k) for k in ident} != ident:
            raise Refuse('stale or unbound partial state; preserved (identity differs)')
        if have > r['size_bytes']:
            raise Refuse('partial larger than listed size; preserved', decision=dict(action='operator_decide', path=part, have=have))
        if have == r['size_bytes']:
            ok, rec = verify_complete(part, r)
            if not ok:
                raise Refuse('fully received partial fails content validation; preserved untouched (operator decision required)', decision=dict(action='operator_decide', path=part, **rec))
            os.replace(part, final)
            json.dump(dict(rec, promoted_from_partial=True), open(final + '.receipt.json', 'w'))
            if os.path.exists(pm):
                os.remove(pm)
            return 'promoted_verified_partial'
    opener = opener or urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(r['url'])
    if have:
        req.add_header('Range', 'bytes=%d-' % have)
    rid = ctx.reserve(r['name'], r['size_bytes'] - have if have else r['size_bytes'])
    received = 0
    try:
        try:
            resp = _net(opener.open, req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise Refuse('HTTP %s (redirects/errors refused)' % e.code)
        code = resp.status
        cl = int(resp.headers.get('Content-Length', -1))
        server = {'etag': resp.headers.get('ETag', ''), 'last_modified': resp.headers.get('Last-Modified', '')}
        if r.get('etag') and server['etag'].strip('"') not in ('', r['etag']):
            raise Refuse('server ETag differs from listed identity')
        if have and code == 206:
            old = json.load(open(pm))
            if old.get('server') and old['server'].get('etag') and server['etag'] and (old['server']['etag'] != server['etag']):
                raise Refuse('server identity changed since partial was started; preserved')
            m = re.match('bytes (\\d+)-(\\d+)/(\\d+)$', resp.headers.get('Content-Range', ''))
            if not m or int(m.group(1)) != have or int(m.group(2)) != r['size_bytes'] - 1 or (int(m.group(3)) != r['size_bytes']) or (cl != int(m.group(2)) - int(m.group(1)) + 1):
                raise Refuse('inconsistent Content-Range/Content-Length; partial preserved')
            mode, expect = ('ab', r['size_bytes'] - have)
        elif have and code == 200:
            if restart_policy != 'restart_keep_old':
                raise Refuse('server ignored Range; partial preserved; rerun with explicit restart_policy=restart_keep_old')
            n = 1
            while os.path.exists('%s.superseded.%d' % (part, n)):
                n += 1
            os.replace(part, '%s.superseded.%d' % (part, n))
            have = 0
            mode, expect = ('wb', r['size_bytes'])
            ctx.settle(rid, r['name'], 0)
            rid = ctx.reserve(r['name'], r['size_bytes'])
            if cl != r['size_bytes']:
                raise Refuse('Content-Length %d != listed %d' % (cl, r['size_bytes']))
        elif code == 200:
            if cl != r['size_bytes']:
                raise Refuse('Content-Length %d != listed size %d' % (cl, r['size_bytes']))
            mode, expect = ('wb', r['size_bytes'])
        else:
            raise Refuse('unexpected status %d' % code)
        json.dump(dict(ident, server=server), open(pm, 'w'))
        with open(part, mode) as f:
            while True:
                ctx.check()
                ctx.check_resources(min(chunk, expect - received + 1))
                b = _net(resp.read, min(chunk, expect - received + 1))
                if not b:
                    break
                received += len(b)
                if received > expect:
                    keep = b[:max(0, expect - (received - len(b)))]
                    ctx.check_resources(len(keep))
                    f.write(keep)
                    ctx.note_written(len(keep))
                    raise Refuse('server returned more bytes than expected; partial preserved')
                ctx.check()
                ctx.check_resources(len(b))
                f.write(b)
                ctx.note_written(len(b))
        if received != expect:
            raise Refuse('short body %d/%d; partial preserved (resumable)' % (received, expect))
    finally:
        ctx.settle(rid, r['name'], received)
    os.replace(part, final)
    ok, rec = verify_complete(final, r)
    if not ok:
        os.replace(final, final + '.FAILED_VALIDATION')
        raise Refuse('content validation failed after transfer', decision=dict(action='operator_decide', path=final + '.FAILED_VALIDATION', **rec))
    json.dump(rec, open(final + '.receipt.json', 'w'))
    if os.path.exists(pm):
        os.remove(pm)
    return 'downloaded' if mode == 'wb' else 'resumed'
STATE_FLUSH_MEMBERS = 64
STATE_FLUSH_S = 10.0
ZIP_RATIO_MAX = 20.0
ZIP_MEMBER_MAX = 8 * 2 ** 30
ZIP_TOTAL_FACTOR = 4.0

def _atomic_json(path, obj):
    json.dump(obj, open(path + '.tmp', 'w'))
    os.replace(path + '.tmp', path)

def _case_path_map(infos):
    """Case-insensitive volumes (exFAT here) merge paths that differ only by letter case (real case: CQ500-CT-172 has series dirs 'CT Thin PLAIN' and
    'CT Thin Plain'). Every case-variant of a parent directory after the first (in archive order) gets its own '<dir>~caseN'; remaining file-level
    case collisions get '<stem>~caseN<ext>'. Returns {original member name: extracted relative path} for changed members only (deterministic per archive)."""
    seen_dir = {}
    out = {}
    used = set()
    for i in infos:
        n = i.filename
        d, _, f = n.rpartition('/')
        var = seen_dir.setdefault(d.lower(), [])
        if d not in var:
            var.append(d)
        k = var.index(d)
        nd = d if k == 0 else d + '~case%d' % k
        path = nd + '/' + f if d else f
        if path.lower() in used:
            stem, dot, ext = f.rpartition('.')
            j = 1
            while True:
                cand = stem + '~case%d.' % j + ext if dot else f + '~case%d' % j
                cp = nd + '/' + cand if d else cand
                if cp.lower() not in used:
                    path = cp
                    break
                j += 1
        used.add(path.lower())
        if path != n:
            out[n] = path
    return out

def safe_extract(zpath, ctx, name):
    ctx.check()
    final = os.path.join(ctx.dest, 'extracted', os.path.splitext(name)[0])
    work = final + '.partial'
    stp = work + '.state.json'
    if os.path.exists(final):
        raise Refuse('extraction target already exists; not overwritten')
    z = zipfile.ZipFile(zpath)
    infos = [i for i in z.infolist() if not i.is_dir()]
    tot = 0
    arch = os.path.getsize(zpath)
    asha = sha_file(zpath)
    manifest = {i.filename: i.file_size for i in infos}
    pmap = _case_path_map(infos)
    tgt = lambda m: pmap.get(m, m)
    for i in z.infolist():
        n = i.filename
        if n.startswith('/') or '..' in n.replace('\\', '/').split('/') or ':' in n.split('/')[0]:
            raise Refuse('unsafe member path ' + n)
        if i.external_attr >> 16 & 61440 == stat.S_IFLNK:
            raise Refuse('symlink member ' + n)
        if i.file_size > ZIP_MEMBER_MAX:
            raise Refuse('member too large ' + n)
        if i.compress_size and i.file_size / max(i.compress_size, 1) > ZIP_RATIO_MAX and (i.file_size > 10 * 2 ** 20):
            raise Refuse('compression ratio too high ' + n)
        tot += i.file_size
    if tot > max(arch * ZIP_TOTAL_FACTOR, 64 * 2 ** 20):
        raise Refuse('declared extracted size exceeds %.0fx archive' % ZIP_TOTAL_FACTOR)
    st = {'archive_sha256': asha, 'members': manifest, 'done': {}, 'path_map': pmap}
    if os.path.exists(stp):
        old = json.load(open(stp))
        if old.get('archive_sha256') != asha or old.get('members') != manifest:
            raise Refuse('extraction state bound to a different archive/member manifest; partials untouched')
        if old.get('path_map', {}) != pmap:
            raise Refuse('extraction state bound to a different member path mapping (case-collision handling); partials untouched')
        old['path_map'] = pmap
        for m, h in old['done'].items():
            p = os.path.join(work, tgt(m))
            if not os.path.exists(p) or os.path.getsize(p) != manifest.get(m) or sha_file(p) != h:
                raise Refuse('completed member %s missing or corrupt; not accepted; partial untouched' % m)
        st = old
    elif os.path.isdir(work) and os.listdir(work):
        raise Refuse('unbound partial extraction directory (no identity state); untouched')
    os.makedirs(work, exist_ok=True)
    for dp, _, fn in os.walk(work):
        for f_ in fn:
            if f_.endswith('.tmp') and (not f_.endswith('state.json.tmp')) and (not f_.startswith('._')):
                try:
                    os.remove(os.path.join(dp, f_))
                except FileNotFoundError:
                    pass
    if not os.path.exists(stp):
        _atomic_json(stp, st)
    ctx.admit_bytes(0)
    remaining = tot - sum((manifest[m] for m in st['done']))
    if ctx.stored(force=True) + remaining > ctx.storage_cap:
        raise Refuse('storage cap would be exceeded by extraction (archives retained)')
    if ctx.free_fn(ctx.dest) < remaining + ctx.min_free:
        raise Refuse('free space insufficient for extraction')
    since, last_flush = (0, time.time())
    try:
        for i in infos:
            if i.filename in st['done']:
                continue
            ctx.check()
            ctx.check_resources(i.file_size)
            t = os.path.realpath(os.path.join(work, tgt(i.filename)))
            if not t.startswith(os.path.realpath(work) + os.sep):
                raise Refuse('path escapes target ' + i.filename)
            os.makedirs(os.path.dirname(t), exist_ok=True)
            n = 0
            h = hashlib.sha256()
            with z.open(i) as s_, open(t + '.tmp', 'wb') as f:
                while True:
                    ctx.check()
                    ctx.check_resources(min(1 << 20, i.file_size - n + 1))
                    b = s_.read(min(1 << 20, i.file_size - n + 1))
                    if not b:
                        break
                    n += len(b)
                    if n > i.file_size:
                        raise Refuse('actual bytes exceed declared member size')
                    ctx.check_resources(len(b))
                    f.write(b)
                    h.update(b)
                    ctx.note_written(len(b))
            if n != i.file_size:
                raise Refuse('member size mismatch ' + i.filename)
            os.replace(t + '.tmp', t)
            st['done'][i.filename] = h.hexdigest()
            since += 1
            if since >= STATE_FLUSH_MEMBERS or time.time() - last_flush >= STATE_FLUSH_S:
                _atomic_json(stp, st)
                since = 0
                last_flush = time.time()
    except Stopped:
        try:
            _atomic_json(stp, st)
        except OSError:
            pass
        raise
    _atomic_json(stp, st)
    for m, h in st['done'].items():
        if sha_file(os.path.join(work, tgt(m))) != h:
            raise Refuse('final verification failed for ' + m)
    os.replace(work, final)
    if pmap:
        _atomic_json(final + '.CASE_PATH_MAP.json', {'archive_sha256': asha, 'note': 'member -> extracted relative path (case-insensitive volume)', 'map': pmap})
    _atomic_json(final + '.EXTRACTED.json', {'archive_sha256': asha, 'verified_members': len(st['done']), 'declared_bytes': tot, 'complete': True, 'case_variant_members': len(pmap)})
    return tot

def probe_header_bounded(url, allow_hosts, opener, max_header_bytes=512, allow_http=False):
    """One-byte ranges only: no requested byte lies beyond the first newline. 200 or any range anomaly is refused BEFORE a body is read."""
    u = urllib.parse.urlparse(url)
    if u.hostname not in allow_hosts or (u.scheme != 'https' and (not (allow_http and u.scheme == 'http'))):
        raise Refuse('probe host/scheme not allowed')
    out = bytearray()
    for i in range(max_header_bytes + 1):
        req = urllib.request.Request(url)
        req.add_header('Range', 'bytes=%d-%d' % (i, i))
        try:
            resp = opener.open(req, timeout=30)
        except urllib.error.HTTPError as e:
            raise Refuse('probe HTTP %s' % e.code)
        if resp.status != 206:
            raise Refuse('probe: server did not honour Range (status %d); body not read' % resp.status)
        m = re.match('bytes (\\d+)-(\\d+)/(\\d+|\\*)$', resp.headers.get('Content-Range', ''))
        if not m or int(m.group(1)) != i or int(m.group(2)) != i or (resp.headers.get('Content-Length') != '1'):
            raise Refuse('probe: malformed/unexpected range; body not read')
        b = resp.read(2)
        if len(b) != 1:
            raise Refuse('probe: body length != 1')
        if b == b'\n':
            return out.decode('utf-8', 'replace').rstrip('\r')
        out += b
    raise Refuse('header line exceeds %d bytes: unresolved; request the schema from the source instead' % max_header_bytes)
RETRY_BACKOFF_S = (15, 30, 60, 120, 240, 300)

def _fetch_with_retries(r, ctx, sl, hosts, opener, restart_policy, allow_http, chunk, sleep_fn, backoff):
    errs = []
    for k in range(len(backoff) + 1):
        try:
            res = fetch_one(r, ctx, sl, hosts, opener, restart_policy, allow_http, chunk)
            return {'name': r['name'], 'result': res, 'transient_retries': k, 'transient_errors': errs}
        except Transient as e:
            errs.append({'attempt': k + 1, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'error': str(e)})
            if k == len(backoff):
                break
            end = time.time() + backoff[k]
            while time.time() < end:
                ctx.check()
                sleep_fn(min(5, max(0, end - time.time())))
    return {'name': r['name'], 'result': 'TRANSIENT_EXHAUSTED', 'why': 'network errors on %d attempts; identity-bound partial preserved (resumable)' % len(errs), 'transient_errors': errs}

def run_fetch(op, deadline, opener=None, clock=time.time, restart_policy='refuse', allow_http=False, chunk=1 << 20, stop_files=(), sleep_fn=time.sleep, backoff=RETRY_BACKOFF_S, workers=1):
    for k in ('source_list_sha256', 'expected_files', 'accounting_root', 'expected_payload_bytes', 'dest'):
        if not op.get(k):
            raise Refuse('operation config missing %s (refused before any write)' % k, code=6)
    if sha_file(op['source_list']) != op['source_list_sha256']:
        raise Refuse('source list digest differs from the approved operation', code=6)
    if any((os.path.exists(p) for p in list(op.get('stop_files', [])) + list(stop_files))):
        raise Refuse('STOP present; refused before any write', code=4)
    rows = read_list(op['source_list'])
    sel = select(rows, tuple(op.get('roles', ['images'])), tuple(op.get('stage_permissions', [])))
    if len(sel) != op['expected_files']:
        raise Refuse('file count differs from approved operation', code=6)
    if sum((r['size_bytes'] for r in sel)) != op.get('expected_payload_bytes', sum((r['size_bytes'] for r in sel))):
        raise Refuse('payload differs from approved operation bytes')
    base = None
    if op.get('baseline_file') and os.path.exists(op['baseline_file']):
        try:
            b = json.load(open(op['baseline_file']))
        except Exception:
            raise Refuse('baseline file malformed', code=6)
        if not (isinstance(b, dict) and isinstance(b.get('bytes'), int) and (not isinstance(b.get('bytes'), bool)) and (b['bytes'] >= 0) and b.get('recorded_by') and b.get('basis')):
            raise Refuse('baseline malformed/unestablished (needs nonnegative bytes, recorded_by, basis)', code=6)
        base = b['bytes']
    if base is None:
        raise Refuse('cumulative baseline unresolved: no recorded approved baseline; refusing before any destination write (no guessed zero)', code=6)
    if not os.path.isdir(op['accounting_root']):
        raise Refuse('accounting root missing (volume not mounted?); refused before any write', code=6)
    ctx = OpContext(op['dest'], deadline, list(op.get('stop_files', [])) + list(stop_files), clock, op.get('storage_cap_bytes', 400 * 10 ** 9), op.get('cumulative_cap_bytes', 200 * 10 ** 9), op.get('min_free_bytes', 50 * 10 ** 9), base, accounting_root=op['accounting_root'], free_fn=op.get('_free_fn'), tree_fn=op.get('_tree_fn'))
    os.makedirs(op['dest'], exist_ok=True)
    ctx.acquire()
    sl = sha_file(op['source_list'])
    log = []
    try:
        ctx.cumulative_used()

        def work(r):
            try:
                return _fetch_with_retries(r, ctx, sl, set(op['allow_hosts']), opener, restart_policy, allow_http, chunk, sleep_fn, backoff)
            except Refuse as e:
                return {'name': r['name'], 'result': 'REFUSED', 'why': str(e), 'decision': e.decision}
        if workers <= 1:
            for r in sel:
                log.append(work(r))
        else:
            first_stop = None
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(work, r) for r in sel]
                for f in futs:
                    try:
                        log.append(f.result())
                    except concurrent.futures.CancelledError:
                        pass
                    except Stopped as s_:
                        ctx._stop_evt.set()
                        [x.cancel() for x in futs]
                        if first_stop is None or first_stop == 'peer_worker_stopped':
                            first_stop = str(s_)
            if first_stop is not None:
                raise Stopped(first_stop)
    except Stopped as s:
        log.append({'result': 'STOPPED', 'why': str(s), 'resume': 'state preserved'})
        return (4, log, ctx)
    finally:
        ctx.release()
    return (0 if all((l['result'] in ('downloaded', 'resumed', 'complete_verified', 'promoted_verified_partial') for l in log)) else 3, log, ctx)

def _baseline(op):
    if not (op.get('baseline_file') and os.path.exists(op['baseline_file'])):
        raise Refuse('cumulative baseline unresolved', code=6)
    b = json.load(open(op['baseline_file']))
    if not (isinstance(b, dict) and isinstance(b.get('bytes'), int) and (not isinstance(b.get('bytes'), bool)) and (b['bytes'] >= 0) and b.get('recorded_by') and b.get('basis')):
        raise Refuse('baseline malformed', code=6)
    return b['bytes']

def run_extract(op, deadline, clock=time.time, stop_files=(), workers=1):
    """Extract every verified archive (receipt present) of the approved operation; identity-bound resume; never overwrites a completed extraction."""
    if sha_file(op['source_list']) != op['source_list_sha256']:
        raise Refuse('source list digest differs from the approved operation', code=6)
    if any((os.path.exists(p) for p in list(op.get('stop_files', [])) + list(stop_files))):
        raise Refuse('STOP present; refused before any write', code=4)
    sel = select(read_list(op['source_list']), tuple(op.get('roles', ['images'])), tuple(op.get('stage_permissions', [])))
    if not os.path.isdir(op['accounting_root']):
        raise Refuse('accounting root missing (volume not mounted?); refused before any write', code=6)
    ctx = OpContext(op['dest'], deadline, list(op.get('stop_files', [])) + list(stop_files), clock, op.get('storage_cap_bytes', 400 * 10 ** 9), op.get('cumulative_cap_bytes', 200 * 10 ** 9), op.get('min_free_bytes', 50 * 10 ** 9), _baseline(op), accounting_root=op['accounting_root'], free_fn=op.get('_free_fn'), tree_fn=op.get('_tree_fn'))
    ctx.acquire()
    log = []

    def one(r):
        ctx.check()
        z = os.path.join(op['dest'], 'archives', r['name'])
        stem = os.path.splitext(r['name'])[0]
        if os.path.exists(os.path.join(op['dest'], 'extracted', stem + '.EXTRACTED.json')):
            return {'name': r['name'], 'result': 'already_extracted'}
        if not (os.path.exists(z) and os.path.exists(z + '.receipt.json')):
            return {'name': r['name'], 'result': 'NOT_DOWNLOADED'}
        try:
            return {'name': r['name'], 'result': 'extracted', 'bytes': safe_extract(z, ctx, r['name'])}
        except Refuse as e:
            return {'name': r['name'], 'result': 'REFUSED', 'why': str(e)}
        except zipfile.BadZipFile as e:
            return {'name': r['name'], 'result': 'REFUSED', 'why': 'unreadable archive (BadZipFile: %s); not extracted, archive untouched' % e}
    try:
        if workers <= 1:
            for r in sel:
                log.append(one(r))
        else:
            first_stop = None
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(one, r) for r in sel]
                for f in futs:
                    try:
                        log.append(f.result())
                    except concurrent.futures.CancelledError:
                        pass
                    except Stopped as s_:
                        ctx._stop_evt.set()
                        [x.cancel() for x in futs]
                        if first_stop is None or first_stop == 'peer_worker_stopped':
                            first_stop = str(s_)
            if first_stop is not None:
                raise Stopped(first_stop)
    except Stopped as s:
        log.append({'result': 'STOPPED', 'why': str(s), 'resume': 'state preserved'})
        return (4, log)
    finally:
        ctx.release()
    return (0 if all((l['result'] in ('extracted', 'already_extracted') for l in log)) else 3, log)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'fetch', 'extract'])
    ap.add_argument('--operation', required=True)
    ap.add_argument('--deadline-utc')
    ap.add_argument('--log-json')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args(argv)
    op = json.load(open(a.operation))
    if a.cmd == 'extract':
        code, log = run_extract(op, a.deadline_utc, workers=a.workers)
        if a.log_json:
            json.dump(log, open(a.log_json, 'w'), indent=1, default=str)
        print(json.dumps({'code': code, 'n': len(log), 'counts': {k: sum((1 for l in log if l['result'] == k)) for k in {l['result'] for l in log}}}))
        return code
    if a.cmd == 'plan':
        rows = select(read_list(op['source_list']), tuple(op.get('roles', ['images'])))
        tot = sum((r['size_bytes'] for r in rows))
        bl = json.load(open(op['baseline_file']))['bytes'] if op.get('baseline_file') and os.path.exists(op['baseline_file']) else None
        print(json.dumps({'files': len(rows), 'payload_bytes': tot, 'payload_expected': op.get('expected_payload_bytes'), 'dest': op['dest'], 'dest_exists': os.path.exists(op['dest']), 'stored_bytes_at_dest': tree_bytes(op['dest']), 'free_bytes': free_bytes(op['dest']), 'cumulative_baseline': bl if bl is not None else 'UNRESOLVED (no approved baseline file; fetch will refuse)', 'extracted_bytes': 'UNKNOWN', 'ready': 'ready_except_baseline_and_authorization' if bl is None else 'ready_except_authorization'}, indent=1))
        return 0
    code, log, _ = run_fetch(op, a.deadline_utc, workers=a.workers)
    if a.log_json:
        json.dump(log, open(a.log_json, 'w'), indent=1, default=str)
    print(json.dumps({'code': code, 'n': len(log), 'counts': {k: sum((1 for l in log if l['result'] == k)) for k in {l['result'] for l in log}}}))
    return code
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Refuse as e:
        sys.stderr.write('REFUSED: %s\n' % e)
        sys.exit(e.code)

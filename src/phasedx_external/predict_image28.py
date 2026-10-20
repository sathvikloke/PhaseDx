"""predict_image28.py: manifest identity is RECOMPUTED from consumed files and compared to the execution lock BEFORE model load; resume re-hashes completed exams' slice files once before reuse.
usage: --design-lock D --execution-lock E --role primary_image|secondary_image --manifest-dir M --checkpoint-dir DIR --out RUNDIR [--max-exams N]"""

def main(argv=None):
    import sys, os, json, argparse, time, datetime, hashlib
    HERE = os.path.dirname(os.path.abspath(__file__))
    B = os.path.dirname(HERE)
    ap = argparse.ArgumentParser()
    [ap.add_argument('--' + k, required=True) for k in ('design-lock', 'execution-lock', 'role', 'manifest-dir', 'checkpoint-dir', 'out')]
    ap.add_argument('--max-exams', type=int, default=0)
    ap.add_argument('--cycle-control', required=True)
    ap.add_argument('--decoder-policy', choices=('uncompressed_v1', 'lossless_sv1_pylibjpeg_v1'), default='uncompressed_v1')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--exit-zero-on-checkpoint', action='store_true')
    ap.add_argument('--admission-report', default='')
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    require_registration(__file__, None)
    from . import protocol28 as P
    if a.role not in ('primary_image', 'secondary_image'):
        sys.stderr.write('role must be an image role\n')
        sys.exit(2)

    def fail(msg):
        sys.stderr.write('ADMISSION_FAILED: %s\n' % msg)
        if a.admission_report:
            json.dump({'exit_code': 5, 'message': msg, 'model_loaded': False}, open(a.admission_report, 'w'))
            sys.exit(0)
        sys.exit(5)
    design = json.load(open(a.design_lock))
    policy = json.load(open(os.path.join(a.manifest_dir, 'policy.json')))
    if a.decoder_policy != design['preprocessing'].get('decoder_policy', 'uncompressed_v1'):
        fail('decoder policy differs from frozen design')
    if a.decoder_policy != policy.get('decoder_policy', 'uncompressed_v1'):
        fail('decoder policy differs from the input manifest')
    if a.decoder_policy != 'uncompressed_v1' and not all(design['preprocessing'].get(k) for k in ('decoder_sha256', 'adapter_sha256', 'decoder_packages')):
        fail('lossless policy requires source and package identities before prediction')
    actual = {'model_id': design['models'][a.role]['model_id'], 'checkpoint_sha256': P.sha_file(a.checkpoint_dir.rstrip('/') + '/best_model.pt'), 'recipe_sha256': P.sha_file(HERE + '/a_ct_recipe_b2_v7_verbatim.py'), 'precision_path': 'recipe_f32->float16_roundtrip->f32_normalize', 'aggregation': 'max_slice_logit', 'representation_id': 'n/a'}
    if 'decoder_policy' in design['preprocessing']:
        import importlib.metadata
        actual.update(decoder_policy=a.decoder_policy, decoder_sha256=P.sha_file(HERE + '/dicom_decode.py'),
                      adapter_sha256=P.sha_file(HERE + '/image_adapter27.py'),
                      decoder_packages={k: importlib.metadata.version(k) for k in design['preprocessing']['decoder_packages']})
    try:
        start, resumed = P.start_run(a.out, a.role, a.design_lock, a.execution_lock, a.manifest_dir, actual)
    except P.AdmissionError as e:
        fail(str(e))
    import pandas as pd
    cand = pd.read_csv(a.manifest_dir + '/candidates.csv', dtype=str).fillna('')
    man = pd.read_csv(a.manifest_dir + '/slices.csv', dtype=str).fillna('')
    by = {k: g for k, g in man.groupby('exam_key', sort=True)}
    jp = a.out + '/exam_rows.jsonl'
    done = {}
    if os.path.exists(jp):
        for l in open(jp):
            if l.strip():
                r = json.loads(l)
                done[r['exam_key']] = r
    if resumed and done:
        for ek in done:
            for p, h in zip(by.get(ek, pd.DataFrame(columns=['dicom_path', 'sha256']))['dicom_path'], by.get(ek, pd.DataFrame(columns=['dicom_path', 'sha256']))['sha256']):
                if not os.path.exists(p) or P.sha_file(p) != h:
                    fail('completed exam %s: source file changed or missing since inventory (%s); predictions not reusable' % (ek, p))
    from . import image_adapter27 as IA
    ctl = json.load(open(a.cycle_control))
    deadline = datetime.datetime.fromisoformat(ctl['deadline_utc']).timestamp()
    stops = [os.path.join(os.path.dirname(a.cycle_control), 'STOP_REQUESTED')]
    if any(os.path.exists(s) for s in stops) or time.time() >= deadline:
        fail('STOP or deadline before model load')
    model, prov = IA.load_model(a.checkpoint_dir.rstrip('/') + '/best_model.pt', actual['checkpoint_sha256'])
    n_new = 0
    status = 'complete'
    with open(jp, 'a') as out:
        for ek, he in zip(cand['exam_key'], cand['header_eligibility']):
            if ek in done:
                continue
            if any((os.path.exists(s) for s in stops)) or time.time() > deadline:
                status = 'checkpoint_stop_or_deadline'
                break
            if a.max_exams and n_new >= a.max_exams:
                status = 'checkpoint_max_exams'
                break
            if he != 'eligible':
                r = {'n_listed': 0, 'n_scored': 0, 'n_rejected': 0, 'reasons': {}, 'score': float('nan'), 'status': 'not_eligible_' + he}
            elif ek not in by:
                r = {'n_listed': 0, 'n_scored': 0, 'n_rejected': 0, 'reasons': {}, 'score': float('nan'), 'status': 'no_slices_in_manifest'}
            else:
                g = by[ek]
                r = IA.score_exam(model, [{'dicom_path': p, 'sop': s, 'sha256': h} for p, s, h in zip(g['dicom_path'], g['sop'], g['sha256'])], a.batch_size, decoder_policy=a.decoder_policy)
            r.update({'exam_key': ek, 'model_id': actual['model_id'], 'run_id': start['run_id'], 'execution_lock_sha256': start['execution_lock_sha256'], 'identity_sha256': start['actual_sha256']})
            out.write(json.dumps(r) + '\n')
            out.flush()
            done[ek] = r
            n_new += 1
    rows = [done[e] for e in cand['exam_key'] if e in done]
    pp = a.out + '/predictions.csv'
    _df = pd.DataFrame(rows, columns=None if rows else ['exam_key', 'model_id', 'score', 'status', 'reasons', 'run_id', 'execution_lock_sha256', 'identity_sha256'])
    if len(_df):
        _df['reasons'] = _df['reasons'].map(json.dumps)
    _df.to_csv(pp, index=False)
    if status == 'complete':
        P.end_run(a.out, pp, len(rows))
    json.dump({'status': status, 'n_roster': len(cand), 'n_done': len(rows), 'n_new': n_new}, open(a.out + '/RUN_STATUS.json', 'w'))
    print(status, len(rows), n_new)
    sys.exit(0 if status == 'complete' or a.exit_zero_on_checkpoint else 92)

"""predict_geometry28.py: transported physical-rank G2 (frozen reconstructed). Manifest identity recomputed from consumed files before the model is unpickled; slices.csv numeric headers are part of the identity."""

def main(argv=None):
    import sys, os, json, argparse
    HERE = os.path.dirname(os.path.abspath(__file__))
    B = os.path.dirname(HERE)
    ap = argparse.ArgumentParser()
    [ap.add_argument('--' + k, required=True) for k in ('design-lock', 'execution-lock', 'manifest-dir', 'model', 'out')]
    ap.add_argument('--admission-report', default='')
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    require_registration(__file__, None)
    from . import protocol28 as P
    REP = 'TRANSPORTED_PHYSICAL_RANK_FROZEN_RECONSTRUCTED_G2'
    design = json.load(open(a.design_lock))
    actual = {'model_id': design['models']['geometry']['model_id'], 'checkpoint_sha256': P.sha_file(a.model), 'recipe_sha256': 'n/a', 'precision_path': 'n/a', 'aggregation': 'n/a', 'representation_id': REP}
    try:
        start, _ = P.start_run(a.out, 'geometry', a.design_lock, a.execution_lock, a.manifest_dir, actual)
    except P.AdmissionError as e:
        sys.stderr.write('ADMISSION_FAILED: %s\n' % e)
        if a.admission_report:
            json.dump({'exit_code': 5, 'message': str(e), 'model_loaded': False}, open(a.admission_report, 'w'))
            sys.exit(0)
        sys.exit(5)
    import pandas as pd
    from . import geometry_adapter27 as GA
    model = GA.load_model(a.model, actual['checkpoint_sha256'])
    H = pd.read_csv(a.manifest_dir + '/slices.csv', dtype={'exam_key': str, 'series_key': str})
    C = pd.read_csv(a.manifest_dir + '/candidates.csv', dtype=str).fillna('')
    res = GA.predict_external(model, H, list(C['exam_key']), {k: v for k, v in zip(C['exam_key'], C['selected_series_key']) if v}, False)
    res = res.assign(model_id=actual['model_id'], representation_id=REP, run_id=start['run_id'], execution_lock_sha256=start['execution_lock_sha256'], identity_sha256=start['actual_sha256'])
    pp = a.out + '/predictions.csv'
    res.to_csv(pp, index=False)
    P.end_run(a.out, pp, len(res))
    print(res['status'].value_counts().to_dict())

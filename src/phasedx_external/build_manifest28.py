"""build_manifest28.py: roster-first inventory + outcome-blind clinical series selection + component-digest identity. usage:
build_manifest28.py --roster R.csv --root DIR --policy technical_policy.json --out OUT [--selection series_selection.csv]
roster columns: source_exam_id[,folder][,expected_study_uid]. Without --selection every exam is clinical_unresolved and a reviewer packet (selection_packet.csv, metadata only) is written."""

def main(argv=None):
    import sys, os, json, hashlib, argparse, csv, shutil
    HERE = os.path.dirname(os.path.abspath(__file__))
    B = os.path.dirname(HERE)
    ap = argparse.ArgumentParser()
    [ap.add_argument('--' + k, required=True) for k in ('roster', 'root', 'policy', 'out')]
    ap.add_argument('--selection', default='')
    ap.add_argument('--header-cache', help='Authenticated complete header-shard index; avoids another image read')
    a = ap.parse_args(argv)
    from .prospective_guard import require_registration
    require_registration(__file__, None)
    import numpy as np, pandas as pd
    from . import geometry_adapter27 as GA, selection28 as SEL, protocol28 as P
    pol = json.load(open(a.policy))
    from .dicom_decode import allowed_syntaxes
    SUP_TS = allowed_syntaxes(pol.get('decoder_policy', 'uncompressed_v1'))
    root = os.path.abspath(a.root)
    out = a.out
    os.makedirs(out, exist_ok=True)
    ros = pd.read_csv(a.roster, dtype=str).fillna('')
    for optional in ('folder', 'expected_study_uid'):
        if optional not in ros:
            ros[optional] = ''
    if ros['source_exam_id'].duplicated().any() or (ros['source_exam_id'] == '').any():
        sys.stderr.write('roster ids must be unique, non-empty\n')
        sys.exit(2)
    folder_of = {r.source_exam_id: r.folder or r.source_exam_id for r in ros.itertuples()}
    unit_of = {v: k for k, v in folder_of.items()}

    def hdr(path):
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        g = lambda k, d='': ds.get(k) if ds.get(k) is not None else d
        ipp = [float(x) for x in g('ImagePositionPatient', [])]
        iop = [float(x) for x in g('ImageOrientationPatient', [])]
        return dict(StudyInstanceUID=str(g('StudyInstanceUID')), SeriesInstanceUID=str(g('SeriesInstanceUID')), SOPInstanceUID=str(g('SOPInstanceUID')), instance_number=g('InstanceNumber'), modality=str(g('Modality')), transfer_syntax=str(getattr(getattr(ds, 'file_meta', None), 'TransferSyntaxUID', '')), number_of_frames=g('NumberOfFrames', 1), has_rescale=bool('RescaleSlope' in ds and 'RescaleIntercept' in ds), contrast_agent=str(g('ContrastBolusAgent')), convolution_kernel=str(g('ConvolutionKernel')), series_description=str(g('SeriesDescription')), image_type=str(g('ImageType')), body_part=str(g('BodyPartExamined')), manufacturer=str(g('Manufacturer')), lossy=str(g('LossyImageCompression', '00')), slice_thickness=g('SliceThickness'), n_ipp=len(ipp), n_iop=len(iop), **{f'ipp{i}': ipp[i] if len(ipp) == 3 else np.nan for i in range(3)}, **{f'iop{i}': iop[i] if len(iop) == 6 else np.nan for i in range(6)})

    def shaf(p):
        return P.sha_file(p)
    rows = []
    if a.header_cache:
        from .header_cache import load
        rows = load(a.header_cache, root)
        for r in rows:
            top = r['relpath'].split(os.sep)[0] if os.sep in r['relpath'] else ''
            if r['source_exam_id'] != unit_of.get(top, ''):
                raise ValueError('cached source-to-roster assignment differs from current roster')
    elif not os.path.isdir(root):
        raise FileNotFoundError('image root is unavailable; no cohort can be enumerated')
    for dp, dn, fn in ([] if a.header_cache else os.walk(root)):
        dn.sort()
        for f in sorted(fn):
            if f.startswith('.'):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root)
            top = rel.split(os.sep)[0] if os.sep in rel else ''
            cand = unit_of.get(top, '') if top else ''
            d = {'relpath': rel, 'size': os.path.getsize(p), 'sha256': shaf(p), 'source_exam_id': cand, 'assignment': 'roster_folder' if cand else 'root_level_file' if not top else 'folder_not_in_roster', 'readable': True, 'unreadable_reason': ''}
            try:
                d.update(hdr(p))
            except Exception as e:
                d.update(readable=False, unreadable_reason=type(e).__name__)
            rows.append(d)
    inv = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['relpath', 'size', 'sha256', 'source_exam_id', 'assignment', 'readable'])
    inv['disposition'] = 'kept'
    for c in ('StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'):
        if c not in inv:
            inv[c] = ''
    inv = inv.fillna({'StudyInstanceUID': '', 'SeriesInstanceUID': '', 'SOPInstanceUID': ''})
    ok = inv[(inv['source_exam_id'] != '') & inv['readable']]
    conflict_series, collide = ({}, {})
    for (ek, sop), g in ok[ok['SOPInstanceUID'] != ''].groupby(['source_exam_id', 'SOPInstanceUID']):
        if len(g) > 1:
            if g['sha256'].nunique() == 1:
                inv.loc[g.index[1:], 'disposition'] = 'collapsed_exact_copy'
            else:
                inv.loc[g.index, 'disposition'] = 'conflict_same_sop_different_bytes'
                conflict_series.setdefault(ek, set()).update(g['SeriesInstanceUID'])
    for col in ('SOPInstanceUID', 'SeriesInstanceUID'):
        for k, g in ok[ok[col] != ''].groupby(col):
            ids = sorted(g['source_exam_id'].unique())
            if len(ids) > 1:
                for e in ids:
                    collide.setdefault(e, set()).add(col + ':' + k)
                m = inv.loc[g.index, 'disposition'] == 'kept'
                inv.loc[g.index[m.to_numpy()], 'disposition'] = 'uid_collision_across_candidates'
    live = inv[(inv['source_exam_id'] != '') & inv['readable'] & (inv['disposition'] == 'kept')]
    srows = []
    for (ek, sk), g in live.groupby(['source_exam_id', 'SeriesInstanceUID']):
        d, _ = GA.series_row(sk, ek, g.rename(columns={'SOPInstanceUID': 'sop'}).assign(exam_key=ek))
        t = []
        if pol.get('modality') and (g['modality'] != pol['modality']).any():
            t.append('modality')
        if g['n_ipp'].lt(3).any() or g['n_iop'].lt(6).any():
            t.append('missing_ipp_iop')
        elif not d['orientation_valid']:
            t.append('orientation_invalid')
        if len(g) < 2:
            t.append('single_slice')
        if not g['transfer_syntax'].isin(SUP_TS).all():
            t.append('unsupported_transfer_syntax')
        if 'lossy' in g and (g['lossy'].astype(str) == '01').any():
            t.append('declared_lossy_compression')
        if (pd.to_numeric(g['number_of_frames'], errors='coerce').fillna(1) > 1).any():
            t.append('multiframe')
        if not g['has_rescale'].all():
            t.append('missing_rescale')
        if sk in conflict_series.get(ek, set()):
            t.append('sop_conflict_bytes')
        if any((x.startswith('SeriesInstanceUID:' + sk) for x in collide.get(ek, set()))):
            t.append('uid_collision')
        d.update(source_exam_id=ek, series_key=sk, n_files_kept=len(g), technical_usable=not t, technical_reason=';'.join(t), study_uid=g['StudyInstanceUID'].iloc[0], series_description='|'.join(sorted(set(g['series_description']))), convolution_kernel='|'.join(sorted(set(g['convolution_kernel']))), contrast_agent='|'.join(sorted(set(g['contrast_agent']))), image_type='|'.join(sorted(set(g['image_type']))), body_part='|'.join(sorted(set(g['body_part']))), manufacturer='|'.join(sorted(set(g['manufacturer']))), slice_thickness='|'.join(sorted(set(map(str, g['slice_thickness'])))))
        srows.append(d)
    for ek, ss in conflict_series.items():
        for sk in ss:
            if not any((r['source_exam_id'] == ek and r['series_key'] == sk for r in srows)):
                srows.append({'source_exam_id': ek, 'series_key': sk, 'n_files_kept': 0, 'technical_usable': False, 'technical_reason': 'sop_conflict_bytes', 'study_uid': '', 'span_mm': None, 'median_gap_mm': None})
    ser = pd.DataFrame(srows)
    if not len(ser):
        ser = pd.DataFrame(columns=['source_exam_id', 'series_key', 'technical_usable', 'technical_reason', 'span_mm', 'median_gap_mm', 'study_uid'])
    packet = ser[['source_exam_id', 'study_uid', 'series_key', 'n_files_kept', 'slice_thickness', 'span_mm', 'median_gap_mm', 'series_description', 'convolution_kernel', 'contrast_agent', 'image_type', 'body_part', 'manufacturer', 'technical_usable', 'technical_reason']].rename(columns={'series_key': 'series_uid'}) if len(ser) else pd.DataFrame()
    packet.to_csv(out + '/selection_packet.csv', index=False)
    selcols = SEL.COLS
    if a.selection:
        shutil.copy(a.selection, out + '/series_selection.csv')
    else:
        pd.DataFrame(columns=selcols).to_csv(out + '/series_selection.csv', index=False)
    sel = SEL.load_selection(out + '/series_selection.csv')
    cand, chosen = ([], {})
    for r in ros.itertuples():
        e = r.source_exam_id
        files = inv[inv['source_exam_id'] == e]
        rd = files[files['readable']]
        uids = sorted(set(rd['StudyInstanceUID']) - {''})
        uid_ok = not (len(uids) > 1 or (r.expected_study_uid and (not (len(uids) == 1 and uids[0] == r.expected_study_uid))))
        why = []
        if e in collide:
            why.append('uid_collision:%d' % len(collide[e]))
        if len(files) == 0:
            cs, he, ss_, sr = ('candidate_no_source_files', 'not_eligible', 'no_source_files', 'no source files')
        elif len(rd) == 0:
            cs, he, ss_, sr = ('candidate_unreadable_only', 'not_eligible', 'no_readable_dicom', 'no readable DICOM')
        else:
            cs = 'candidate_readable'
            if e in collide:
                he, ss_, sr = ('not_eligible', 'uid_collision', ';'.join(sorted(collide[e]))[:200])
            else:
                ex = ser[ser['source_exam_id'] == e].to_dict('records') if len(ser) else []
                sk, ss_, sr = SEL.select(e, ex, sel, uid_ok)
                if ss_ == 'selected':
                    chosen[e] = sk
                he = 'eligible' if ss_ == 'selected' else 'unresolved' if ss_ in ('clinical_unresolved', 'study_uid_unresolved') else 'not_eligible'
        cand.append({'exam_key': e, 'source_file_count': len(files), 'readable_count': len(rd), 'resolved_study_uids': '|'.join(uids), 'expected_study_uid': r.expected_study_uid, 'study_uid_resolution': 'ok' if uid_ok else 'UNRESOLVED', 'candidate_status': cs, 'header_eligibility': he, 'selection_status': ss_, 'selection_reason': sr, 'selected_series_key': chosen.get(e, ''), 'image_decode_prediction_status': 'pending', 'geometry_applicability_status': 'pending', 'label_status': 'not_joined', 'paired_evaluation_eligibility': 'pending'})
    cand = pd.DataFrame(cand)
    sl = []
    for e, sk in chosen.items():
        g = live[(live['source_exam_id'] == e) & (live['SeriesInstanceUID'] == sk)].drop_duplicates('SOPInstanceUID')
        for _, x in g.iterrows():
            sl.append({'exam_key': e, 'series_key': sk, 'dicom_path': os.path.join(root, x['relpath']), 'sop': x['SOPInstanceUID'], 'sha256': x['sha256'], 'instance_number': x['instance_number'], **{f'ipp{i}': x[f'ipp{i}'] for i in range(3)}, **{f'iop{i}': x[f'iop{i}'] for i in range(6)}})
    sl = pd.DataFrame(sl, columns=['exam_key', 'series_key', 'dicom_path', 'sop', 'sha256', 'instance_number'] + [f'ipp{i}' for i in range(3)] + [f'iop{i}' for i in range(6)])
    pd.DataFrame([{'exam_key': e, 'series_key': chosen.get(e, '')} for e in ros['source_exam_id']]).to_csv(out + '/selected_series.csv', index=False)
    inv_out = inv[['relpath', 'size', 'sha256', 'source_exam_id', 'assignment', 'readable', 'unreadable_reason', 'disposition', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']].sort_values('relpath')
    inv_out.to_csv(out + '/inventory.csv', index=False)
    cand.to_csv(out + '/candidates.csv', index=False)
    ser.to_csv(out + '/series.csv', index=False)
    sl.to_csv(out + '/slices.csv', index=False)
    sl.to_csv(out + '/geometry_headers.csv', index=False)
    inv_out[inv_out['assignment'] != 'roster_folder'][['relpath', 'size', 'sha256', 'assignment']].to_csv(out + '/unassigned_files.csv', index=False)
    shutil.copy(a.roster, out + '/roster.csv')
    shutil.copy(a.policy, out + '/policy.json')
    if a.header_cache:
        shutil.copy(a.header_cache, out + '/HEADER_CACHE.json')
    comp = {c: shaf(os.path.join(out, c)) for c in P.COMPONENTS}
    inv_digest = hashlib.sha256('\n'.join((f'{r.relpath}\t{r.sha256}' for r in inv_out.itertuples())).encode()).hexdigest()
    fl = {'roster_candidates': len(cand), 'files_total': len(inv), 'files_unassigned': int((inv['assignment'] != 'roster_folder').sum()), 'files_collapsed_exact_copy': int((inv['disposition'] == 'collapsed_exact_copy').sum()), 'files_conflict_bytes': int((inv['disposition'] == 'conflict_same_sop_different_bytes').sum()), 'files_uid_collision': int((inv['disposition'] == 'uid_collision_across_candidates').sum()), **{f'selection_{k}': int(v) for k, v in cand['selection_status'].value_counts().items()}, 'eligible': int((cand['header_eligibility'] == 'eligible').sum()), 'unresolved': int((cand['header_eligibility'] == 'unresolved').sum())}
    json.dump({'components': comp, 'inventory_digest': inv_digest, 'manifest_sha256': P.sha_obj({'components': comp, 'inventory_digest': inv_digest}), 'flow': fl, 'note': 'digests are RECOMPUTED from files by consumers; this file is a claim, never trusted alone'}, open(out + '/MANIFEST_IDENTITY.json', 'w'), indent=1)
    print(json.dumps(fl))

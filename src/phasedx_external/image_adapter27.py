"""Frozen CT image model adapter v26. Canonical input path (source evidence, not output matching):
   HU = px*slope+intercept (float64) -> recipe preprocess (float32 [3,224,224] window values) -> float16 storage cast -> back to float32 -> ImageNet normalise (float32) -> ResNet18 -> logit.
Locators: lanes/A/code/a_cache_build.py:97-98 (x.astype(np.float16), back=h16.astype(np.float32)); lanes/A/code/a_ct_source.py:75-78 (recipe_f32(hu).astype(np.float16) ... return h16.astype(np.float32));
lanes/B/code/ct_common.py:100-108 (normalize applied to cached values, D-002); lanes/B/code/btrain.py:159,297 (cache record float32 -> C.normalize -> model); lanes/A/code/a_ct_recipe.py:84-88 (recipe_f32 = preprocess(hu).numpy()).
Exam score = max slice logit. Labels never enter this module."""
import os, sys, io, hashlib, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PRECISION_PATH_ID = 'recipe_f32->float16_roundtrip->f32_normalize'
SUPPORTED_TS = {'1.2.840.10008.1.2': 'ImplicitVRLittleEndian', '1.2.840.10008.1.2.1': 'ExplicitVRLittleEndian'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Reject(Exception):

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def load_model(best_model_pt, expected_sha256, threads=1):
    import torch, torchvision
    torch.set_num_threads(threads)
    got = sha256_file(best_model_pt)
    if got != expected_sha256:
        raise RuntimeError('checkpoint sha mismatch %s' % got)
    ck = torch.load(best_model_pt, map_location='cpu', weights_only=False)
    m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Linear(512, 1)
    m.load_state_dict(ck['model'], strict=True)
    m.eval()
    return (m, {'checkpoint_sha256': got, 'selected_epoch': ck.get('epoch')})

def normalize(x):
    import torch
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x.to(torch.float32) - mean) / std

def f16_roundtrip(x):
    import torch
    return x.to(torch.float16).to(torch.float32)

def score_tensors(model, x, batch=32):
    """x: float32 [N,3,224,224] window values AFTER the historical float16 storage cast (cached-record equivalent)."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(normalize(x[i:i + batch]))[:, 0].numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros(0)

def decode_hu(ds, allow_default_rescale=False, decoder_policy='uncompressed_v1'):
    """Validated HU decoding. Raises Reject(reason) for every unsupported case; no guessed transforms."""
    ts = str(getattr(ds.file_meta, 'TransferSyntaxUID', '')) if hasattr(ds, 'file_meta') else ''
    from .dicom_decode import allowed_syntaxes, decode_pixels
    if ts not in allowed_syntaxes(decoder_policy):
        raise Reject('unsupported_transfer_syntax:' + (ts or 'absent'))
    nf = getattr(ds, 'NumberOfFrames', None)
    if nf is not None and int(nf) > 1:
        raise Reject('multiframe_unsupported')
    if int(getattr(ds, 'SamplesPerPixel', 1)) != 1:
        raise Reject('samples_per_pixel_ne_1')
    if str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')) != 'MONOCHROME2':
        raise Reject('photometric_not_monochrome2')
    if 'PixelData' not in ds:
        raise Reject('no_pixel_data')
    have = 'RescaleSlope' in ds and 'RescaleIntercept' in ds
    if not have and (not allow_default_rescale):
        raise Reject('missing_rescale_slope_or_intercept')
    slope = float(ds.RescaleSlope) if 'RescaleSlope' in ds else 1.0
    inter = float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else 0.0
    if not (np.isfinite(slope) and np.isfinite(inter)) or slope == 0:
        raise Reject('invalid_rescale_values')
    try:
        px = decode_pixels(ds, decoder_policy)
    except Exception as e:
        raise Reject('pixel_decode_error:' + type(e).__name__)
    hu = px.astype(np.float64) * slope + inter
    if not np.isfinite(hu).all():
        raise Reject('nonfinite_pixels')
    if hu.ndim != 2:
        raise Reject('not_2d_image')
    return hu

def preprocess_hu(hu):
    """Canonical tensor at the historical cache boundary: float32 [3,224,224] after float16 storage cast."""
    import torch
    from .a_ct_recipe_b2_v7_verbatim import preprocess
    return f16_roundtrip(preprocess(hu).to(torch.float32))

def read_dicom(path_or_bytes, allow_default_rescale=False, decoder_policy='uncompressed_v1'):
    import pydicom
    from pydicom import config as _cfg
    try:
        _cfg.settings.reading_validation_mode = _cfg.IGNORE
    except Exception:
        pass
    try:
        ds = pydicom.dcmread(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes)
    except Exception as e:
        raise Reject('unreadable_dicom:' + type(e).__name__)
    return (ds, decode_hu(ds, allow_default_rescale, decoder_policy))

def preprocess_dicom(path_or_bytes, allow_default_rescale=False, decoder_policy='uncompressed_v1'):
    ds, hu = read_dicom(path_or_bytes, allow_default_rescale, decoder_policy)
    return (preprocess_hu(hu), ds)

def score_exam(model, slice_rows, batch=32, allow_default_rescale=False, decoder_policy='uncompressed_v1'):
    """slice_rows: list of dict(dicom_path, sop). Returns dict(n_listed, n_scored, n_rejected, reasons, score, status). Any rejected slice => status 'incomplete' (score withheld)."""
    import torch
    xs, reasons, seen, nlisted = ([], {}, {}, len(slice_rows))
    for r in slice_rows:
        try:
            src = r['dicom_path']
            if r.get('sha256'):
                with open(src, 'rb') as fh:
                    raw = fh.read()
                if hashlib.sha256(raw).hexdigest() != r['sha256']:
                    raise Reject('content_changed_since_inventory')
                src = raw
            t, ds = preprocess_dicom(src, allow_default_rescale, decoder_policy)
            sop = r.get('sop') or str(getattr(ds, 'SOPInstanceUID', ''))
            hsh = hashlib.sha256(t.numpy().tobytes()).hexdigest()
            if sop and sop in seen:
                if seen[sop] != hsh:
                    raise Reject('duplicate_sop_conflicting_pixels')
                reasons['duplicate_sop_identical_skipped'] = reasons.get('duplicate_sop_identical_skipped', 0) + 1
                continue
            if sop:
                seen[sop] = hsh
            xs.append(t)
        except Reject as e:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
    n_rej = sum((v for k, v in reasons.items() if not k.endswith('_skipped')))
    if not xs:
        return {'n_listed': nlisted, 'n_scored': 0, 'n_rejected': n_rej, 'reasons': reasons, 'score': float('nan'), 'status': 'failed_no_scorable_slice'}
    lg = score_tensors(model, torch.stack(xs), batch)
    if n_rej:
        return {'n_listed': nlisted, 'n_scored': len(xs), 'n_rejected': n_rej, 'reasons': reasons, 'score': float('nan'), 'status': 'incomplete_slices_rejected', 'partial_max_logit_not_a_score': float(lg.max())}
    return {'n_listed': nlisted, 'n_scored': len(xs), 'n_rejected': 0, 'reasons': reasons, 'score': float(lg.max()), 'status': 'scored_complete'}

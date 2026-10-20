"""Lane E development CT-loader candidate (RSNA ICH DICOM) + independent reference read path.

Candidate (what Lane F times):
    read_hu(path)            pydicom.dcmread -> ds.pixel_array -> float64 HU = px * RescaleSlope + RescaleIntercept
    preprocess(hu)           frozen window recipe (lane_E_config_v1.json E5_loader.window_recipe) -> float32 [3,224,224]
    load_example(path)       preprocess(read_hu(path))
Reference (independent; used only for verification):
    read_hu_reference(path, mask_to_bits_stored=True)
        minimal DICOM Part-10 parser written here (no pydicom): file meta (explicit VR LE), dataset in
        implicit VR LE (1.2.840.10008.1.2) or explicit VR LE (1.2.840.10008.1.2.1); raw PixelData ->
        numpy uint16 -> mask to BitsStored -> manual two's-complement sign extension when
        PixelRepresentation == 1 -> float64 * slope + intercept.
Window recipe source: github.com/darraghdog/rsna @521a957 scripts/prepare_meta_dicom.py l.40-72
(2019 RSNA ICH 2nd place, linked from the official RSNA challenge page). Values exactly as published.
No crop. Bilinear antialiased resize to 224x224. No per-image statistics.
"""
import logging
import struct
import warnings

import numpy as np

WINDOWS = ((40, 80), (80, 200), (40, 380))  # (center, width): brain, subdural, source-labelled 'bone'
OUT_HW = (224, 224)
INPUT_SHAPE = (3, 224, 224)

logging.getLogger('pydicom').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='pydicom')


# ----------------------------------------------------------------------------------- candidate path
def read_hu(path):
    import pydicom
    from pydicom import config as _cfg
    try:
        _cfg.settings.reading_validation_mode = _cfg.IGNORE
    except Exception:
        pass
    ds = pydicom.dcmread(path)
    px = ds.pixel_array
    slope = float(ds.RescaleSlope) if 'RescaleSlope' in ds else 1.0
    inter = float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else 0.0
    return px.astype(np.float64) * slope + inter


def window_bounds(center, width):
    return center - width // 2, center + width // 2


def preprocess(hu):
    import torch
    ch = []
    for c, w in WINDOWS:
        lo, hi = window_bounds(c, w)
        ch.append((np.clip(hu, lo, hi) - lo) / float(hi - lo))
    x = torch.from_numpy(np.stack(ch).astype(np.float32))[None]
    y = torch.nn.functional.interpolate(x, size=OUT_HW, mode='bilinear', align_corners=False, antialias=True)
    return y[0]


def load_example(path):
    return preprocess(read_hu(path))


# ----------------------------------------------------------------------------------- reference path
_LONG_VR = {b'OB', b'OW', b'OF', b'SQ', b'UT', b'UN', b'UC', b'UR', b'OD', b'OL', b'OV', b'SV', b'UV'}
_WANT = {(0x0028, 0x0002): 'SamplesPerPixel', (0x0028, 0x0010): 'Rows', (0x0028, 0x0011): 'Columns',
         (0x0028, 0x0100): 'BitsAllocated', (0x0028, 0x0101): 'BitsStored', (0x0028, 0x0102): 'HighBit',
         (0x0028, 0x0103): 'PixelRepresentation', (0x0028, 0x1052): 'RescaleIntercept',
         (0x0028, 0x1053): 'RescaleSlope', (0x7FE0, 0x0010): 'PixelData'}
_US = {'SamplesPerPixel', 'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation'}


def _skip_undefined(buf, i, explicit):
    """Skip an undefined-length sequence value starting at i; return index after the delimiter."""
    while True:
        g, e = struct.unpack_from('<HH', buf, i)
        ln = struct.unpack_from('<I', buf, i + 4)[0]
        i += 8
        if (g, e) == (0xFFFE, 0xE0DD):
            return i
        if (g, e) != (0xFFFE, 0xE000):
            raise ValueError('bad item tag in sequence')
        if ln != 0xFFFFFFFF:
            i += ln
            continue
        while True:  # undefined-length item: parse elements until item delimiter
            g2, e2 = struct.unpack_from('<HH', buf, i)
            if (g2, e2) == (0xFFFE, 0xE00D):
                i += 8
                break
            i, _, _ = _read_elem(buf, i, explicit)


def _read_elem(buf, i, explicit):
    g, e = struct.unpack_from('<HH', buf, i)
    if explicit and g != 0xFFFE:
        vr = buf[i + 4:i + 6]
        if vr in _LONG_VR:
            ln = struct.unpack_from('<I', buf, i + 8)[0]
            j = i + 12
        else:
            ln = struct.unpack_from('<H', buf, i + 6)[0]
            j = i + 8
    else:
        ln = struct.unpack_from('<I', buf, i + 4)[0]
        j = i + 8
    if ln == 0xFFFFFFFF:
        if (g, e) == (0x7FE0, 0x0010):
            raise ValueError('encapsulated PixelData not supported by reference path')
        return _skip_undefined(buf, j, explicit), (g, e), None
    return j + ln, (g, e), buf[j:j + ln]


def parse_reference(path):
    buf = open(path, 'rb').read()
    if buf[128:132] != b'DICM':
        raise ValueError('no DICM prefix')
    i = 132
    ts = None
    while i < len(buf):
        g = struct.unpack_from('<H', buf, i)[0]
        if g != 0x0002:
            break
        i, tag, val = _read_elem(buf, i, True)
        if tag == (0x0002, 0x0010):
            ts = val.decode('ascii').strip('\x00 ')
    if ts == '1.2.840.10008.1.2':
        explicit = False
    elif ts == '1.2.840.10008.1.2.1':
        explicit = True
    else:
        raise ValueError(f'transfer syntax {ts} not supported by reference path')
    out = {'TransferSyntaxUID': ts}
    while i < len(buf):
        i, tag, val = _read_elem(buf, i, explicit)
        name = _WANT.get(tag)
        if name is None or val is None:
            continue
        if name in _US:
            out[name] = struct.unpack('<H', val[:2])[0]
        elif name == 'PixelData':
            out[name] = val
        else:
            out[name] = float(val.decode('ascii').strip('\x00 ').split('\\')[0])
    return out


def read_hu_reference(path, mask_to_bits_stored=True):
    h = parse_reference(path)
    if h.get('BitsAllocated') != 16 or h.get('SamplesPerPixel', 1) != 1:
        raise ValueError('reference path supports 16-bit single-sample data only')
    rows, cols, bs, hb, pr = h['Rows'], h['Columns'], h['BitsStored'], h['HighBit'], h['PixelRepresentation']
    raw = np.frombuffer(h['PixelData'], dtype='<u2', count=rows * cols).reshape(rows, cols).astype(np.int64)
    if mask_to_bits_stored:
        shift = hb + 1 - bs
        v = (raw >> shift) & ((1 << bs) - 1)
        if pr == 1:
            v = np.where(v >= (1 << (bs - 1)), v - (1 << bs), v)
    else:
        v = raw.astype(np.uint16).view(np.int16).astype(np.int64) if pr == 1 else raw
    slope = h.get('RescaleSlope', 1.0)
    inter = h.get('RescaleIntercept', 0.0)
    meta = {k: h.get(k) for k in ('Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit',
                                  'PixelRepresentation', 'RescaleSlope', 'RescaleIntercept', 'TransferSyntaxUID')}
    meta['raw_min'] = int(raw.min())
    meta['raw_max'] = int(raw.max())
    meta['n_raw_above_bits_stored'] = int((raw >= (1 << bs)).sum())
    return v.astype(np.float64) * slope + inter, meta

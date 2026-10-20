"""Explicit, lossless-only DICOM pixel decoding; no changes to input files."""
import numpy as np

UNCOMPRESSED = frozenset(("1.2.840.10008.1.2", "1.2.840.10008.1.2.1"))
JPEG_LOSSLESS_SV1 = "1.2.840.10008.1.2.4.70"
POLICIES = ("uncompressed_v1", "lossless_sv1_pylibjpeg_v1")


def allowed_syntaxes(policy):
    if policy not in POLICIES:
        raise ValueError("unknown decoder policy: " + str(policy))
    return UNCOMPRESSED | ({JPEG_LOSSLESS_SV1} if policy == POLICIES[1] else set())


def decode_pixels(ds, policy="uncompressed_v1"):
    ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
    if ts not in allowed_syntaxes(policy):
        raise ValueError("unsupported_transfer_syntax:" + (ts or "absent"))
    if str(getattr(ds, "LossyImageCompression", "00")) == "01":
        raise ValueError("declared_lossy_image")
    if ts == JPEG_LOSSLESS_SV1:
        from pydicom.pixels import pixel_array
        # Pin a backend; never accept whichever optional decoder happens to load first.
        return pixel_array(ds, decoding_plugin="pylibjpeg", raw=True)
    return ds.pixel_array


def compare_lossless_backends(ds):
    """Development diagnostic, not a model prediction or clinical qualification."""
    from pydicom.pixels import pixel_array
    ts = str(ds.file_meta.TransferSyntaxUID)
    if ts != JPEG_LOSSLESS_SV1:
        raise ValueError("diagnostic requires JPEG Lossless SV1")
    a = pixel_array(ds, decoding_plugin="pylibjpeg", raw=True)
    b = pixel_array(ds, decoding_plugin="gdcm", raw=True)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "same_dtype": a.dtype == b.dtype,
            "exact_equal": bool(a.shape == b.shape and np.array_equal(a, b)),
            "max_absolute_difference": int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))}

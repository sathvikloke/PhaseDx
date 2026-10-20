"""Known synthetic pixel truth; no cohort images, inference or random draws."""
from pathlib import Path

import numpy as np
import pytest

gdcm = pytest.importorskip("gdcm")
pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from phasedx_external.dicom_decode import compare_lossless_backends, decode_pixels


@pytest.mark.parametrize("signed,bits", [(False, 12), (False, 16), (True, 12), (True, 16)])
def test_lossless_pixels_match_known_truth(tmp_path, signed, bits):
    values = np.arange(4096, dtype=np.int64).reshape(64, 64)
    if bits == 16:
        values *= 16
    if signed:
        values -= 2 ** (bits - 1)
    truth = values.astype(np.int16 if signed else np.uint16)
    src = tmp_path / "native.dcm"
    dst = tmp_path / "lossless.dcm"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(src), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PatientName = "SYNTHETIC^TEST"
    ds.PatientID = "SYNTHETIC"
    ds.Rows, ds.Columns = truth.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = int(signed)
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.PixelData = truth.tobytes()
    ds.save_as(src, enforce_file_format=True)
    convert = gdcm.FileChangeTransferSyntax()
    convert.SetInputFileName(str(src))
    convert.SetOutputFileName(str(dst))
    convert.SetTransferSyntax(gdcm.TransferSyntax(gdcm.TransferSyntax.JPEGLosslessProcess14_1))
    assert convert.Change()
    encoded = pydicom.dcmread(dst)
    assert str(encoded.file_meta.TransferSyntaxUID) == "1.2.840.10008.1.2.4.70"
    with pytest.raises(ValueError, match="unsupported_transfer_syntax"):
        decode_pixels(encoded)
    pixels = decode_pixels(encoded, "lossless_sv1_pylibjpeg_v1")
    assert np.array_equal(pixels, truth)
    result = compare_lossless_backends(encoded)
    assert result["exact_equal"] and result["same_dtype"]
    assert result["max_absolute_difference"] == 0
    assert np.array_equal(pixels.astype(np.float64) - 1024, truth.astype(np.float64) - 1024)


def test_lossy_policy_refuses():
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.50"
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0"*128)
    with pytest.raises(ValueError, match="unsupported_transfer_syntax"):
        decode_pixels(ds, "lossless_sv1_pylibjpeg_v1")

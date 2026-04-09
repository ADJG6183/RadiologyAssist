"""
Unit tests for DICOMProcessor.

These tests use synthetic pydicom Datasets constructed in memory — no real
.dcm files needed.  This is the same pattern as using MockLLMClient: we
control the input completely so every edge case is deterministic.

Why build fake Datasets instead of committing a real DICOM file?
  - Real DICOM files contain PHI (patient data) — never commit those.
  - Synthetic datasets are tiny and cover exactly the cases we care about.
  - pydicom lets you set any tag directly on a Dataset object.

Key concepts tested:
  - RescaleSlope/Intercept → Hounsfield Unit conversion (_apply_modality_lut)
  - Clinical windowing → 8-bit PNG output (to_png_bytes)
  - Slice selection across a multi-slice volume (get_representative_slices)
  - Metadata extraction (extract_metadata)
"""

import io

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper — build a DICOMProcessor from a numpy array (no filesystem needed)
# ---------------------------------------------------------------------------

def _processor_from_array(pixel_array, slope=1.0, intercept=-1024.0):
    """
    Build a DICOMProcessor directly from a numpy array, bypassing filesystem.

    `pixel_array` is a read-only *property* on pydicom Dataset — it decodes
    the raw PixelData bytes on-the-fly and cannot be assigned.  Instead we
    construct the Dataset for metadata only and set proc._pixel_array directly
    (after applying the LUT manually), which is all the processor needs.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    from app.services.dicom import DICOMProcessor

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Modality = "CT"
    ds.BodyPartExamined = "CHEST"
    ds.SliceThickness = 1.5
    ds.KVP = 120.0
    ds.PatientPosition = "HFS"
    ds.Rows = int(pixel_array.shape[-2])
    ds.Columns = int(pixel_array.shape[-1])
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept

    proc = DICOMProcessor()
    proc._ds = ds
    # Apply LUT manually — same formula as _apply_modality_lut()
    proc._pixel_array = pixel_array.astype(float) * slope + intercept
    return proc


# ---------------------------------------------------------------------------
# Test 1: metadata extraction
# ---------------------------------------------------------------------------

def test_extract_metadata_returns_expected_keys():
    """
    extract_metadata() should return a dict with all documented keys.
    We check both that the keys exist and that the values are the right type.
    """
    pixel_array = np.full((64, 64), fill_value=512, dtype=np.int16)
    proc = _processor_from_array(pixel_array)
    meta = proc.extract_metadata()

    assert "modality" in meta
    assert "body_part_examined" in meta
    assert "slice_thickness" in meta
    assert "num_slices" in meta

    assert meta["modality"] == "CT"
    assert meta["body_part_examined"] == "CHEST"
    assert isinstance(meta["slice_thickness"], float)
    assert meta["num_slices"] == 1  # single-slice array


# ---------------------------------------------------------------------------
# Test 2: slice selection — single slice
# ---------------------------------------------------------------------------

def test_get_representative_slices_single_slice_returns_zero():
    """
    A single-slice DICOM (a plain X-ray, for instance) should always return [0].
    There is nothing to distribute — the one slice IS the representative slice.
    """
    pixel_array = np.full((64, 64), fill_value=512, dtype=np.int16)
    proc = _processor_from_array(pixel_array)
    slices = proc.get_representative_slices(max_slices=5)
    assert slices == [0]


# ---------------------------------------------------------------------------
# Test 3: slice selection — multi-slice volume
# ---------------------------------------------------------------------------

def test_get_representative_slices_distributes_across_volume():
    """
    For a 100-slice volume with max_slices=5, we expect 5 indices spread
    across the volume — specifically the midpoint of each 20-slice segment.

    100 slices / 5 segments = 20 slices per segment
    Midpoints: 10, 30, 50, 70, 90

    This ensures Claude sees anatomy from top, upper-mid, mid, lower-mid,
    and bottom of the scan — not just the first 5 slices.
    """
    # Build a 100-slice volume where adjacent slices differ enough
    # (mean pixel difference ≥ 10) to avoid deduplication
    pixel_array = np.zeros((100, 16, 16), dtype=np.int16)
    for i in range(100):
        pixel_array[i, :, :] = i * 20  # slices differ by 20 * slope + intercept

    proc = _processor_from_array(pixel_array)
    slices = proc.get_representative_slices(max_slices=5)

    assert len(slices) == 5
    # Indices should be spread out — none should be in the same quartile
    assert slices[0] < 25
    assert slices[-1] > 74


# ---------------------------------------------------------------------------
# Test 4: windowed PNG output
# ---------------------------------------------------------------------------

def test_to_png_bytes_produces_valid_png():
    """
    to_png_bytes() should return bytes that start with the PNG magic header
    (\\x89PNG).  We don't decode the full image — just check it's valid PNG.

    Using lung window (centre=-600, width=1500).
    Our pixel value 512 with slope=1/intercept=-1024 → HU = -512.
    HU -512 is within the lung window [-1350, 150], so it will map to a
    non-zero grey value — not clipped to black.
    """
    pixel_array = np.full((64, 64), fill_value=512, dtype=np.int16)
    proc = _processor_from_array(pixel_array)
    png_bytes = proc.to_png_bytes(slice_idx=0, window_center=-600, window_width=1500)

    # PNG files always start with this 8-byte signature
    assert png_bytes[:4] == b'\x89PNG'
    assert len(png_bytes) > 100  # should be a non-trivial image

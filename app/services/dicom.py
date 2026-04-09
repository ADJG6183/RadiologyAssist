"""
DICOM image processing service.

Why this module exists
----------------------
CT scans store pixel data as 12–16 bit signed integers called Hounsfield Units
(HU).  HU values represent tissue density: air is -1000 HU, water is 0 HU,
bone is +1000 HU.

If you naively rescale a full CT scan to 8-bit (0–255), lung tissue (around
-600 HU) and soft tissue (around +40 HU) collapse into indistinguishably dark
grey.  Clinical radiologists solve this with *windowing*: they pick a HU range
they care about and stretch it to fill the full 0–255 display range.

Two standard windows for chest CT:
  Lung window      centre=-600 HU, width=1500  — shows airways, lung texture
  Mediastinal window centre=+40  HU, width=400  — shows heart, vessels, masses

This module:
  1. Loads a DICOM file and converts pixels to HU
  2. Applies clinical windowing → 8-bit PNG bytes ready for Claude vision
  3. Selects a representative subset of slices from a multi-slice volume
  4. Extracts useful metadata (modality, body part, slice thickness, etc.)

Zero ML training required — Claude's vision API interprets the images
directly, the same way a radiologist would read them on a workstation.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np


class DICOMProcessor:
    """
    Load a DICOM file (or dataset) and produce windowed PNG images.

    Usage::

        proc = DICOMProcessor().load("/path/to/scan.dcm")
        metadata = proc.extract_metadata()
        slices = proc.get_representative_slices(max_slices=5)
        for idx in slices:
            lung_png   = proc.to_png_bytes(idx, window_center=-600, window_width=1500)
            medial_png = proc.to_png_bytes(idx, window_center=40,   window_width=400)

    For unit testing, use load_from_dataset() to bypass the filesystem.
    """

    def __init__(self):
        self._ds = None          # pydicom Dataset
        self._pixel_array = None  # numpy array of HU values, shape (slices, H, W) or (H, W)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> "DICOMProcessor":
        """Load a DICOM file from the filesystem."""
        import pydicom  # lazy import — only needed when actually processing DICOM
        self._ds = pydicom.dcmread(path)
        self._pixel_array = self._apply_modality_lut(self._ds.pixel_array)
        return self

    def load_from_dataset(self, ds) -> "DICOMProcessor":
        """
        Load from an already-parsed pydicom Dataset.
        Used in unit tests to avoid touching the filesystem.
        """
        self._ds = ds
        self._pixel_array = self._apply_modality_lut(ds.pixel_array)
        return self

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def extract_metadata(self) -> dict:
        """
        Return a flat dict of the DICOM tags most useful for prompt context.

        All values are cast to str/float/None so the caller can safely json.dumps()
        the result without worrying about pydicom-specific types.
        """
        ds = self._ds
        return {
            "modality":          getattr(ds, "Modality", None),
            "body_part_examined": getattr(ds, "BodyPartExamined", None),
            "slice_thickness":   _safe_float(getattr(ds, "SliceThickness", None)),
            "kvp":               _safe_float(getattr(ds, "KVP", None)),
            "patient_position":  getattr(ds, "PatientPosition", None),
            "rows":              int(ds.Rows) if hasattr(ds, "Rows") else None,
            "columns":           int(ds.Columns) if hasattr(ds, "Columns") else None,
            "num_slices":        self._num_slices(),
        }

    # ------------------------------------------------------------------
    # Slice selection
    # ------------------------------------------------------------------

    def get_representative_slices(self, max_slices: int = 5) -> list:
        """
        Return up to max_slices slice indices spread evenly across the volume.

        For a 500-slice scan with max_slices=5:
          → divide into 5 equal segments of 100 slices each
          → return the midpoint of each segment: [50, 150, 250, 350, 450]

        Near-duplicate deduplication: if two adjacent candidates have a mean
        pixel difference < 10 HU (nearly identical anatomy), drop the second
        and take the next candidate.  This avoids sending Claude five images
        of the same cross-section.

        For a single-slice file the result is always [0].
        """
        n = self._num_slices()
        if n <= 1:
            return [0]

        # Build initial candidates at segment midpoints
        segment_size = n / max_slices
        candidates = [int((i + 0.5) * segment_size) for i in range(max_slices)]

        # Deduplicate near-identical adjacent slices
        selected = [candidates[0]]
        for idx in candidates[1:]:
            prev_slice = self._get_slice(selected[-1]).astype(float)
            curr_slice = self._get_slice(idx).astype(float)
            if abs(float(np.mean(curr_slice - prev_slice))) >= 10:
                selected.append(idx)

        return selected

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def to_png_bytes(
        self,
        slice_idx: int = 0,
        window_center: int = -600,
        window_width: int = 1500,
    ) -> bytes:
        """
        Apply clinical windowing to one slice and return PNG bytes.

        Windowing formula:
          lower = centre - width/2
          upper = centre + width/2
          clip HU values to [lower, upper]
          stretch to [0, 255]

        Returns raw bytes suitable for base64-encoding and sending to
        Claude's vision API.
        """
        from PIL import Image  # lazy import — only needed here

        hu_slice = self._get_slice(slice_idx).astype(float)

        lower = window_center - window_width / 2
        upper = window_center + window_width / 2

        clipped = np.clip(hu_slice, lower, upper)
        scaled = ((clipped - lower) / window_width * 255).astype(np.uint8)

        img = Image.fromarray(scaled)  # numpy uint8 array → greyscale PIL image
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_modality_lut(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Convert raw stored pixel values to Hounsfield Units using the
        RescaleSlope and RescaleIntercept DICOM tags.

        HU = pixel_value × RescaleSlope + RescaleIntercept

        For most CT scanners: slope=1, intercept=-1024.
        If tags are absent (some older scanners omit them) we return the
        array unchanged.
        """
        slope = float(getattr(self._ds, "RescaleSlope", 1))
        intercept = float(getattr(self._ds, "RescaleIntercept", 0))
        return pixel_array.astype(float) * slope + intercept

    def _num_slices(self) -> int:
        """Number of slices in the loaded dataset."""
        if self._pixel_array is None:
            return 0
        if self._pixel_array.ndim == 3:
            return self._pixel_array.shape[0]
        return 1  # single-slice DICOM file

    def _get_slice(self, idx: int) -> np.ndarray:
        """Return the 2D array for slice idx (works for both 2D and 3D arrays)."""
        if self._pixel_array.ndim == 3:
            return self._pixel_array[idx]
        return self._pixel_array  # single-slice — idx is ignored


def _safe_float(value) -> Optional[float]:
    """Cast a DICOM numeric tag to float, returning None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

"""
src/api/validation.py
=====================
Input validation and sanitization for LeafMind API.
Handles:
  - Image decoding with PIL (JPEG, PNG, WebP)
  - Size and minimum dimension checks
  - EXIF GPS metadata stripping
  - Crop validation and alias resolution
"""
from __future__ import annotations

import io
import logging
from typing import Optional, Tuple
from fastapi import HTTPException, status
from PIL import Image

logger = logging.getLogger("leafmind.api.validation")

# Single source of truth for supported crops from classifier taxonomy
SUPPORTED_CROPS = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]

SINGLE_CLASS_CROPS = [
    "Blueberry",
    "Cherry",
    "Peach",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
]

MULTI_CLASS_CROPS = [
    "Apple",
    "Corn",
    "Grape",
    "Pepper",
    "Potato",
    "Tomato",
]

ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")


def resolve_crop(crop_input: str) -> str:
    """
    Validate and normalize crop name to canonical form.
    Raises HTTPException(422) if crop is unknown or empty.
    """
    c = crop_input.strip()
    if not c:
        supported_str = ", ".join(SUPPORTED_CROPS)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Crop name cannot be empty. Supported crops: {supported_str}",
        )

    # Aliases
    c_lower = c.lower()
    if c_lower in ("corn", "corn_(maize)", "corn (maize)", "maize"):
        return "Corn"
    if c_lower in ("pepper", "pepper, bell", "pepper,_bell", "pepper_bell", "bell_pepper", "bell pepper"):
        return "Pepper"

    for supported in SUPPORTED_CROPS:
        if c_lower == supported.lower():
            return supported

    supported_str = ", ".join(SUPPORTED_CROPS)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Unknown crop '{crop_input}'. Supported crops: {supported_str}",
    )


def validate_and_sanitize_image(
    raw_bytes: bytes,
    max_mb: float = 10.0,
    min_dimension: int = 64,
) -> Tuple[bytes, int, int, str]:
    """
    Validate uploaded image bytes and strip EXIF GPS metadata.

    Returns:
        (sanitized_jpeg_bytes, width, height, format_name)
    Raises:
        HTTPException(422) if validation fails.
    """
    # 1. Size check
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Image file is too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {max_mb:.0f} MB. Please upload a smaller leaf photo."
            ),
        )

    # 2. PIL decode check
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()  # Force load pixel data to verify image integrity
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Uploaded file is not a valid or readable image. Please provide a clear, "
                "uncorrupted JPEG, PNG, or WebP photo of a crop leaf. Capture tip: Take photos "
                "in bright, indirect daylight and hold your camera steady."
            ),
        ) from None

    # 3. Format check
    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported image format '{fmt}'. Please upload a photo in "
                f"JPEG, PNG, or WebP format."
            ),
        )

    # 4. Dimension check
    w, h = img.size
    if w < min_dimension or h < min_dimension:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Image resolution is too small ({w}x{h} px). Leaf images must be at least "
                f"{min_dimension}x{min_dimension} pixels. Please capture a closer photo of the leaf."
            ),
        )

    # 5. Strip EXIF GPS and metadata by re-encoding clean pixel buffer to JPEG
    try:
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))
        if clean_img.mode != "RGB":
            clean_img = clean_img.convert("RGB")
        buf = io.BytesIO()
        clean_img.save(buf, format="JPEG", quality=95)
        sanitized_bytes = buf.getvalue()
    except Exception:
        # Fallback if raw image conversion encounters an issue
        buf = io.BytesIO()
        rgb_img = img.convert("RGB")
        rgb_img.save(buf, format="JPEG", quality=95)
        sanitized_bytes = buf.getvalue()

    return sanitized_bytes, w, h, fmt

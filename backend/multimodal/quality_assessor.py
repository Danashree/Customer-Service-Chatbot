"""
Task 2 Phase 2 – OCR Quality Assessor

Evaluates the quality of OCR output or input images to determine whether
extracted text is reliable enough for field parsing.

Quality tiers:
  - good:       High-confidence extraction, no clarification needed
  - acceptable: Usable text but some uncertainty; proceed with caution
  - poor:       Text is likely unreliable; request clearer image from user
"""

import re
import io
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from PIL import Image, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Thresholds
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_TEXT_LENGTH_GOOD = 50        # chars for "good" quality
MIN_TEXT_LENGTH_ACCEPTABLE = 15  # chars for "acceptable" quality
MAX_NON_ALPHA_RATIO = 0.70       # above this → garbage OCR output
MIN_IMAGE_VARIANCE = 100.0       # std-dev variance; below → blank/uniform image


@dataclass
class QualityResult:
    """Result of a quality assessment pass."""
    quality: str                        # "good" | "acceptable" | "poor"
    clarification_required: bool
    quality_score: float                # 0.0–1.0 numeric score
    quality_notes: List[str] = field(default_factory=list)
    user_message: Optional[str] = None


class QualityAssessor:
    """
    Assesses OCR/extraction quality for multimodal evidence files.
    Works on both raw OCR text and (optionally) PIL Image objects for
    pre-OCR image quality checks.
    """

    # ─── Image-level assessment ────────────────────────────────────────────

    def assess_image(self, image_bytes: bytes) -> QualityResult:
        """
        Check image resolution and pixel variance before OCR.
        Returns a QualityResult flagging images that are too small or blank.
        """
        notes: List[str] = []
        deductions = 0.0

        if not PIL_AVAILABLE:
            return QualityResult(
                quality="acceptable",
                clarification_required=False,
                quality_score=0.6,
                quality_notes=["PIL not available for pre-OCR image check"],
            )

        try:
            bio = io.BytesIO(image_bytes)
            img = Image.open(bio).convert("L")  # grayscale for variance
            width, height = img.size

            # Resolution check
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                notes.append(
                    f"Image resolution too low ({width}x{height}). "
                    f"Minimum is {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}."
                )
                deductions += 0.5

            # Blur / blank check via pixel variance
            stat = ImageStat.Stat(img)
            variance = stat.stddev[0] if stat.stddev else 0.0
            if variance < MIN_IMAGE_VARIANCE:
                notes.append(
                    f"Image appears blank or extremely uniform "
                    f"(pixel std-dev: {variance:.1f}). Likely unreadable."
                )
                deductions += 0.4

        except Exception as e:
            notes.append(f"Could not open image for quality check: {e}")
            deductions += 0.3

        score = max(0.0, 1.0 - deductions)
        quality, clarification_required = self._tier(score)
        user_message = self._user_message(quality, notes)

        return QualityResult(
            quality=quality,
            clarification_required=clarification_required,
            quality_score=round(score, 3),
            quality_notes=notes,
            user_message=user_message,
        )

    # ─── Text-level assessment ─────────────────────────────────────────────

    def assess_text(self, ocr_text: str, extraction_method: str = "ocr") -> QualityResult:
        """
        Assess the quality of already-extracted OCR text.
        Checks length, character composition, and overall readability.
        """
        notes: List[str] = []
        deductions = 0.0

        text = (ocr_text or "").strip()
        text_len = len(text)

        # Length check
        if text_len == 0:
            notes.append("No text was extracted from the document.")
            deductions += 1.0
        elif text_len < MIN_TEXT_LENGTH_ACCEPTABLE:
            notes.append(
                f"Very short extracted text ({text_len} chars). "
                "Document may be blank, image-only with poor OCR, or unreadable."
            )
            deductions += 0.5
        elif text_len < MIN_TEXT_LENGTH_GOOD:
            notes.append(
                f"Limited extracted text ({text_len} chars). "
                "Some fields may not be detectable."
            )
            deductions += 0.2

        # Non-alphabetic character ratio (garbage detection)
        if text_len > 0:
            alpha_count = sum(1 for c in text if c.isalpha())
            non_alpha_ratio = 1.0 - (alpha_count / text_len)
            if non_alpha_ratio > MAX_NON_ALPHA_RATIO:
                notes.append(
                    f"High proportion of non-alphabetic characters "
                    f"({non_alpha_ratio:.0%}). OCR output may be garbled."
                )
                deductions += 0.35

        # PDF text extraction is generally higher quality than OCR
        if extraction_method == "pdf_text" and text_len >= MIN_TEXT_LENGTH_ACCEPTABLE:
            deductions = max(0.0, deductions - 0.1)  # small bonus

        score = max(0.0, 1.0 - deductions)
        quality, clarification_required = self._tier(score)
        user_message = self._user_message(quality, notes)

        return QualityResult(
            quality=quality,
            clarification_required=clarification_required,
            quality_score=round(score, 3),
            quality_notes=notes,
            user_message=user_message,
        )

    # ─── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _tier(score: float):
        """Convert numeric score to (quality_tier, clarification_required)."""
        if score >= 0.75:
            return "good", False
        elif score >= 0.45:
            return "acceptable", False
        else:
            return "poor", True

    @staticmethod
    def _user_message(quality: str, notes: List[str]) -> Optional[str]:
        if quality == "poor":
            base = (
                "The uploaded document could not be read reliably. "
                "Please upload a clearer, higher-resolution image or a text-based PDF. "
            )
            if notes:
                base += "Details: " + " | ".join(notes)
            return base
        return None

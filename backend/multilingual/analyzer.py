"""
Task 6 Phase 1 — Multilingual Message Analyzer
===============================================
Unified entry point that coordinates entity preservation, spelling normalization,
and language detection for customer service interactions.
"""
from __future__ import annotations

from typing import Optional

from .config import MultilingualConfig
from .entity_preserver import EntityPreserver
from .language_detector import LanguageDetector
from .models import MultilingualAnalysisResult
from .spelling_normalizer import SpellingNormalizer


def analyze_multilingual_message(
    text: str,
    config: Optional[MultilingualConfig] = None,
) -> MultilingualAnalysisResult:
    """
    Performs complete Phase 1 multilingual analysis:
      1. Protects critical entities (Names, Order IDs, Dates, Product Codes)
      2. Performs safe spelling normalization on non-entity terms
      3. Restores exact entity values into normalized analysis text
      4. Detects language, mixed-language presence, and transliteration
      5. Enforces confidence thresholds and identifies clarification requirements
    """
    cfg = config or MultilingualConfig.from_env()

    # 1. Shield entities with placeholders
    preservation = EntityPreserver.protect(text)

    # 2. Normalize spelling on protected text (placeholders remain untouched)
    normalized_protected = SpellingNormalizer.normalize_text(
        preservation.protected_text,
        enabled=cfg.spelling_normalization_enabled,
    )

    # 3. Restore exact original entities into normalized text
    normalized_text = EntityPreserver.restore(
        normalized_protected,
        preservation.entities,
    )

    # 4. Perform language detection
    detector = LanguageDetector(config=cfg)
    lang_result = detector.detect(normalized_text)

    # 5. Determine clarification requirement
    requires_clarification = lang_result.requires_clarification
    clarification_reason = lang_result.reason

    return MultilingualAnalysisResult(
        original_text=text,
        normalized_text=normalized_text,
        language=lang_result,
        entities=preservation,
        requires_clarification=requires_clarification,
        clarification_reason=clarification_reason,
    )

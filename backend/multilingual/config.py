"""
Task 6 Phase 1 — Multilingual Configuration
============================================
Central configuration for supported languages, confidence thresholds,
and text normalization flags.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


DEFAULT_SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
}


class MultilingualConfig(BaseModel):
    """Configuration settings for multilingual detection and processing."""
    supported_languages: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SUPPORTED_LANGUAGES),
        description="Mapping of ISO 639-1 language code to human-readable name."
    )
    default_language: str = Field(
        default="en",
        description="Default fallback language code."
    )
    language_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum confidence required to accept detected language without clarification."
    )
    transliteration_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum confidence to accept transliterated language identification."
    )
    spelling_normalization_enabled: bool = Field(
        default=True,
        description="Whether to normalize common non-entity typos in customer messages."
    )

    context_window: int = Field(
        default=10,
        description="Number of recent messages to retain in active conversation context."
    )
    intent_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum confidence to accept detected customer intent without clarification."
    )
    session_inactivity_minutes: int = Field(
        default=30,
        description="Inactivity timeout in minutes before an active session expires."
    )
    session_restore_hours: int = Field(
        default=24,
        description="Maximum hours after session expiry during which the summary can be restored."
    )
    summary_enabled: bool = Field(
        default=True,
        description="Whether to generate and retain conversation summaries upon session expiry."
    )

    @classmethod
    def from_env(cls) -> "MultilingualConfig":
        """Instantiate configuration from environment variables with fallback defaults."""
        supported_env = os.getenv("TASK6_SUPPORTED_LANGUAGES")
        if supported_env:
            langs = [lang.strip().lower() for lang in supported_env.split(",") if lang.strip()]
            supported = {code: DEFAULT_SUPPORTED_LANGUAGES.get(code, code.capitalize()) for code in langs}
        else:
            supported = dict(DEFAULT_SUPPORTED_LANGUAGES)

        default_lang = os.getenv("TASK6_DEFAULT_LANGUAGE", "en").strip().lower()

        # Support both TASK6_CONFIDENCE_THRESHOLD and TASK6_LANGUAGE_CONFIDENCE_THRESHOLD
        lang_thresh_str = os.getenv("TASK6_LANGUAGE_CONFIDENCE_THRESHOLD") or os.getenv("TASK6_CONFIDENCE_THRESHOLD")
        lang_thresh = float(lang_thresh_str) if lang_thresh_str is not None else 0.70

        trans_thresh_str = os.getenv("TASK6_TRANSLITERATION_CONFIDENCE_THRESHOLD")
        trans_thresh = float(trans_thresh_str) if trans_thresh_str is not None else 0.70

        spell_norm_str = os.getenv("TASK6_SPELLING_NORMALIZATION_ENABLED")
        spell_norm = spell_norm_str.lower() in ("true", "1", "yes") if spell_norm_str is not None else True

        context_win_str = os.getenv("TASK6_CONTEXT_WINDOW")
        context_win = int(context_win_str) if context_win_str is not None else 10

        intent_thresh_str = os.getenv("TASK6_INTENT_CONFIDENCE_THRESHOLD")
        intent_thresh = float(intent_thresh_str) if intent_thresh_str is not None else 0.70

        inactivity_str = os.getenv("TASK6_SESSION_INACTIVITY_MINUTES")
        inactivity_min = int(inactivity_str) if inactivity_str is not None else 30

        restore_str = os.getenv("TASK6_SESSION_RESTORE_HOURS")
        restore_hrs = int(restore_str) if restore_str is not None else 24

        summary_str = os.getenv("TASK6_SUMMARY_ENABLED")
        summary_en = summary_str.lower() in ("true", "1", "yes") if summary_str is not None else True

        return cls(
            supported_languages=supported,
            default_language=default_lang,
            language_confidence_threshold=lang_thresh,
            transliteration_confidence_threshold=trans_thresh,
            spelling_normalization_enabled=spell_norm,
            context_window=context_win,
            intent_confidence_threshold=intent_thresh,
            session_inactivity_minutes=inactivity_min,
            session_restore_hours=restore_hrs,
            summary_enabled=summary_en,
        )

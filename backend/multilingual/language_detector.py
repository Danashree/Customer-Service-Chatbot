"""
Task 6 Phase 1 — Language Detector
===================================
Detects languages for incoming customer service messages with support for:
  - English (en), Tamil (ta), Hindi (hi), Malayalam (ml)
  - Mixed-language messages (e.g. Tamil + English, Hindi + English)
  - Transliterated inputs in Roman script
  - Low-confidence and unsupported language identification
  - Independent, stateless per-message analysis for seamless language switching
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .config import MultilingualConfig
from .models import LanguageDetectionResult


# ---------------------------------------------------------------------------
# Script Unicode Ranges
# ---------------------------------------------------------------------------
TAMIL_SCRIPT_REGEX = re.compile(r"[\u0B80-\u0BFF]")
HINDI_SCRIPT_REGEX = re.compile(r"[\u0900-\u097F]")
MALAYALAM_SCRIPT_REGEX = re.compile(r"[\u0D00-\u0D7F]")
LATIN_SCRIPT_REGEX = re.compile(r"[A-Za-z]")

# Non-supported script indicators
UNSUPPORTED_SCRIPT_REGEX = re.compile(
    r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\u0600-\u06FF\u0590-\u05FF\u0E00-\u0E7F]"
)

# Common European / non-English accented Latin characters indicating unsupported language
UNSUPPORTED_LATIN_CHARS = re.compile(
    r"(?i)\b(?:bonjour|merci|comment|vous|est|pourquoi|hola|gracias|dónde|por\s+favor|hallo|danke|bitte|buongiorno)\b|[àâçéèêëîïôûùüÿœæñ¿¡]"
)


# ---------------------------------------------------------------------------
# Transliteration Vocabularies (Roman / Latin Script)
# ---------------------------------------------------------------------------
TAMIL_TRANSLIT_WORDS: Set[str] = {
    "en", "enga", "iruku", "irukku", "enna", "eppadi", "ungal", "nandri",
    "romba", "solla", "mudiyuma", "vanakkam", "thavaru", "panam", "thiruppi",
    "kodunga", "kedaikkum", "illa", "illai", "theriyala", "kekkuren",
    "solreenga", "aachu", "paatha", "thanga", "eppo", "varum", "solunga",
    "aam", "illaiye", "edhuvum",
}

HINDI_TRANSLIT_WORDS: Set[str] = {
    "mera", "meri", "mere", "kaha", "kahan", "hai", "hain", "kya", "mujhe",
    "kripya", "kaise", "kab", "shukriya", "namaste", "batao", "bataiye",
    "paisa", "wapas", "chahiye", "nahi", "nahin", "hoga", "raha", "rahi",
    "karo", "kijiye", "kyun", "kaun", "humara", "aapka", "kariye",
}

MALAYALAM_TRANSLIT_WORDS: Set[str] = {
    "ente", "evide", "aanu", "aan", "enthannu", "enthaanu", "namaskaram",
    "engane", "nanni", "undoo", "ariyikanam", "thirike", "venam", "parayumo",
    "kittiyilla", "kittumo", "undo", "illa", "kaanan", "eppozhum", "eppo",
    "entha", "varumo", "cheyyuka",
}

COMMON_ENGLISH_WORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "order", "status", "refund",
    "cancel", "course", "bootcamp", "certificate", "payment", "invoice",
    "account", "help", "support", "please", "access", "login", "password",
    "where", "why", "hi", "hello", "hey", "tell", "issue", "problem",
}


class LanguageDetector:
    """Stateless language detector with multi-script, mixed-language, and transliteration awareness."""

    def __init__(self, config: Optional[MultilingualConfig] = None):
        self.config = config or MultilingualConfig.from_env()

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Analyzes a single message and returns structured language metadata.
        Stateless: does not remember or bias future calls.
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                primary_language=self.config.default_language,
                confidence=0.50,
                detected_languages=[self.config.default_language],
                is_mixed=False,
                is_transliterated=False,
                requires_clarification=True,
                reason="LOW_LANGUAGE_CONFIDENCE",
            )

        clean_text = text.strip()

        # 1. Check for unsupported foreign scripts or common European languages
        if UNSUPPORTED_SCRIPT_REGEX.search(clean_text) or UNSUPPORTED_LATIN_CHARS.search(clean_text):
            return LanguageDetectionResult(
                primary_language=None,
                confidence=0.30,
                detected_languages=[],
                is_mixed=False,
                is_transliterated=False,
                requires_clarification=True,
                reason="UNSUPPORTED_LANGUAGE",
            )

        # 2. Count characters per script
        ta_count = len(TAMIL_SCRIPT_REGEX.findall(clean_text))
        hi_count = len(HINDI_SCRIPT_REGEX.findall(clean_text))
        ml_count = len(MALAYALAM_SCRIPT_REGEX.findall(clean_text))
        latin_count = len(LATIN_SCRIPT_REGEX.findall(clean_text))

        # Extract Latin tokens for English or Transliteration matching
        tokens = re.findall(r"\b[A-Za-z]+\b", clean_text.lower())
        # Filter out entity placeholders if present
        clean_tokens = [t for t in tokens if not (t.startswith("__preserved_") or t.endswith("__"))]

        # ------------------------------------------------------------------
        # Case A: Tamil native script present
        # ------------------------------------------------------------------
        if ta_count > 0:
            if latin_count > 0 and len(clean_tokens) > 0:
                # Mixed Tamil + English (e.g. "En order status என்ன?" or "என் order எங்கே?")
                return self._finalize_result(
                    primary="ta",
                    detected=["ta", "en"],
                    confidence=0.94,
                    is_mixed=True,
                    is_transliterated=False,
                )
            else:
                # Purely Tamil
                return self._finalize_result(
                    primary="ta",
                    detected=["ta"],
                    confidence=0.98,
                    is_mixed=False,
                    is_transliterated=False,
                )

        # ------------------------------------------------------------------
        # Case B: Hindi (Devanagari) native script present
        # ------------------------------------------------------------------
        if hi_count > 0:
            if latin_count > 0 and len(clean_tokens) > 0:
                # Mixed Hindi + English (e.g. "Mera order status क्या है?" or "मेरा order कहाँ है?")
                return self._finalize_result(
                    primary="hi",
                    detected=["hi", "en"],
                    confidence=0.94,
                    is_mixed=True,
                    is_transliterated=False,
                )
            else:
                # Purely Hindi
                return self._finalize_result(
                    primary="hi",
                    detected=["hi"],
                    confidence=0.98,
                    is_mixed=False,
                    is_transliterated=False,
                )

        # ------------------------------------------------------------------
        # Case C: Malayalam native script present
        # ------------------------------------------------------------------
        if ml_count > 0:
            if latin_count > 0 and len(clean_tokens) > 0:
                # Mixed Malayalam + English (e.g. "Ente order status എന്താണ്?")
                return self._finalize_result(
                    primary="ml",
                    detected=["ml", "en"],
                    confidence=0.94,
                    is_mixed=True,
                    is_transliterated=False,
                )
            else:
                # Purely Malayalam
                return self._finalize_result(
                    primary="ml",
                    detected=["ml"],
                    confidence=0.98,
                    is_mixed=False,
                    is_transliterated=False,
                )

        # ------------------------------------------------------------------
        # Case D: Exclusively Latin script (English or Transliterated Indic)
        # ------------------------------------------------------------------
        if not clean_tokens:
            return LanguageDetectionResult(
                primary_language=None,
                confidence=0.20,
                detected_languages=[],
                is_mixed=False,
                requires_clarification=True,
                reason="LOW_LANGUAGE_CONFIDENCE",
            )

        # Calculate keyword hits
        ta_hits = sum(1 for w in clean_tokens if w in TAMIL_TRANSLIT_WORDS)
        hi_hits = sum(1 for w in clean_tokens if w in HINDI_TRANSLIT_WORDS)
        ml_hits = sum(1 for w in clean_tokens if w in MALAYALAM_TRANSLIT_WORDS)
        en_hits = sum(1 for w in clean_tokens if w in COMMON_ENGLISH_WORDS)

        # Check for Hindi transliteration
        if hi_hits > 0 and hi_hits >= ta_hits and hi_hits >= ml_hits:
            is_mixed = en_hits > 0
            detected = ["hi", "en"] if is_mixed else ["hi"]
            confidence = min(0.75 + 0.08 * hi_hits, 0.96)
            return self._finalize_result(
                primary="hi",
                detected=detected,
                confidence=confidence,
                is_mixed=is_mixed,
                is_transliterated=True,
                transliterated_lang="hi",
            )

        # Check for Tamil transliteration
        if ta_hits > 0 and ta_hits >= hi_hits and ta_hits >= ml_hits:
            is_mixed = en_hits > 0
            detected = ["ta", "en"] if is_mixed else ["ta"]
            confidence = min(0.75 + 0.08 * ta_hits, 0.96)
            return self._finalize_result(
                primary="ta",
                detected=detected,
                confidence=confidence,
                is_mixed=is_mixed,
                is_transliterated=True,
                transliterated_lang="ta",
            )

        # Check for Malayalam transliteration
        if ml_hits > 0 and ml_hits >= hi_hits and ml_hits >= ta_hits:
            is_mixed = en_hits > 0
            detected = ["ml", "en"] if is_mixed else ["ml"]
            confidence = min(0.75 + 0.08 * ml_hits, 0.96)
            return self._finalize_result(
                primary="ml",
                detected=detected,
                confidence=confidence,
                is_mixed=is_mixed,
                is_transliterated=True,
                transliterated_lang="ml",
            )

        # Check for English
        if en_hits > 0:
            ratio = en_hits / len(clean_tokens)
            confidence = 0.95 if ratio >= 0.3 else 0.75
            return self._finalize_result(
                primary="en",
                detected=["en"],
                confidence=confidence,
                is_mixed=False,
                is_transliterated=False,
            )

        # Purely unrecognized Latin tokens / gibberish (e.g. "asdf xyz qwerty")
        return LanguageDetectionResult(
            primary_language=None,
            confidence=0.30,
            detected_languages=[],
            is_mixed=False,
            is_transliterated=False,
            requires_clarification=True,
            reason="LOW_LANGUAGE_CONFIDENCE",
        )

    def _finalize_result(
        self,
        primary: str,
        detected: List[str],
        confidence: float,
        is_mixed: bool,
        is_transliterated: bool,
        transliterated_lang: Optional[str] = None,
    ) -> LanguageDetectionResult:
        """Applies configured confidence thresholds and supported language checks."""
        requires_clarification = False
        reason = None

        # Check supported languages
        if primary not in self.config.supported_languages:
            requires_clarification = True
            reason = "UNSUPPORTED_LANGUAGE"

        # Check confidence threshold
        elif confidence < self.config.language_confidence_threshold:
            requires_clarification = True
            reason = "LOW_LANGUAGE_CONFIDENCE"

        # Check transliteration threshold
        elif is_transliterated and confidence < self.config.transliteration_confidence_threshold:
            requires_clarification = True
            reason = "LOW_LANGUAGE_CONFIDENCE"

        return LanguageDetectionResult(
            primary_language=primary if not requires_clarification or reason != "UNSUPPORTED_LANGUAGE" else None,
            confidence=confidence,
            detected_languages=detected,
            is_mixed=is_mixed,
            is_transliterated=is_transliterated,
            transliterated_language=transliterated_lang,
            requires_clarification=requires_clarification,
            reason=reason,
        )

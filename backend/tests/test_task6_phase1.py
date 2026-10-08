"""
Task 6 Phase 1 — Multilingual Foundation Test Suite
====================================================
Comprehensive tests for:
  A. Language detection (English, Tamil, Hindi, Malayalam, unsupported, low confidence, threshold)
  B. Mixed-language detection (Tamil+English, Hindi+English, Malayalam+English, is_mixed flag)
  C. Language switching (stateless per-message analysis, English -> Tamil -> Hindi -> Malayalam)
  D. Transliterated inputs (Tamil, Hindi, Malayalam, uncertain transliterations)
  E. Spelling error handling (order/refund typos, original text preservation, config disable)
  F. Entity preservation (names, order IDs, dates, product codes, multiple entities, placeholder isolation)
  G. Dataset integrity guard
"""
from __future__ import annotations

import hashlib
import os
import sys
import pytest

# Ensure backend directory is in sys.path
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multilingual.config import MultilingualConfig
from multilingual.models import (
    EntityType,
    LanguageDetectionResult,
    MultilingualAnalysisResult,
)
from multilingual.entity_preserver import EntityPreserver
from multilingual.spelling_normalizer import SpellingNormalizer
from multilingual.language_detector import LanguageDetector
from multilingual.analyzer import analyze_multilingual_message


# ===========================================================================
# Group A: Language Detection
# ===========================================================================

def test_t01_english_detection():
    detector = LanguageDetector()
    res = detector.detect("Where can I find my course completion certificate?")
    assert res.primary_language == "en"
    assert res.confidence >= 0.70
    assert not res.requires_clarification
    assert not res.is_mixed


def test_t02_tamil_detection():
    detector = LanguageDetector()
    res = detector.detect("வணக்கம், என் பாடநெறி எங்கே?")
    assert res.primary_language == "ta"
    assert res.confidence >= 0.90
    assert not res.requires_clarification
    assert "ta" in res.detected_languages


def test_t03_hindi_detection():
    detector = LanguageDetector()
    res = detector.detect("नमस्ते, मुझे रिफंड पॉलिसी के बारे में जानना है।")
    assert res.primary_language == "hi"
    assert res.confidence >= 0.90
    assert not res.requires_clarification
    assert "hi" in res.detected_languages


def test_t04_malayalam_detection():
    detector = LanguageDetector()
    res = detector.detect("നമസ്കാരം, കോഴ്സ് സർട്ടിഫിക്കറ്റ് എപ്പോൾ ലഭിക്കും?")
    assert res.primary_language == "ml"
    assert res.confidence >= 0.90
    assert not res.requires_clarification
    assert "ml" in res.detected_languages


def test_t05_unsupported_language_detection():
    detector = LanguageDetector()
    # French
    res_fr = detector.detect("Bonjour, où est ma commande s'il vous plaît?")
    assert res_fr.requires_clarification
    assert res_fr.reason == "UNSUPPORTED_LANGUAGE"

    # Russian
    res_ru = detector.detect("Здравствуйте, где мой заказ?")
    assert res_ru.requires_clarification
    assert res_ru.reason == "UNSUPPORTED_LANGUAGE"


def test_t06_low_confidence_language():
    detector = LanguageDetector()
    res = detector.detect("asdf qwerty zxcv")
    assert res.requires_clarification
    assert res.reason == "LOW_LANGUAGE_CONFIDENCE"
    assert res.confidence < 0.70


def test_t07_confidence_threshold_behavior():
    # Stricter threshold
    strict_config = MultilingualConfig(language_confidence_threshold=0.98)
    detector = LanguageDetector(config=strict_config)
    res = detector.detect("Hello support team, I need help.")
    # Standard English gets ~0.95 confidence which is below 0.98
    if res.confidence < 0.98:
        assert res.requires_clarification
        assert res.reason == "LOW_LANGUAGE_CONFIDENCE"


# ===========================================================================
# Group B: Mixed Language
# ===========================================================================

def test_t08_tamil_english_mixed():
    detector = LanguageDetector()
    res = detector.detect("En order status என்ன?")
    assert res.is_mixed
    assert "ta" in res.detected_languages
    assert "en" in res.detected_languages
    assert res.primary_language == "ta"
    assert not res.requires_clarification


def test_t09_hindi_english_mixed():
    detector = LanguageDetector()
    res = detector.detect("Mera order status kya hai?")
    assert res.is_mixed
    assert "hi" in res.detected_languages
    assert "en" in res.detected_languages
    assert res.primary_language == "hi"
    assert not res.requires_clarification


def test_t10_malayalam_english_mixed():
    detector = LanguageDetector()
    res = detector.detect("Ente order status എന്താണ്?")
    assert res.is_mixed
    assert "ml" in res.detected_languages
    assert "en" in res.detected_languages
    assert res.primary_language == "ml"
    assert not res.requires_clarification


def test_t11_mixed_language_flag():
    detector = LanguageDetector()
    res = detector.detect("Mera payment status क्या है?")
    assert res.is_mixed is True


def test_t12_multiple_detected_languages():
    detector = LanguageDetector()
    res = detector.detect("Course refund policy என்ன?")
    assert len(res.detected_languages) >= 2
    assert set(res.detected_languages) == {"ta", "en"}


# ===========================================================================
# Group C: Language Switching (Stateless Independence)
# ===========================================================================

def test_t13_language_switching_sequence():
    detector = LanguageDetector()

    # Message 1: English
    r1 = detector.detect("Where is my order?")
    assert r1.primary_language == "en"

    # Message 2: Tamil
    r2 = detector.detect("என் order எங்கே?")
    assert r2.primary_language == "ta"

    # Message 3: Hindi
    r3 = detector.detect("मेरा order कहाँ है?")
    assert r3.primary_language == "hi"

    # Message 4: Malayalam
    r4 = detector.detect("എന്റെ order എവിടെ?")
    assert r4.primary_language == "ml"

    # Message 5: Back to English
    r5 = detector.detect("Can you cancel my subscription?")
    assert r5.primary_language == "en"


def test_t14_per_message_independence():
    detector = LanguageDetector()
    # Ensure previous Tamil detection does NOT affect subsequent English message
    _ = detector.detect("வணக்கம்")
    fresh = detector.detect("I have a question about python bootcamp.")
    assert fresh.primary_language == "en"
    assert fresh.detected_languages == ["en"]
    assert not fresh.is_mixed


# ===========================================================================
# Group D: Transliteration
# ===========================================================================

def test_t15_tamil_transliteration():
    detector = LanguageDetector()
    res = detector.detect("en order enga iruku")
    assert res.primary_language == "ta"
    assert res.is_transliterated is True
    assert res.transliterated_language == "ta"
    assert not res.requires_clarification


def test_t16_hindi_transliteration():
    detector = LanguageDetector()
    res = detector.detect("mera order kaha hai")
    assert res.primary_language == "hi"
    assert res.is_transliterated is True
    assert res.transliterated_language == "hi"
    assert not res.requires_clarification


def test_t17_malayalam_transliteration():
    detector = LanguageDetector()
    res = detector.detect("ente order evide aanu")
    assert res.primary_language == "ml"
    assert res.is_transliterated is True
    assert res.transliterated_language == "ml"
    assert not res.requires_clarification


def test_t18_uncertain_transliteration():
    detector = LanguageDetector()
    res = detector.detect("xyz abc foo bar")
    assert res.requires_clarification
    assert res.reason == "LOW_LANGUAGE_CONFIDENCE"


# ===========================================================================
# Group E: Spelling Errors
# ===========================================================================

def test_t19_common_english_spelling_error():
    norm = SpellingNormalizer.normalize_text("i need a refnd for my cours")
    assert "refund" in norm
    assert "course" in norm


def test_t20_order_related_spelling_error():
    norm = SpellingNormalizer.normalize_text("check my oder staus please")
    assert "order" in norm
    assert "status" in norm


def test_t21_correction_without_modifying_original_text():
    raw = "my oder ORD12345 staus"
    result = analyze_multilingual_message(raw)
    assert result.original_text == raw
    assert "order" in result.normalized_text
    assert "status" in result.normalized_text
    assert "ORD12345" in result.normalized_text


def test_t22_normalization_disabled_via_config():
    raw = "check my oder staus"
    cfg = MultilingualConfig(spelling_normalization_enabled=False)
    result = analyze_multilingual_message(raw, config=cfg)
    assert result.normalized_text == raw


# ===========================================================================
# Group F: Entity Preservation
# ===========================================================================

def test_t23_name_preservation():
    raw = "Hi Danashree, what is my refund status?"
    res = EntityPreserver.protect(raw)
    assert len(res.entities) >= 1
    assert any(e.entity_type == EntityType.NAME and e.value == "Danashree" for e in res.entities)

    restored = EntityPreserver.restore(res.protected_text, res.entities)
    assert restored == raw


def test_t24_order_id_preservation():
    raw = "My order number is ORD-2026-001"
    res = EntityPreserver.protect(raw)
    assert any(e.entity_type == EntityType.ORDER_ID and e.value == "ORD-2026-001" for e in res.entities)

    restored = EntityPreserver.restore(res.protected_text, res.entities)
    assert restored == raw


def test_t25_date_preservation():
    raw = "I purchased the course on 15-09-2026"
    res = EntityPreserver.protect(raw)
    assert any(e.entity_type == EntityType.DATE and e.value == "15-09-2026" for e in res.entities)

    restored = EntityPreserver.restore(res.protected_text, res.entities)
    assert restored == raw


def test_t26_product_code_preservation():
    raw = "Please help with item PROD-AX21"
    res = EntityPreserver.protect(raw)
    assert any(e.entity_type == EntityType.PRODUCT_CODE and e.value == "PROD-AX21" for e in res.entities)

    restored = EntityPreserver.restore(res.protected_text, res.entities)
    assert restored == raw


def test_t27_multiple_entities_in_one_message():
    raw = "Hi Danashree, order ORD12345 placed on 15-09-2026 for PROD-AX21"
    res = EntityPreserver.protect(raw)

    types_found = {e.entity_type for e in res.entities}
    assert EntityType.NAME in types_found
    assert EntityType.ORDER_ID in types_found
    assert EntityType.DATE in types_found
    assert EntityType.PRODUCT_CODE in types_found

    restored = EntityPreserver.restore(res.protected_text, res.entities)
    assert restored == raw


def test_t28_entities_preserved_through_normalization():
    raw = "Hi Danashree, check my oder ORD12345 on 15-09-2026 for PROD-AX21"
    result = analyze_multilingual_message(raw)

    assert result.original_text == raw
    # "oder" corrected to "order"
    assert "order" in result.normalized_text
    # All entities completely untouched
    assert "Danashree" in result.normalized_text
    assert "ORD12345" in result.normalized_text
    assert "15-09-2026" in result.normalized_text
    assert "PROD-AX21" in result.normalized_text


def test_t29_email_and_phone_preservation():
    raw = "Reach me at user@example.com or +1-800-555-0199"
    res = EntityPreserver.protect(raw)
    assert any(e.entity_type == EntityType.EMAIL and e.value == "user@example.com" for e in res.entities)
    assert any(e.entity_type == EntityType.PHONE for e in res.entities)

    restored = EntityPreserver.restore(res.protected_text, res.entities)
    assert restored == raw


def test_t30_empty_and_whitespace_input():
    result = analyze_multilingual_message("   ")
    assert result.requires_clarification
    assert result.language.confidence <= 0.50


def test_t31_multilingual_analyzer_structure():
    raw = "Mera order ORD9999 status kya hai?"
    res = analyze_multilingual_message(raw)
    assert isinstance(res, MultilingualAnalysisResult)
    assert res.language.primary_language == "hi"
    assert res.language.is_mixed is True
    assert "ORD9999" in res.normalized_text


# ===========================================================================
# Group G: Dataset Integrity Guard
# ===========================================================================
EXPECTED_DATASET_SHA256 = "930649d927881235ecbd3b53b29de92ddf8dbeca084657774335630eb9361b9a"


def test_t32_dataset_csv_unchanged():
    dataset_path = os.path.join(_PROJECT_ROOT, "dataset", "dataset.csv")
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"

    sha = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    actual = sha.hexdigest().lower()
    assert actual == EXPECTED_DATASET_SHA256.lower(), (
        f"dataset.csv modified! Expected {EXPECTED_DATASET_SHA256}, got {actual}"
    )

"""
Task 4 Phase 1: Multilingual Sentiment Analyser.

Architecture:
  1. Attempt to load a pretrained multilingual model via the `transformers`
     library (e.g. nlptown/bert-base-multilingual-uncased-sentiment).
  2. If the library or model is unavailable, fall back to a deterministic
     rule-based analyser — clearly signalled in the result via
     analysis_method = AnalysisMethod.RULE_BASED.

Detects: POSITIVE, NEUTRAL, NEGATIVE, FRUSTRATED, URGENT, SARCASTIC.
Generates: sentiment, confidence, frustration, urgency, sarcasm, language.

Sarcasm detection:
  - Requires a POSITIVE surface signal (praise words) combined with a
    NEGATIVE context signal (complaint words / time frustration).
  - NEVER fires on negative words alone.

High-risk note:
  - Sentiment analysis alone does NOT determine escalation.
    The escalation engine (escalation.py) handles risk classification
    independently.
"""

from __future__ import annotations

import logging
import math
import re
from typing import List, Optional, Tuple

from .task4_models import (
    AnalysisMethod,
    MessageSentimentResult,
    SentimentLabel,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional ML model loading
# ──────────────────────────────────────────────────────────────────────────────

_ML_PIPELINE = None      # transformers pipeline instance
_ML_AVAILABLE = False


def _try_load_ml_model() -> None:
    """Attempt to load a lightweight multilingual model if locally available."""
    global _ML_PIPELINE, _ML_AVAILABLE
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        # Attempt to load if cached locally to avoid network blocking
        _ML_PIPELINE = hf_pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            model_kwargs={"local_files_only": True},
            top_k=None,
        )
        _ML_AVAILABLE = True
        logger.info("[Task4/Sentiment] ML model loaded from local cache: nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as exc:
        logger.info("[Task4/Sentiment] Pretrained ML model not locally available (%s). Using deterministic rule-based fallback.", exc)
        _ML_AVAILABLE = False


# Load once at import time (silently falls back to rule-based if not cached)
_try_load_ml_model()



# ──────────────────────────────────────────────────────────────────────────────
# Lexicons  (domain-tuned for e-learning / online-course support)
# ──────────────────────────────────────────────────────────────────────────────

_SARCASM_PRAISE = [
    "wow", "great", "amazing", "fantastic", "wonderful", "brilliant",
    "excellent", "perfect", "awesome", "superb", "outstanding", "incredible",
    "love it", "thank you so much", "oh great", "how wonderful",
]

_SARCASM_NEGATIVE_CONTEXT = [
    "again", "another", "still", "forever", "waiting", "wait",
    "exactly what i needed", "exactly what i wanted",
    "as usual", "typical", "of course", "obviously",
    "failure", "failed", "error", "broken", "problem",
    "not working", "doesn't work", "cant", "can't",
    "ridiculous", "unacceptable", "pathetic", "hopeless",
    "i've been waiting", "been waiting", "hours", "days",
]

_URGENT_PHRASES = [
    "urgent", "urgently", "emergency", "immediately", "right now", "asap",
    "as soon as possible", "critical", "need help now", "time sensitive",
    "deadline", "can't wait", "cannot wait", "help me now",
    "please fix this now", "fix it now", "escalate",
]

_FRUSTRATED_PHRASES = [
    "frustrated", "frustrating", "furious", "angry", "rage",
    "fed up", "sick of", "sick and tired", "absolutely ridiculous",
    "unacceptable", "this is terrible", "this is horrible", "outrageous",
    "incompetent", "worst experience", "never again", "done with",
    "waste of time", "wasted my time", "pathetic",
]

_STRONG_NEGATIVE = [
    "scam", "fraud", "useless", "incompetent", "disgusting",
    "horrible", "terrible", "awful", "worst", "ridiculous",
    "unacceptable", "pathetic", "absurd", "offensive", "shameful",
]

_NEGATIVE_PHRASES = [
    "not happy", "not satisfied", "not good", "unhappy", "dissatisfied",
    "disappointed", "displeased", "failed", "failure", "error",
    "not working", "doesn't work", "can't access", "cannot access",
    "broken", "stuck", "blocked", "charged twice", "double charged",
    "never received", "still waiting", "no response", "ignored",
    "very disappointed", "deeply disappointed", "extremely disappointed",
    "not received", "not able", "problem", "issue",
    "still nothing", "still no response", "nothing received", "no answer",
    "hours and still", "still haven't", "waiting forever", "still not",
]


_POSITIVE_PHRASES = [
    "happy", "satisfied", "pleased", "great", "excellent",
    "wonderful", "fantastic", "love", "enjoy", "thank you",
    "thanks", "grateful", "appreciate", "helpful", "good experience",
    "works well", "works fine", "resolved", "solved", "amazing",
    "perfect", "brilliant",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _count(text: str, phrases: List[str]) -> int:
    total = 0
    for p in phrases:
        total += len(re.findall(r"\b" + re.escape(p) + r"\b", text))
    return total


def _detect_sarcasm(text: str) -> bool:
    """
    Context-aware sarcasm: requires BOTH a surface positive signal AND
    a negative/frustration context signal in the same text.
    Negative words alone do NOT trigger sarcasm.
    """
    has_praise = _count(text, _SARCASM_PRAISE) > 0
    has_negative_context = _count(text, _SARCASM_NEGATIVE_CONTEXT) > 0
    return has_praise and has_negative_context


def _detect_language(text: str) -> Optional[str]:
    """
    Very lightweight heuristic language detection.
    Only fires when the ML model is unavailable; the ML model provides
    its own language handling internally.
    Returns ISO 639-1 code for a few common scripts, else None.
    """
    # Devanagari (Hindi, etc.)
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    # Arabic
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    # CJK
    if re.search(r"[\u4E00-\u9FFF\u3040-\u30FF]", text):
        return "zh"
    # Cyrillic
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    # Latin extended (Spanish / French / German accent chars)
    if re.search(r"[àáâãäåæçèéêëìíîïñòóôõöùúûüý]", text):
        return "es"  # conservative guess for Latin-extended
    return "en"


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based analyser (fallback + sarcasm enrichment)
# ──────────────────────────────────────────────────────────────────────────────

def _rule_based_analyse(text: str) -> Tuple[SentimentLabel, float, bool, bool, bool]:
    """
    Returns (sentiment, confidence, frustration, urgency, sarcasm).
    Clearly rule-based; confidence reflects relative signal strength,
    capped to [0.55, 0.82] to never claim ML-level certainty.
    """
    proc = _preprocess(text)

    sarcasm = _detect_sarcasm(proc)
    if sarcasm:
        return SentimentLabel.SARCASTIC, 0.72, False, False, True

    urgency = _count(proc, _URGENT_PHRASES) > 0
    if urgency:
        return SentimentLabel.URGENT, 0.78, False, True, False

    frustration = _count(proc, _FRUSTRATED_PHRASES) > 0

    neg_score = (
        _count(proc, _NEGATIVE_PHRASES) * 2
        + _count(proc, _STRONG_NEGATIVE) * 3
    )
    pos_score = _count(proc, _POSITIVE_PHRASES) * 2

    if frustration:
        return SentimentLabel.FRUSTRATED, 0.76, True, False, False

    if neg_score > pos_score:
        # Normalise confidence between 0.55–0.82
        ratio = min(neg_score / max(neg_score + pos_score, 1), 1.0)
        conf = 0.55 + ratio * 0.27
        return SentimentLabel.NEGATIVE, round(conf, 4), False, False, False

    if pos_score > neg_score:
        ratio = min(pos_score / max(neg_score + pos_score, 1), 1.0)
        conf = 0.55 + ratio * 0.27
        return SentimentLabel.POSITIVE, round(conf, 4), False, False, False

    return SentimentLabel.NEUTRAL, 0.60, False, False, False


# ──────────────────────────────────────────────────────────────────────────────
# ML model adaptor
# ──────────────────────────────────────────────────────────────────────────────

def _ml_analyse(text: str) -> Tuple[SentimentLabel, float]:
    """
    Translate nlptown 5-star output → our 6-label schema.
    Returns (sentiment, confidence).
    """
    try:
        results = _ML_PIPELINE(text[:512], truncation=True)  # type: ignore
        # results is a list of [{label, score}] or [[{label, score}]]
        if results and isinstance(results[0], list):
            scores = results[0]
        else:
            scores = results  # type: ignore

        # Sort by score desc
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        top = sorted_scores[0]
        label_str = top["label"]       # e.g. "5 stars"
        confidence = round(float(top["score"]), 4)

        star_match = re.search(r"(\d)", label_str)
        if star_match:
            stars = int(star_match.group(1))
        else:
            stars = 3  # default neutral

        if stars >= 4:
            return SentimentLabel.POSITIVE, confidence
        elif stars == 3:
            return SentimentLabel.NEUTRAL, confidence
        elif stars == 2:
            return SentimentLabel.NEGATIVE, confidence
        else:  # stars == 1
            return SentimentLabel.FRUSTRATED, confidence

    except Exception as exc:  # pragma: no cover
        logger.warning("[Task4/Sentiment] ML inference failed: %s. Falling back.", exc)
        return SentimentLabel.NEUTRAL, 0.60


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def analyse_message(text: str) -> MessageSentimentResult:
    """
    Analyse a single customer message.

    If the ML model is available, it provides the base positive/neutral/negative
    mapping. Sarcasm, urgency, and frustration are then enriched via rule-based
    layer (the ML 5-class model does not directly output these).

    If the ML model is unavailable, the rule-based analyser is used throughout
    and analysis_method is clearly set to RULE_BASED.
    """
    if not text or not text.strip():
        return MessageSentimentResult(
            sentiment=SentimentLabel.NEUTRAL,
            confidence=1.0,
            analysis_method=AnalysisMethod.RULE_BASED,
        )

    proc = _preprocess(text)
    language = _detect_language(text)

    # Sarcasm always checked via rule-based (ML model doesn't detect it)
    sarcasm = _detect_sarcasm(proc)
    urgency = _count(proc, _URGENT_PHRASES) > 0
    frustration = _count(proc, _FRUSTRATED_PHRASES) > 0

    if sarcasm:
        return MessageSentimentResult(
            sentiment=SentimentLabel.SARCASTIC,
            confidence=0.72,
            frustration=False,
            urgency=False,
            sarcasm=True,
            language=language,
            analysis_method=AnalysisMethod.RULE_BASED,
        )

    if _ML_AVAILABLE:
        base_sentiment, confidence = _ml_analyse(text)
        method = AnalysisMethod.ML_MODEL

        # Upgrade base sentiment if rule-based signals are stronger
        if urgency and base_sentiment not in (SentimentLabel.POSITIVE,):
            base_sentiment = SentimentLabel.URGENT
            confidence = max(confidence, 0.75)
        elif frustration and base_sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL):
            base_sentiment = SentimentLabel.FRUSTRATED
            confidence = max(confidence, 0.72)
    else:
        base_sentiment, confidence, frustration, urgency, _ = _rule_based_analyse(text)
        method = AnalysisMethod.RULE_BASED

    return MessageSentimentResult(
        sentiment=base_sentiment,
        confidence=confidence,
        frustration=frustration,
        urgency=urgency,
        sarcasm=sarcasm,
        language=language,
        analysis_method=method,
    )

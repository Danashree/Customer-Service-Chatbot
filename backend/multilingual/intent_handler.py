"""
Task 6 Phase 2 — Multilingual Intent & Request Handler
======================================================
Provides intent detection, multi-intent decomposition, confidence evaluation,
and ambiguity/clarification detection for English, Tamil, Hindi, and Malayalam
(including transliterated and mixed-language inputs).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import MultilingualConfig
from .models import (
    ClarificationResult,
    ConversationContext,
    EntityType,
    IntentCandidate,
    IntentResult,
)


# =======================================================================
# Multilingual Intent Definitions & Match Patterns
# =======================================================================

INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "order_status": {
        "keywords": [
            # English
            r"\border\b", r"\btrack(?:ing)?\b", r"\bstatus\b", r"\bdelivery\b",
            r"\barrive\b", r"\bdispatch(?:ed)?\b", r"\bship(?:ment|ped)?\b",
            r"\bpackage\b", r"\bwhere is (?:my )?order\b", r"\bwhen will (?:it|my order) arrive\b",
            # Tamil & Tanglish
            r"எங்கே", r"நிலை", r"டெலிவரி", r"எப்போது வரும்", r"எப்போ வரும்", r"எப்போ(?:து)? வரும்",
            r"வந்து சேரும்", r"இன்னும் வரல", r"வரல",
            r"\benga(?:ey)?\b", r"\biruku\b", r"\beppodhu varum\b", r"\beppo varum\b", r"\bvandu serum\b",
            # Hindi & Hinglish
            r"कहाँ", r"ट्रैक", r"स्थिति", r"कब आएगा", r"डिलीवरी", r"पहुंचेगा",
            r"\bkaha(?:n)? hai\b", r"\bkab aayega\b", r"\bkab tak\b", r"\bpahunchega\b",
            # Malayalam & Manglish
            r"എവിടെ(?:യാണ്)?", r"ട്രാക്ക്", r"സ്റ്റാറ്റസ്", r"എപ്പോൾ വരും", r"ഡെലിവറി",
            r"\bevide(?:yaanu)?\b", r"\beppol varum\b",
        ],
        "weight": 1.0,
    },
    "refund_request": {
        "keywords": [
            # English
            r"\brefund\b", r"\bmoney back\b", r"\breturn\b", r"\bcancel(?:lation)?\b",
            r"\breimburse(?:ment)?\b", r"\bget a refund\b",
            # Tamil & Tanglish
            r"ரீஃபண்ட்", r"பணம் திரும்ப", r"பணம் திருப்பி", r"திரும்ப கிடைக்குமா",
            r"ரத்து", r"ரத்து செய்ய", r"cancel பண்ண", r"cancel பண்ணலாமா",
            r"\bpanam thirumba\b", r"\bthirumba kidaikuma\b", r"\bthirike venam\b", r"\bcancel\b",
            # Hindi & Hinglish
            r"रिफंड", r"पैसे वापस", r"पैसा वापस", r"वापसी",
            r"\bpaisa wapas\b", r"\bpaise wapas\b", r"\bwapas chahiye\b", r"\bpaise vapis\b",
            # Malayalam & Manglish
            r"റീഫണ്ട്", r"പണം തിരികെ", r"തിരികെ വേണം", r"റീഫണ്ട് ലഭിക്കുമോ",
            r"\bpanam thirike\b", r"\bthirike venam\b", r"\brefund labhikkumo\b",
        ],
        "weight": 1.0,
    },
    "course_access": {
        "keywords": [
            # English
            r"\bcourse access\b", r"\bcannot access\b", r"\baccess\b", r"\blogin\b", r"\bportal\b", r"\bpassword\b",
            r"\bvideo(?:s)?\b", r"\blecture(?:s)?\b", r"\bmodule(?:s)?\b", r"\bnot playing\b",
            r"\bcannot view\b", r"\benroll(?:ment)?\b", r"\bcourse not (?:working|opening)\b",
            # Tamil & Tanglish
            r"கோர்ஸ்", r"பாடநெறி", r"உள்நுழைவு", r"பாஸ்வேர்ட்", r"வீடியோ",
            r"\bvideo odala\b", r"\bpaarka mudiyala\b", r"\bcourse access\b",
            # Hindi & Hinglish
            r"कोर्स", r"लॉगिन", r"पासवर्ड", r"वीडियो नहीं चल रहा",
            r"\bcourse access nahi\b", r"\bvideo nahi chal raha\b", r"\bdekh nahi pa raha\b",
            # Malayalam & Manglish
            r"കോഴ്സ്", r"ലോഗിൻ", r"പാസ്‌വേഡ്", r"വീഡിയോ",
            r"\bvideo kaanunilla\b", r"\blogin cheyyaan pattunnilla\b",
        ],
        "weight": 1.0,
    },
    "payment_issue": {
        "keywords": [
            # English
            r"\bpayment (?:failed|fail(?:ure)?|declined|error|issue|problem|not working|stuck|pending)\b",
            r"\bpayment\b.*?\b(?:failed|fail(?:ure)?|declined|error|issue|problem|not working|stuck|did not go through)\b",
            r"\b(?:failed|declined|error|issue|problem)\b.*?\bpayment\b",
            r"\bcharged (?:twice|double)\b", r"\bdeducted\b",
            r"\breceipt\b", r"\binvoice\b", r"\bbilling\b", r"\btransaction failed\b",
            r"\btransaction fail(?:ure)?\b",
            # Tamil & Tanglish
            r"பேமெண்ட்", r"பணம் பிடித்தம்", r"இரண்டு முறை", r"ரசீது",
            r"\bpayment failed\b", r"\bpanam deducted\b", r"\birandu murai\b",
            # Hindi & Hinglish
            r"पेमेंट", r"पैसे कट गए", r"दो बार", r"रसीद",
            r"\bpayment fail\b", r"\bdo baar\b", r"\bkat gaye\b",
            # Malayalam & Manglish
            r"പേയ്മെന്റ്", r"പണം പോയി", r"രണ്ടുതവണ", r"രസീത്",
            r"\bpayment poyi\b", r"\branduthavana\b",
        ],
        "weight": 1.0,
    },
    "policy_inquiry": {
        "keywords": [
            # English
            r"\bpolicy\b", r"\bterms\b", r"\bcertificate\b", r"\bvalidity\b",
            r"\bduration\b", r"\brules\b", r"\bsyllabus\b", r"\baccreditation\b",
            # Tamil & Tanglish
            r"கொள்கை", r"சான்றிதழ்", r"விதிகள்", r"கால அளவு",
            r"\bcertificate kedaikuma\b", r"\bvithigal\b",
            # Hindi & Hinglish
            r"नीति", r"नियम", r"प्रमाणपत्र", r"सर्टिफिकेट", r"अवधि",
            r"\bcertificate milega\b", r"\bniyam kya hai\b",
            # Malayalam & Manglish
            r"നയം", r"സർട്ടിഫിക്കറ്റ്", r"നിയമങ്ങൾ", r"കാലാവധി",
            r"\bcertificate kittumo\b",
        ],
        "weight": 1.0,
    },
    "escalation": {
        "keywords": [
            # English
            r"\bhuman\b", r"\bagent\b", r"\brepresentative\b", r"\bspeak to (?:a )?(?:human|person)\b",
            r"\btalk to (?:an )?agent\b", r"\bmanager\b", r"\bsupervisor\b",
            # Tamil & Tanglish
            r"மனிதர்", r"ஏஜென்ட்", r"பேச வேண்டும்", r"\bpesa vendum\b",
            # Hindi & Hinglish
            r"इंसान", r"एजेंट", r"बात करनी है", r"\bbaat karni hai\b",
            # Malayalam & Manglish
            r"ഏജന്റ്", r"മനുഷ്യൻ", r"സംസാരിക്കണം", r"\bsamsarikkanam\b",
        ],
        "weight": 1.0,
    },
    "greeting": {
        "keywords": [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b", r"\bgood evening\b",
            r"வணக்கம்", r"\bvanakkam\b",
            r"नमस्ते", r"नमस्कार", r"\bnamaste\b", r"\bpranam\b",
            r"നമസ്കാരം", r"\bnamaskaram\b",
        ],
        "weight": 0.85,
    },
    "gratitude": {
        "keywords": [
            r"\bthank you\b", r"\bthanks\b", r"\bappreciate\b",
            r"நன்றி", r"\bnanri\b",
            r"धन्यवाद", r"शुक्रिया", r"\bdhanyavad\b", r"\bshukriya\b",
            r"നന്ദി", r"\bnanni\b",
        ],
        "weight": 0.85,
    },
    "general_inquiry": {
        "keywords": [
            # EMI & Payment options inquiries
            r"\b(?:do you )?(?:offer|have|provide|accept)?\s*emi\b",
            r"\bemi (?:payment(?:s)?|option(?:s)?|facility|available|plan(?:s)?)\b",
            r"\b(?:what|which)\s+(?:are the\s+)?payment (?:option|method|mode)(?:s)?\b",
            r"\bpayment (?:option|method|mode)(?:s)?(?: are available)?\b",
            r"\bhow (?:can i|to|do i) pay\b",
            r"\bways to pay\b",
            r"\binstallment(?:s)? (?:plan|option|available)?\b",
            # Course offerings & details inquiries
            r"\bwhat courses (?:do you |are )?offer(?:ed)?\b",
            r"\b(?:what|which) course(?:s)? (?:do you have|are available|are there|can i take)\b",
            r"\btell me about (?:the )?course(?:s)?\b",
            r"\bdetails? (?:about|of) (?:the )?course(?:s)?\b",
            r"\binformation (?:about|on) (?:the )?course(?:s)?\b",
            r"\bcourse (?:offering|catalog|list|curriculum|syllabus|details)\b",
            r"\blist of (?:available )?courses\b",
            r"\bavailable course(?:s)?\b",
            # General service & offering inquiries
            r"\bdo you (?:offer|have|provide)\b",
            r"\bshould i (?:learn|choose|take)\b",
            r"\bcan i (?:use|take|learn|enroll)\b",
            r"\bwhat (?:do you|does \w+) offer\b",
            r"\btell me about\b",
            r"\boptions are available\b",
            r"\bcan you tell me about\b",
            # Multilingual inquiries (Tamil, Hindi, Malayalam)
            r"என்னென்ன கோர்ஸ்", r"கோர்ஸ் விவரங்கள்", r"கோர்ஸ் பற்றி", r"emi வசதி", r"பேமெண்ட் முறைகள்",
            r"\bemi option iruka\b", r"\benna course(?:s)? iruku\b", r"\bcourse pathi sollunga\b", r"\bpayment options enna\b",
            r"कौन से कोर्स", r"कोर्स के बारे में", r"emi उपलब्ध", r"पेमेंट विकल्प",
            r"\bkaun se course\b", r"\bcourse ke baare mein\b", r"\bemi mil sakti hai\b", r"\bpayment options kya hai\b",
            r"ഏതൊക്കെ കോഴ്സ്", r"കോഴ്സുകളെക്കുറിച്ച്", r"emi ലഭ്യമാണോ", r"പേയ്മെന്റ് ഓപ്ഷനുകൾ",
            r"\bethekkya courses\b", r"\bcourseine kurichu\b", r"\bemi undoo\b",
        ],
        "weight": 1.0,
    },
}

# Domain specific intents that take priority over general informational inquiries
SPECIFIC_INTENTS: Set[str] = {
    "order_status",
    "refund_request",
    "course_access",
    "payment_issue",
    "policy_inquiry",
    "escalation",
}

# Multi-intent connectors across languages
MULTI_INTENT_CONNECTORS = [
    # English
    r"\band\b", r"\balso\b", r"\bas well as\b", r"\bplus\b", r"\bin addition to\b",
    # Tamil
    r"மற்றும்", r"கூட", r"மேலும்", r"\bmatrum\b", r"\bkooda\b", r"\bappuram\b",
    # Hindi
    r"और", r"तथा", r"एवं", r"भी", r"\baur\b", r"\btatha\b", r"\bbhi\b", r"\bsaath mein\b",
    # Malayalam
    r"കൂടാതെ", r"പിന്നെ", r"\bkoodaathe\b", r"\bkoodathe\b", r"\bpinne\b", r"\bathukoodathe\b",
]

# Ambiguous phrases that cannot be resolved without domain context
GENERIC_AMBIGUOUS_PATTERNS = [
    r"^(?:can you )?(?:please )?fix it\??$",
    r"^(?:please )?help me\??$",
    r"^what about this\??$",
    r"^do something\??$",
    r"^issue\??$",
    r"^problem\??$",
    r"^help\??$",
    r"^courses?\??$",
    r"^payments?\??$",
    r"^what\??$",
    r"^how\??$",
]

# Interrogative openings that mark a well-formed question worth answering from the
# knowledge base even when no support-intent keyword matches.
SUBSTANTIVE_QUESTION_PATTERN = re.compile(
    r"^(?:"
    r"(?:why|what|which|when|where|who|whom|whose|how)\b"
    r"|(?:should|could|would|will|can|may|might|is|are|was|were|do|does|did|have|has|had)"
    r"\s+(?:i|we|you|it|this|that|these|those|there|my|our|your|the|a|an)\b"
    r")",
    re.IGNORECASE,
)
MIN_SUBSTANTIVE_WORDS = 4


class IntentHandler:
    """Detects and decomposes customer requests across multiple languages."""

    def __init__(self, config: Optional[MultilingualConfig] = None):
        self.config = config or MultilingualConfig.from_env()

    def detect_intents(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
    ) -> IntentResult:
        """
        Analyzes user text to extract:
          - Single or multiple intent candidates
          - Intent confidence scores
          - Multi-intent flag
          - Clarification requirements for ambiguous or low-confidence queries
        """
        stripped = text.strip().strip("\"'“”’‘")
        if not stripped:
            return IntentResult(
                intents=[],
                primary_intent=None,
                confidence=0.0,
                is_multi_intent=False,
                requires_clarification=True,
                clarification_reason="EMPTY_INPUT",
                clarification_question="How may I assist you with your course, order, or payment today?",
            )

        # 1. Check for genuine ambiguity (e.g. "Can you fix it?")
        for pattern in GENERIC_AMBIGUOUS_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                return IntentResult(
                    intents=[],
                    primary_intent=None,
                    confidence=0.30,
                    is_multi_intent=False,
                    requires_clarification=True,
                    clarification_reason="AMBIGUOUS_INTENT",
                    clarification_question="Could you clarify whether you mean your course access, payment, or order?",
                )

        # 2. Split clauses on multi-intent connectors or punctuation
        clauses = self._split_clauses(stripped)

        # 3. Detect candidate intents for each clause and overall
        candidates: List[IntentCandidate] = []
        seen_intents: Set[str] = set()

        if len(clauses) > 1:
            for clause in clauses:
                clause_candidates = self._score_text(clause)
                for cand in clause_candidates:
                    if cand.intent not in seen_intents and cand.confidence >= self.config.intent_confidence_threshold:
                        cand.matched_text = clause
                        candidates.append(cand)
                        seen_intents.add(cand.intent)

        # If clause splitting did not yield multi-intents, score the full text
        full_text_candidates = self._score_text(stripped)
        for cand in full_text_candidates:
            if cand.intent not in seen_intents:
                candidates.append(cand)
                seen_intents.add(cand.intent)

        # Sort candidates: specific domain intents take priority over general_inquiry / greeting
        candidates.sort(
            key=lambda c: (
                1 if c.intent in SPECIFIC_INTENTS else 0,
                c.confidence,
            ),
            reverse=True,
        )

        # If a specific domain intent is matched above threshold in a single-clause query,
        # general_inquiry yields to the specific intent
        has_specific = any(
            c.intent in SPECIFIC_INTENTS and c.confidence >= self.config.intent_confidence_threshold
            for c in candidates
        )
        if has_specific and len(clauses) <= 1:
            candidates = [c for c in candidates if c.intent != "general_inquiry"]

        # 4. Check if any intent was detected
        if not candidates:
            if self._is_substantive_question(stripped):
                # Well-formed question that matches no support intent: treat as a
                # general FAQ inquiry instead of demanding clarification.
                candidates = [
                    IntentCandidate(
                        intent="general_inquiry",
                        confidence=0.72,
                        matched_text=stripped,
                    )
                ]
            else:
                return IntentResult(
                    intents=[],
                    primary_intent=None,
                    confidence=0.0,
                    is_multi_intent=False,
                    requires_clarification=True,
                    clarification_reason="LOW_INTENT_CONFIDENCE",
                    clarification_question="Could you please clarify your request so I can assist you better?",
                )

        primary = candidates[0]

        # 5. Check confidence against threshold
        if primary.confidence < self.config.intent_confidence_threshold:
            return IntentResult(
                intents=candidates,
                primary_intent=primary.intent,
                confidence=primary.confidence,
                is_multi_intent=False,
                requires_clarification=True,
                clarification_reason="LOW_INTENT_CONFIDENCE",
                clarification_question="Could you please clarify your request so I can assist you better?",
            )

        # Filter candidates above threshold for multi-intent
        valid_candidates = [
            c for c in candidates if c.confidence >= self.config.intent_confidence_threshold
        ]
        is_multi = len(valid_candidates) > 1

        # 6. Check context-sensitive follow-up (e.g. "When will it arrive?" or "Can I cancel it?")
        # If the intent is order_status or refund_request/cancellation, check if context has ambiguity in order IDs
        if primary.intent in ("order_status", "refund_request") and context is not None:
            order_clarification = self._resolve_order_ambiguity(stripped, context)
            if order_clarification.requires_clarification:
                return IntentResult(
                    intents=valid_candidates,
                    primary_intent=primary.intent,
                    confidence=primary.confidence,
                    is_multi_intent=is_multi,
                    requires_clarification=True,
                    clarification_reason=order_clarification.reason,
                    clarification_question=order_clarification.clarification_question,
                )
            elif order_clarification.candidate_entities:
                # Successfully resolved from context
                primary.parameters["order_id"] = order_clarification.candidate_entities[0]

        return IntentResult(
            intents=valid_candidates,
            primary_intent=primary.intent,
            confidence=primary.confidence,
            is_multi_intent=is_multi,
            requires_clarification=False,
            clarification_reason=None,
            clarification_question=None,
        )

    def _is_substantive_question(self, text: str) -> bool:
        """True for well-formed questions long enough to answer from the knowledge base."""
        cleaned = text.strip("\"'“”’‘ ")
        if len(cleaned.split()) < MIN_SUBSTANTIVE_WORDS:
            return False
        return bool(SUBSTANTIVE_QUESTION_PATTERN.search(cleaned))

    def _split_clauses(self, text: str) -> List[str]:
        """Splits a compound message into logical request segments."""
        # Connector regex pattern
        connector_pattern = r"(?:[;,]|\?+|\b(?:and|also|as well as|plus|மற்றும்|கூட|மேலும்|matrum|kooda|aur|tatha|bhi|കൂടാതെ|പിന്നെ|koodaathe|pinne)\b)"
        raw_parts = re.split(connector_pattern, text, flags=re.IGNORECASE)
        clauses = [p.strip() for p in raw_parts if p.strip() and len(p.strip()) > 2]
        return clauses if len(clauses) > 1 else [text]

    def _score_text(self, text: str) -> List[IntentCandidate]:
        """Calculates confidence for all intent categories on a given text segment."""
        candidates = []
        lower_text = text.lower()

        for intent_name, defn in INTENT_PATTERNS.items():
            keywords = defn["keywords"]
            base_weight = defn["weight"]

            matches = 0
            for kw in keywords:
                if re.search(kw, lower_text, re.IGNORECASE):
                    matches += 1

            if matches > 0:
                # Calculate confidence: strong keyword match gets 0.85 - 0.95
                confidence = min(0.95, 0.70 + (matches * 0.10)) * base_weight
                candidates.append(
                    IntentCandidate(
                        intent=intent_name,
                        confidence=round(confidence, 2),
                        matched_text=text,
                    )
                )

        return candidates

    def _resolve_order_ambiguity(
        self,
        current_text: str,
        context: ConversationContext,
    ) -> ClarificationResult:
        """
        Inspects conversation context to determine if an order-related follow-up
        (e.g. 'When will it arrive?') cleanly resolves to an active order ID or requires clarification.
        """
        from .entity_preserver import ORDER_ID_PATTERN
        if ORDER_ID_PATTERN.search(current_text):
            return ClarificationResult(requires_clarification=False)

        # Check order IDs in context
        order_history = context.entity_history.get(EntityType.ORDER_ID.value, [])
        active_order = context.active_entities.get(EntityType.ORDER_ID.value)

        # Remove duplicate mentions preserving order
        unique_orders: List[str] = []
        for oid in order_history:
            if oid not in unique_orders:
                unique_orders.append(oid)

        # If a single active order ID is clearly active:
        # Note: If a correction happened, active_order was explicitly set and superseded old orders
        has_correction = any(c.entity_type == EntityType.ORDER_ID for c in context.corrections)

        if has_correction and active_order:
            # Active order was resolved by correction
            return ClarificationResult(
                requires_clarification=False,
                candidate_entities=[active_order],
            )

        if len(unique_orders) == 1:
            return ClarificationResult(
                requires_clarification=False,
                candidate_entities=[unique_orders[0]],
            )

        if len(unique_orders) > 1 and not has_correction:
            # Multiple competing order IDs exist in conversation without correction!
            joined_orders = " or ".join(unique_orders)
            return ClarificationResult(
                requires_clarification=True,
                reason="AMBIGUOUS_ENTITY",
                clarification_question=f"Which order do you mean: {joined_orders}?",
                candidate_entities=unique_orders,
            )

        return ClarificationResult(requires_clarification=False)

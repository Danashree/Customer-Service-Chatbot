"""Task 3 Support Ticket Module (Phase 1 + Phase 2 + Phase 3)."""

from .models import (
    TicketBase,
    TicketStatus,
    TicketCreateRequest,
    TicketStatusUpdateRequest,
    TicketValidationResult,
    TicketResponse,
    PriorityResponse,
    SLAResponse,
    SLAConfigUpdateRequest,
)
from .priority_enums import (
    Severity,
    Sentiment,
    CustomerImpact,
    Priority,
    SLAStatus,
    EscalationStatus,
)
from .business_hours import (
    BusinessHoursConfig,
    is_business_day,
    is_business_time,
    next_business_moment,
    calculate_business_minutes,
    add_business_minutes,
    add_business_hours,
    calculate_sla_due_at,
    calculate_sla_warning_at,
)
from .sla_config import (
    SLAConfig,
    get_sla_config,
    update_sla_config,
    reset_sla_config,
)
from .sentiment import detect_sentiment
from .severity import classify_severity, classify_customer_impact
from .priority_calculator import calculate_priority, PriorityBreakdown
from .sla_manager import initialise_sla, check_sla, resolve_sla, SLAState
from .extractor import ConversationExtractor
from .validator import MandatoryFieldValidator
from .storage import TicketStorage
from .service import TicketService

# Phase 3
from .routing_models import (
    RoutingStatus,
    Team,
    Agent,
    RoutingResult,
    RoutingConfigUpdate,
)
from .routing_config import (
    RoutingConfig,
    get_routing_config,
    update_routing_config,
    reset_routing_config,
)
from .router import (
    SkillDetector,
    TicketRouter,
)

# Phase 4
from .duplicate_config import (
    DuplicateConfig,
    get_duplicate_config,
    set_duplicate_config,
    reset_duplicate_config,
)
from .phase4_models import (
    DuplicateStatus,
    RelationshipType,
    DuplicateComparison,
    DuplicateCheckResponse,
    TicketRelationship,
    RelationshipResponse,
    IssueGroup,
    GroupCreateRequest,
    GroupResponse,
    GroupListResponse,
    HandoffSummary,
)
from .duplicate_detector import (
    DuplicateDetector,
    compute_text_similarity,
)
from .issue_grouper import (
    GroupStorage,
    IssueGrouper,
)
from .handoff import (
    HandoffGenerator,
)

# Task 4
from .task4_models import (
    SentimentLabel,
    AnalysisMethod,
    RiskLevel,
    RiskCondition,
    QueueType,
    EscalationRecordStatus,
    ResponseTone,
    MessageSentimentResult,
    ConversationSentimentResult,
    SentimentAnalysisResponse,
    RiskAssessment,
    EscalationRequest,
    EscalationEvaluation,
    EscalationRecord,
    ToneControlResult,
    SentimentAnalyzeRequest,
)
from .sentiment_multilingual import (
    analyse_message,
)
from .conversation_analyzer import (
    analyse_conversation,
)
from .tone_controller import (
    get_tone,
    evaluate_tone,
)
from .escalation import (
    assess_risk,
    EscalationConfig,
    get_escalation_config,
    update_escalation_config,
    reset_escalation_config,
    EscalationStorage,
    EscalationEngine,
    get_escalation_engine,
)

__all__ = [
    # Phase 1
    "TicketBase",
    "TicketStatus",
    "TicketCreateRequest",
    "TicketStatusUpdateRequest",
    "TicketValidationResult",
    "TicketResponse",
    "ConversationExtractor",
    "MandatoryFieldValidator",
    "TicketStorage",
    "TicketService",
    # Phase 2 Enums
    "Severity",
    "Sentiment",
    "CustomerImpact",
    "Priority",
    "SLAStatus",
    "EscalationStatus",
    # Phase 2 Models
    "PriorityResponse",
    "SLAResponse",
    "SLAConfigUpdateRequest",
    # Phase 2 Logic
    "BusinessHoursConfig",
    "is_business_day",
    "is_business_time",
    "next_business_moment",
    "calculate_business_minutes",
    "add_business_minutes",
    "add_business_hours",
    "calculate_sla_due_at",
    "calculate_sla_warning_at",
    "SLAConfig",
    "get_sla_config",
    "update_sla_config",
    "reset_sla_config",
    "detect_sentiment",
    "classify_severity",
    "classify_customer_impact",
    "calculate_priority",
    "PriorityBreakdown",
    "initialise_sla",
    "check_sla",
    "resolve_sla",
    "SLAState",
    # Phase 3
    "RoutingStatus",
    "Team",
    "Agent",
    "RoutingResult",
    "RoutingConfigUpdate",
    "RoutingConfig",
    "get_routing_config",
    "update_routing_config",
    "reset_routing_config",
    "SkillDetector",
    "TicketRouter",
    # Phase 4
    "DuplicateConfig",
    "get_duplicate_config",
    "set_duplicate_config",
    "reset_duplicate_config",
    "DuplicateStatus",
    "RelationshipType",
    "DuplicateComparison",
    "DuplicateCheckResponse",
    "TicketRelationship",
    "RelationshipResponse",
    "IssueGroup",
    "GroupCreateRequest",
    "GroupResponse",
    "GroupListResponse",
    "HandoffSummary",
    "DuplicateDetector",
    "compute_text_similarity",
    "GroupStorage",
    "IssueGrouper",
    "HandoffGenerator",
    # Task 4
    "SentimentLabel",
    "AnalysisMethod",
    "RiskLevel",
    "RiskCondition",
    "QueueType",
    "EscalationRecordStatus",
    "ResponseTone",
    "MessageSentimentResult",
    "ConversationSentimentResult",
    "SentimentAnalysisResponse",
    "RiskAssessment",
    "EscalationRequest",
    "EscalationEvaluation",
    "EscalationRecord",
    "ToneControlResult",
    "SentimentAnalyzeRequest",
    "analyse_message",
    "analyse_conversation",
    "get_tone",
    "evaluate_tone",
    "assess_risk",
    "EscalationConfig",
    "get_escalation_config",
    "update_escalation_config",
    "reset_escalation_config",
    "EscalationStorage",
    "EscalationEngine",
    "get_escalation_engine",
]


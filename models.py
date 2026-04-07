from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
# base model for all models
#defines common fields for all models



# these are the actions that can be taken by the agent
class ActionType(str, Enum):
    request_category = "request_category"
    cross_reference = "cross_reference"
    flag_anomaly = "flag_anomaly"
    submit_findings = "submit_findings"
    request_external_confirmation = "request_external_confirmation"

# gives various types of fraud that can be detected
class FraudType(str, Enum):
    channel_stuffing = "channel_stuffing"
    round_tripping = "round_tripping"
    phantom_revenue = "phantom_revenue"
    cookie_jar_reservation = "cookie_jar_reservation"
    bill_and_hold = "bill_and_hold"

# types of evidence that a document can be (hidden from the agent)
class EvidenceType(str, Enum):
    smoking_gun = "smoking_gun"
    circumstantial = "circumstantial"
    clean = "clean"


class DocumentStatus(str, Enum):
    available = "available"
    hidden = "hidden"
    reclassified = "reclassified"
    delayed = "delayed"
    destroyed = "destroyed"


# gives various categories of documents that can be audited
class DocumentCategory(str, Enum):
    financial_statements = "financial_statements"
    bank_records = "bank_records"
    invoices = "invoices"
    contracts = "contracts"
    correspondence = "correspondence"
    tax_filings = "tax_filings"
    hr_records = "hr_records"
    audit_trails = "audit_trails"

# env input(tells the environment what the current state of the audit is)
class AuditAction(BaseModel):
    action_type: ActionType # type   restricted by ActionType has to be one of the values in ActionType
    params: Dict[str, Any] = Field(default_factory=dict) # parameters for the action




# true document details (full)
class Document(BaseModel):
    id: str
    category: DocumentCategory = Field(default=DocumentCategory.financial_statements) # GROUND TRUTH, what the document is actually about

    # ground truth
    evidence_type: EvidenceType = Field(default=EvidenceType.smoking_gun) 

    # dynamic state
    status: DocumentStatus = Field(default=DocumentStatus.available)  # "available", "hidden", "reclassified", "destroyed", "delayed"
    location: str # WHERE THE DOCUMENT IS CURRENTLY LOCATED, after CFO moves it around
    # no default value, force it to be set on creation
    # location is string instead of DocumentCategory enum to allow for hidden categories (not available in enum)

    # content abstraction
    key_signals: List[str] = Field(default_factory=list)

    # history (for delta tracking)
    history: List[str] = Field(default_factory=list)

    # for when the document is hidden
    hidden_until_step: Optional[int] = None




# Summary of document from agent's perspective
class DocumentSummary(BaseModel):
    id: str = Field(default="")
    category: DocumentCategory = Field(default=DocumentCategory.financial_statements)
    status: DocumentStatus = Field(default=DocumentStatus.available)
    anomalies_flagged: int = Field(default=0)
    key_signals: List[str] = Field(default_factory=list)

    



# the env ouput what agent sees after every step
class AuditObservation(BaseModel):
    step: int
    remaining_steps: int

    document_status: Dict[str, DocumentStatus] = Field(default_factory=dict)
    # {
    #   "Q3_revenue_ledge": "available"
    #   "loans": "reclassified"
    #   "Q4_revenue_ledger": "missing"
    # }

    # tracking changes in document status
    document_status_delta: List[str] = Field(default_factory=list)

    # tracking categories requested so far
    requested_categories_so_far: List[str] = Field(default_factory=list)

    # tracking documents received so far
    documents_received: Dict[str, DocumentSummary] = Field(default_factory=dict)



    anomalies_flagged: List[str] = Field(default_factory=list)

    # submitted conclusions
    findings: List[str] = Field(default_factory=list)

    # will only have current round actions, not the history
    cfo_visible_actions: List[str] = Field(default_factory=list)

    # messages from the environment to the agent
    env_feedback: List[str] = Field(default_factory=list)

    done: bool=False

# full internal state of the audit
class AuditState(BaseModel):
     # ===== Ground Truth =====
    true_fraud_type: FraudType

    # ===== Documents =====
    document_index: Dict[str, Document]  # doc_id → Document

    # category → list of doc_ids (what lives where currently)
    document_location_index: Dict[str, List[str]]

    # ===== CFO (hidden logic) =====
    cfo_strategy: str
    cfo_budget_remaining: int
    cfo_actions_history: List[str]
    cfo_actions_current_round: List[str]

    # track which docs CFO cares about hiding
    critical_docs: List[str]

    # ===== Agent Progress (internal tracking) =====
    step: int
    max_steps: int

    requested_categories: List[str]
    received_doc_ids: List[str]

    anomalies_flagged: List[str]
    findings_submitted: Optional[Dict[str, Any]] = None

    # ===== For Observation Mapping =====
    last_known_status: Dict[str, DocumentStatus]  # for delta computation

    # ===== Reward bookkeeping =====
    reward_events: List[str]

    # messages from the environment to the agent, FOR EACH STEP
    env_feedback: List[str] = Field(default_factory=list)

    # ===== Config =====
    difficulty: str
    
    # ===== External Confirmation Tracking =====
    external_confirmations_used: int = 0
    env_feedback: List[str] = Field(default_factory=list)

# reward info for the agent grader output
class RewardInfo(BaseModel):
    total: float

    components: Dict[str, float] = Field(default_factory=dict)

    events: List[str] = Field(default_factory=list)


# task difficulty settings
class TaskConfig(BaseModel):
    # identity
    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"

    # core setup
    fraud_type: FraudType
    company_seed: int 
    max_steps: int 

    # CFO configuration
    cfo_strategy: str  # "passive" | "reactive" | "proactive"
    cfo_budget_per_round: int 
    hide_duration_rounds: int 

    # evidence distribution
    smoking_gun_count: int 
    circumstantial_count: int 
    clean_count: int

    # external confirmation constraints
    max_external_confirmations: int 
    external_confirmation_cost: int   

    
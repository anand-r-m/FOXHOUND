from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
# base model for all models
#defines common fields for all models



# these are the actions that can be taken by the agent
class ActionType(str, Enum):
    request_document = "request_document"
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
    parameters: Dict[str, Any] = Field(default_factory=dict) # parameters for the action

# the env ouput what agent sees after every step
class AuditObservation(BaseModel):
    step: int
    remaining_steps: int

    available_documents: List[DocumentCategory] = Field(default_factory=list)
    requested_so_far: List[DocumentCategory] = Field(default_factory=list)

    documents_received: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    anomalies_flagged: List[str] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)

    CFO

# full internal state of the audit
class AuditState(BaseModel):
    pass

# reward info for the agent
class RewardInfo(BaseModel):
    pass

# task difficulty settings
class TaskConfig(BaseModel):
    pass

    
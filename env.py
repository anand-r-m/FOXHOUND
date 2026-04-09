import random
from typing import Optional
from models import (
    ActionType, EvidenceType, DocumentStatus, DocumentCategory,
    AuditAction, AuditObservation, AuditState, RewardInfo, TaskConfig,
    Document, DocumentSummary, FraudType
)

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD TEMPLATES
# Static document blueprints for each fraud archetype.
# Each template has three buckets: smoking_gun, circumstantial, clean.
# The env plants documents from these at reset() time according to TaskConfig counts.
# key_signals are the content the agent actually reads — they're deliberately written
# to hint at the fraud without naming it outright.
# ─────────────────────────────────────────────────────────────────────────────

FRAUD_TEMPLATES = {
    FraudType.channel_stuffing: {
        "smoking_gun": [
            {
                "id": "inv_q3_early",
                "category": DocumentCategory.invoices,
                "key_signals": [
                    "Invoice date: Sep 28 — Delivery date: Oct 15",
                    "Revenue recognized at invoice, not delivery",
                    "340% spike in Q3 invoice volume vs Q2"
                ]
            }
        ],
        "circumstantial": [
            {
                "id": "fin_q3_statement",
                "category": DocumentCategory.financial_statements,
                "key_signals": ["Q3 revenue 340% above Q2", "Accounts receivable doubled"]
            },
            {
                "id": "contract_no_milestone",
                "category": DocumentCategory.contracts,
                "key_signals": ["No milestone-based delivery terms", "Revenue on signature"]
            },
            {
                "id": "corr_sales_pressure",
                "category": DocumentCategory.correspondence,
                "key_signals": ["Email: 'push everything through before quarter close'"]
            }
        ],
        "clean": [
            {"id": "tax_q2",      "category": DocumentCategory.tax_filings,         "key_signals": []},
            {"id": "hr_standard", "category": DocumentCategory.hr_records,           "key_signals": []},
            {"id": "audit_q1",    "category": DocumentCategory.audit_trails,         "key_signals": []},
            {"id": "bank_q1",     "category": DocumentCategory.bank_records,         "key_signals": []},
            {"id": "inv_q1",      "category": DocumentCategory.invoices,             "key_signals": []},
            {"id": "fin_q1",      "category": DocumentCategory.financial_statements, "key_signals": []},
        ]
    },

    FraudType.round_tripping: {
        "smoking_gun": [
            {
                "id": "bank_wire_roundtrip",
                "category": DocumentCategory.bank_records,
                "key_signals": [
                    "Wire transfer $4.2M to Cayman entity Sep 3",
                    "Incoming $4.1M from same entity Sep 28",
                    "Net $100k difference booked as revenue"
                ]
            }
        ],
        "circumstantial": [
            {
                "id": "fin_revenue_spike",
                "category": DocumentCategory.financial_statements,
                "key_signals": ["Revenue spike with no COGS increase"]
            },
            {
                "id": "audit_interco",
                "category": DocumentCategory.audit_trails,
                "key_signals": ["Intercompany transfers not reconciled"]
            },
            {
                "id": "corr_shell",
                "category": DocumentCategory.correspondence,
                "key_signals": ["Emails to unregistered entity 'Cayman Trade Partners LLC'"]
            }
        ],
        "clean": [
            {"id": "tax_standard", "category": DocumentCategory.tax_filings,         "key_signals": []},
            {"id": "hr_standard",  "category": DocumentCategory.hr_records,           "key_signals": []},
            {"id": "inv_q1",       "category": DocumentCategory.invoices,             "key_signals": []},
            {"id": "contract_std", "category": DocumentCategory.contracts,            "key_signals": []},
            {"id": "fin_q1",       "category": DocumentCategory.financial_statements, "key_signals": []},
            {"id": "bank_q1",      "category": DocumentCategory.bank_records,         "key_signals": []},
        ]
    },

    FraudType.phantom_revenue: {
        "smoking_gun": [
            {
                "id": "bank_no_receipt",
                "category": DocumentCategory.bank_records,
                "key_signals": [
                    "Revenue recognised Q3: $8.2M — Cash receipts Q3: $1.1M",
                    "No wire transfers matching invoiced amounts"
                ]
            },
            {
                "id": "contract_fake",
                "category": DocumentCategory.contracts,
                "key_signals": [
                    "Signed by 'GlobalTech Solutions' — company dissolved 2022",
                    "No corresponding purchase orders"
                ]
            }
        ],
        "circumstantial": [
            {
                "id": "fin_ar_spike",
                "category": DocumentCategory.financial_statements,
                "key_signals": ["AR 820% of industry average", "No bad debt provisions"]
            },
            {
                "id": "inv_unmatched",
                "category": DocumentCategory.invoices,
                "key_signals": ["14 invoices with no delivery records"]
            },
            {
                "id": "audit_gaps",
                "category": DocumentCategory.audit_trails,
                "key_signals": ["Approval workflow bypassed for 9 transactions"]
            },
            {
                "id": "corr_pressure",
                "category": DocumentCategory.correspondence,
                "key_signals": ["CFO email: 'we need 8M by EOQ, make it work'"]
            }
        ],
        "clean": [
            {"id": "tax_standard", "category": DocumentCategory.tax_filings,         "key_signals": []},
            {"id": "hr_standard",  "category": DocumentCategory.hr_records,           "key_signals": []},
            {"id": "contract_q1",  "category": DocumentCategory.contracts,            "key_signals": []},
            {"id": "bank_q1",      "category": DocumentCategory.bank_records,         "key_signals": []},
            {"id": "fin_q1",       "category": DocumentCategory.financial_statements, "key_signals": []},
            {"id": "inv_q1",       "category": DocumentCategory.invoices,             "key_signals": []},
        ]
    }
}


class ForensicAuditEnv:

    def __init__(self, task_config: TaskConfig):
        # store config for use across all methods
        self.config = task_config
        # _state is None until reset() is called — asserts guard against this
        self._state: Optional[AuditState] = None
        # _rng is the seeded random instance for ALL randomness in this env
        # stored on the instance so CFO methods can use it without being passed it
        self._rng: Optional[random.Random] = None

    def _require_state(self) -> AuditState:
        """Narrow Optional state for type checkers; same invariant as runtime asserts in step()."""
        assert self._state is not None, "Call reset() before using the environment"
        return self._state

    def _require_rng(self) -> random.Random:
        assert self._rng is not None, "Call reset() before CFO randomness is available"
        return self._rng

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def reset(self) -> AuditObservation:
        # isolated rng — doesn't touch global random state
        # same seed always produces identical env layout AND identical CFO behavior
        self._rng = random.Random(self.config.company_seed)

        templates = FRAUD_TEMPLATES[self.config.fraud_type]
        documents: dict[str, Document] = {}
        critical_docs: list[str] = []

        # plant smoking gun docs up to the count specified in task config
        for tmpl in templates["smoking_gun"][: self.config.smoking_gun_count]:
            doc = Document(
                id=tmpl["id"],
                category=tmpl["category"],
                evidence_type=EvidenceType.smoking_gun,
                status=DocumentStatus.available,
                # location starts as the true category — CFO moves it away from here
                location=tmpl["category"].value,
                key_signals=tmpl["key_signals"],
            )
            documents[doc.id] = doc
            # critical_docs is the CFO's hit list — what it will try to protect
            critical_docs.append(doc.id)

        # same pattern for circumstantial evidence
        for tmpl in templates["circumstantial"][: self.config.circumstantial_count]:
            doc = Document(
                id=tmpl["id"],
                category=tmpl["category"],
                evidence_type=EvidenceType.circumstantial,
                status=DocumentStatus.available,
                location=tmpl["category"].value,
                key_signals=tmpl["key_signals"],
            )
            documents[doc.id] = doc
            critical_docs.append(doc.id)

        # clean docs are filler — they exist to make the category space noisy
        for tmpl in templates["clean"][: self.config.clean_count]:
            doc = Document(
                id=tmpl["id"],
                category=tmpl["category"],
                evidence_type=EvidenceType.clean,
                status=DocumentStatus.available,
                location=tmpl["category"].value,
                key_signals=tmpl["key_signals"],
            )
            documents[doc.id] = doc
            # clean docs are NOT added to critical_docs — CFO ignores them

        # location_index maps current_location → [doc_ids]
        # this is what the agent queries when it requests a category
        # CFO mutates this index when it reclassifies docs
        location_index: dict[str, list[str]] = {}
        for doc in documents.values():
            location_index.setdefault(doc.location, []).append(doc.id)

        # NOTE: models.py requires two additions to AuditState for this to work:
        #   external_confirmations_used: int = 0
        #   env_feedback: List[str] = Field(default_factory=list)
        self._state = AuditState(
            true_fraud_type=self.config.fraud_type,
            document_index=documents,
            document_location_index=location_index,
            cfo_strategy=self.config.cfo_strategy,
            cfo_budget_remaining=self.config.cfo_budget_per_round,
            cfo_actions_history=[],
            cfo_actions_current_round=[],
            critical_docs=critical_docs,
            step=0,
            max_steps=self.config.max_steps,
            requested_categories=[],
            received_doc_ids=[],
            anomalies_flagged=[],
            findings_submitted=None,
            # last_known_status seeds the delta tracker — starts as the clean initial state
            last_known_status={doc_id: doc.status for doc_id, doc in documents.items()},
            reward_events=[],
            env_feedback=[],
            external_confirmations_used=0,
            difficulty=self.config.difficulty,
        )

        return self._build_observation()

    def step(self, action: AuditAction) -> tuple[AuditObservation, RewardInfo, bool, dict]:
        s = self._require_state()

        # wipe all per-round fields at the start of each step
        # these are rebuilt fresh every round so the observation only shows current-round info
        s.cfo_actions_current_round = []
        s.env_feedback = []
        s.reward_events = []
        reward_components: dict[str, float] = {}

        # check hide timers BEFORE handling the action
        # so a doc that expired this step is already available when the agent requests it
        self._restore_expired_docs()

        # dispatch table — maps action type to its handler
        # cleaner than a chain of if/elif, and makes it easy to add new action types
        handlers = {
            ActionType.request_category:              self._handle_request_category,
            ActionType.cross_reference:               self._handle_cross_reference,
            ActionType.flag_anomaly:                  self._handle_flag_anomaly,
            ActionType.request_external_confirmation: self._handle_external_confirmation,
            ActionType.submit_findings:               self._handle_submit_findings,
        }
        handlers[action.action_type](action, reward_components)

        # CFO reacts AFTER the agent's action is fully processed
        # this means the agent already received whatever was available before CFO hides it
        # next round the delta will show what the CFO did
        self._cfo_react(action)

        # snapshot current status AFTER CFO has acted
        # this becomes the baseline for next round's delta computation
        s.last_known_status = {
            doc_id: doc.status for doc_id, doc in s.document_index.items()
        }

        # advance the clock — handlers that cost extra steps (external confirmation)
        # already bumped s.step internally before this increment
        s.step += 1
        done = s.step >= s.max_steps or s.findings_submitted is not None

        raw_total = sum(reward_components.values())
        reward_info = RewardInfo(
            total=max(1e-6, min(raw_total, 1 - 1e-6)),
            components=reward_components,
            events=list(s.reward_events),
        )

        return self._build_observation(), reward_info, done, {}

    def state(self) -> AuditState:
        return self._require_state()

    # ─────────────────────────────────────────
    # Action Handlers
    # each handler takes the action and the reward_components dict
    # and mutates both the state and the reward dict in place
    # ─────────────────────────────────────────

    def _handle_request_category(self, action: AuditAction, rewards: dict):
        s = self._require_state()
        category = action.params.get("category")

        # normalize: agent may pass a DocumentCategory enum or a raw string
        # the location_index is keyed by string values, so we always convert
        if isinstance(category, DocumentCategory):
            category = category.value

        if not category:
            s.env_feedback.append("ERROR: missing 'category' param")
            return

        # penalize repeat requests — agent already has whatever was in this category
        if category in s.requested_categories:
            rewards["repeat_request_penalty"] = -0.04
            s.reward_events.append("repeat_request_penalty")
            s.env_feedback.append(f"'{category}' already requested — wasted action")
            return

        s.requested_categories.append(category)

        # look up what documents currently live at this location
        # note: after CFO reclassifies, some docs may have moved out of their original category
        doc_ids_here = s.document_location_index.get(category, [])
        returned = []

        for doc_id in doc_ids_here:
            doc = s.document_index[doc_id]

            if doc.status == DocumentStatus.available:
                if doc_id not in s.received_doc_ids:
                    s.received_doc_ids.append(doc_id)
                doc.history.append(f"step {s.step}: received by agent")
                returned.append(doc_id)

                # step rewards for finding evidence — encourages the agent to probe hot categories
                if doc.evidence_type == EvidenceType.smoking_gun:
                    rewards["smoking_gun_received"] = rewards.get("smoking_gun_received", 0) + 0.08
                    s.reward_events.append(f"smoking_gun_received:{doc_id}")
                elif doc.evidence_type == EvidenceType.circumstantial:
                    rewards["circumstantial_received"] = rewards.get("circumstantial_received", 0) + 0.04
                    s.reward_events.append(f"circumstantial_received:{doc_id}")

            else:
                # doc is present but concealed — agent learns something is being withheld
                # this is valuable information, NOT a failure — no penalty here
                s.env_feedback.append(
                    f"'{doc_id}' in '{category}' unavailable ({doc.status.value})"
                )

        # only penalize if this category genuinely has NO documents at all
        # if docs exist but are hidden, that's useful information — not a wasted action
        if not returned and len(doc_ids_here) == 0:
            rewards["empty_category_penalty"] = -0.06
            s.reward_events.append("empty_category_penalty")
            s.env_feedback.append(f"No documents exist in '{category}'")
        elif not returned:
            # docs exist but all concealed — inform agent without penalizing
            s.env_feedback.append(f"All documents in '{category}' are currently concealed")

    def _handle_cross_reference(self, action: AuditAction, rewards: dict):
        s = self._require_state()
        doc_a_id = action.params.get("doc_a")
        doc_b_id = action.params.get("doc_b")

        if not doc_a_id or not doc_b_id:
            s.env_feedback.append("ERROR: cross_reference requires 'doc_a' and 'doc_b'")
            return

        # agent can only cross-reference docs it has actually received
        for doc_id in (doc_a_id, doc_b_id):
            if doc_id not in s.received_doc_ids:
                s.env_feedback.append(f"ERROR: '{doc_id}' not in received documents")
                return

        doc_a = s.document_index[doc_a_id]
        doc_b = s.document_index[doc_b_id]

        # contradiction logic: two non-clean docs from different categories with signals
        # the agent is finding a logical inconsistency between two pieces of evidence
        contradiction = (
            doc_a.evidence_type != EvidenceType.clean
            and doc_b.evidence_type != EvidenceType.clean
            and doc_a.category != doc_b.category
            and len(doc_a.key_signals) > 0
            and len(doc_b.key_signals) > 0
        )

        if contradiction:
            rewards["cross_reference_hit"] = 0.12
            s.reward_events.append(f"cross_reference_hit:{doc_a_id}+{doc_b_id}")
            s.env_feedback.append(f"CONTRADICTION FOUND: {doc_a_id} and {doc_b_id} conflict")
        else:
            s.env_feedback.append(f"No contradiction found between {doc_a_id} and {doc_b_id}")

    def _handle_flag_anomaly(self, action: AuditAction, rewards: dict):
        s = self._require_state()
        doc_id = action.params.get("doc_id")
        description = action.params.get("description", "no description")

        if not doc_id:
            s.env_feedback.append("ERROR: flag_anomaly requires 'doc_id'")
            return

        if doc_id not in s.received_doc_ids:
            s.env_feedback.append(f"ERROR: '{doc_id}' not in received documents")
            return

        flag = f"{doc_id}: {description}"
        if flag in s.anomalies_flagged:
            s.env_feedback.append("Duplicate anomaly flag — ignored")
            return

        s.anomalies_flagged.append(flag)
        doc = s.document_index[doc_id]

        if doc.evidence_type != EvidenceType.clean:
            rewards["valid_anomaly"] = rewards.get("valid_anomaly", 0) + 0.05
            s.reward_events.append(f"valid_anomaly:{doc_id}")
        else:
            # false positive — agent is generating noise, penalized at grading time too
            rewards["false_positive"] = rewards.get("false_positive", 0) - 0.08
            s.reward_events.append(f"false_positive:{doc_id}")

        s.env_feedback.append(f"Anomaly recorded on '{doc_id}'")

    def _handle_external_confirmation(self, action: AuditAction, rewards: dict):
        s = self._require_state()
        category = action.params.get("category")

        # normalize enum → string same as request_category
        if isinstance(category, DocumentCategory):
            category = category.value

        # hard cap on external confirmations — prevents agent from bypassing CFO entirely
        if s.external_confirmations_used >= self.config.max_external_confirmations:
            s.env_feedback.append("External confirmation limit reached — request denied")
            return

        if not category:
            s.env_feedback.append("ERROR: request_external_confirmation requires 'category'")
            return

        s.external_confirmations_used += 1

        # costs extra steps — step() will add 1 more at end, so we pre-burn (cost - 1) here
        # capped at max_steps so we don't go negative on remaining_steps
        s.step = min(
            s.step + self.config.external_confirmation_cost - 1,
            s.max_steps
        )

        doc_ids_here = s.document_location_index.get(category, [])
        returned = []

        for doc_id in doc_ids_here:
            doc = s.document_index[doc_id]
            # destroyed is the only status that external confirmation cannot bypass
            # everything else (hidden, reclassified, delayed) is retrievable via third party
            if doc.status != DocumentStatus.destroyed:
                if doc_id not in s.received_doc_ids:
                    s.received_doc_ids.append(doc_id)
                doc.history.append(f"step {s.step}: obtained via external confirmation")
                returned.append(doc_id)

                # extra reward if the doc was actively concealed — agent successfully bypassed CFO
                if doc.status in (
                    DocumentStatus.hidden,
                    DocumentStatus.reclassified,
                    DocumentStatus.delayed
                ):
                    rewards["external_bypass"] = rewards.get("external_bypass", 0) + 0.15
                    s.reward_events.append(f"external_bypass:{doc_id}")

        # penalize if external confirmation returned nothing — costly wasted action
        if not returned:
            rewards["external_waste"] = -0.05
            s.reward_events.append("external_waste")

        s.env_feedback.append(
            f"External confirmation '{category}': {len(returned)} doc(s) obtained"
        )

    def _handle_submit_findings(self, action: AuditAction, rewards: dict):
        s = self._require_state()
        # store the submission — done flag will be True on next observation build
        s.findings_submitted = action.params
        s.env_feedback.append("Findings submitted — episode ending")

    # ─────────────────────────────────────────
    # CFO Adversary
    # ─────────────────────────────────────────

    def _cfo_react(self, agent_action: AuditAction):
        s = self._require_state()
        rng = self._require_rng()

        # passive CFO does nothing — easy task
        if s.cfo_strategy == "passive":
            return

        # budget resets every round — CFO can't carry unused actions forward
        budget = self.config.cfo_budget_per_round

        if s.cfo_strategy == "reactive":
            # reactive CFO only responds to the category the agent just touched
            # if agent requests invoices, CFO only tries to hide/reclassify invoice docs
            category = agent_action.params.get("category")
            if isinstance(category, DocumentCategory):
                category = category.value
            if not category:
                return

            for doc_id in s.critical_docs:
                if budget <= 0:
                    break
                doc = s.document_index[doc_id]
                if doc.location == category and doc.status == DocumentStatus.available:
                    # 50/50 between hide and reclassify — uses seeded rng for determinism
                    if rng.random() < 0.5:
                        self._cfo_hide(doc_id)
                    else:
                        self._cfo_reclassify(doc_id)
                    budget -= 1

        elif s.cfo_strategy == "proactive":
            # proactive CFO prioritizes smoking gun docs regardless of what agent requested
            # acts preemptively — if a smoking gun is exposed, hide it immediately
            for doc_id in s.critical_docs:
                if budget <= 0:
                    break
                doc = s.document_index[doc_id]
                if (
                    doc.evidence_type == EvidenceType.smoking_gun
                    and doc.status == DocumentStatus.available
                ):
                    if rng.random() < 0.5:
                        self._cfo_hide(doc_id)
                    else:
                        self._cfo_reclassify(doc_id)
                    budget -= 1

    def _cfo_hide(self, doc_id: str):
        """Temporarily hide a document. Auto-restores after hide_duration_rounds steps."""
        s = self._require_state()
        doc = s.document_index[doc_id]
        doc.status = DocumentStatus.hidden
        # set the expiry step — _restore_expired_docs checks this every step
        doc.hidden_until_step = s.step + self.config.hide_duration_rounds
        doc.history.append(
            f"step {s.step}: CFO hid — expires step {doc.hidden_until_step}"
        )
        entry = f"step {s.step}: CFO hid '{doc_id}'"
        s.cfo_actions_current_round.append(entry)
        s.cfo_actions_history.append(entry)

    def _cfo_reclassify(self, doc_id: str):
        """Move a document to misc_ops, making it unfindable via its original category."""
        s = self._require_state()
        doc = s.document_index[doc_id]

        old_location = doc.location

        # remove from current location index
        if old_location in s.document_location_index:
            if doc_id in s.document_location_index[old_location]:
                s.document_location_index[old_location].remove(doc_id)

        # move to misc_ops — a hidden category not in the standard DocumentCategory enum
        doc.location = "misc_ops"
        doc.status = DocumentStatus.reclassified
        # set the same hide timer so reclassified docs also auto-restore
        doc.hidden_until_step = s.step + self.config.hide_duration_rounds

        # register in the new location
        s.document_location_index.setdefault("misc_ops", []).append(doc_id)

        doc.history.append(f"step {s.step}: CFO reclassified from {old_location} → misc_ops")
        entry = f"step {s.step}: CFO reclassified '{doc_id}' from {old_location}"
        s.cfo_actions_current_round.append(entry)
        s.cfo_actions_history.append(entry)

    def _restore_expired_docs(self):
        """Run at the start of each step. Returns hidden/reclassified docs whose timer expired."""
        s = self._require_state()
        for doc_id, doc in s.document_index.items():
            if (
                doc.status in (DocumentStatus.hidden, DocumentStatus.reclassified)
                and doc.hidden_until_step is not None
                and s.step >= doc.hidden_until_step
            ):
                # restore to its true category location
                # first remove from wherever it currently is
                if doc.location in s.document_location_index:
                    if doc_id in s.document_location_index[doc.location]:
                        s.document_location_index[doc.location].remove(doc_id)

                # put it back at its true category
                true_location = doc.category.value
                doc.location = true_location
                doc.status = DocumentStatus.available
                doc.hidden_until_step = None
                s.document_location_index.setdefault(true_location, []).append(doc_id)

                doc.history.append(f"step {s.step}: auto-restored to {true_location}")
                s.env_feedback.append(f"'{doc_id}' restored to '{true_location}'")

    # ─────────────────────────────────────────
    # Observation Builder
    # The single place where ground truth is filtered into what the agent can see.
    # If you're ever unsure whether the agent "should" know something, the answer
    # lives here.
    # ─────────────────────────────────────────

    def _build_observation(self) -> AuditObservation:
        s = self._require_state()

        # document_status: only docs the agent has already received
        # agent has no visibility into docs it hasn't touched yet
        document_status = {
            doc_id: s.document_index[doc_id].status
            for doc_id in s.received_doc_ids
        }

        # delta: what changed between last observation and now for known docs
        # this is the core mechanic of the hard task — disappearances are evidence
        document_status_delta = [
            f"{doc_id}: {s.last_known_status[doc_id].value} → {s.document_index[doc_id].status.value}"
            for doc_id in s.received_doc_ids
            if doc_id in s.last_known_status
            and s.document_index[doc_id].status != s.last_known_status[doc_id]
        ]

        # CFO visibility: only actions that touched docs the agent already knows about
        # CFO moves on unseen docs stay completely invisible — that's the information asymmetry
        cfo_visible_actions = [
            entry for entry in s.cfo_actions_current_round
            if any(doc_id in entry for doc_id in s.received_doc_ids)
        ]

        # build document summaries — strips ground truth fields (evidence_type, history,
        # hidden_until_step) and exposes only what an auditor could actually observe
        documents_received = {}
        for doc_id in s.received_doc_ids:
            doc = s.document_index[doc_id]
            anomaly_count = sum(1 for a in s.anomalies_flagged if a.startswith(doc_id))
            documents_received[doc_id] = DocumentSummary(
                id=doc_id,
                category=doc.category,
                status=doc.status,
                anomalies_flagged=anomaly_count,
                # append current location so agent can see when a doc has been reclassified
                # without this, the agent has no way to know a doc moved to misc_ops
                key_signals=doc.key_signals + [f"location:{doc.location}"],
            )

        findings = [str(s.findings_submitted)] if s.findings_submitted else []

        return AuditObservation(
            step=s.step,
            remaining_steps=s.max_steps - s.step,
            document_status=document_status,
            document_status_delta=document_status_delta,
            requested_categories_so_far=list(s.requested_categories),
            documents_received=documents_received,
            anomalies_flagged=list(s.anomalies_flagged),
            findings=findings,
            cfo_visible_actions=cfo_visible_actions,
            env_feedback=list(s.env_feedback),
            done=s.step >= s.max_steps or s.findings_submitted is not None,
        )
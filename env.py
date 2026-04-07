class ForensicAuditEnv:

    def __init__(self, task_config: TaskConfig):
        self.config = task_config
        self._state: Optional[AuditState] = None

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def reset(self) -> AuditObservation:
        # random.Random(seed) gives us an isolated rng that doesn't
        # affect global random state — same seed always gives same env
        rng = random.Random(self.config.company_seed)  # noqa: F841 (available for future use)

        templates = FRAUD_TEMPLATES[self.config.fraud_type]
        documents: dict[str, Document] = {}
        critical_docs: list[str] = []

        # plant evidence according to task config counts
        for tmpl in templates["smoking_gun"][: self.config.smoking_gun_count]:
            doc = Document(
                id=tmpl["id"],
                category=tmpl["category"],
                evidence_type=EvidenceType.smoking_gun,
                status=DocumentStatus.available,
                location=tmpl["category"].value,
                key_signals=tmpl["key_signals"],
            )
            documents[doc.id] = doc
            critical_docs.append(doc.id)

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

        # location index: category_value → [doc_ids currently there]
        # this is what the CFO manipulates when it moves/reclassifies docs
        location_index: dict[str, list[str]] = {}
        for doc in documents.values():
            location_index.setdefault(doc.location, []).append(doc.id)

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
            last_known_status={doc_id: doc.status for doc_id, doc in documents.items()},
            reward_events=[],
            env_feedback=[],
            difficulty=self.config.difficulty,
        )

        return self._build_observation()

    def step(self, action: AuditAction) -> tuple[AuditObservation, RewardInfo, bool, dict]:
        assert self._state is not None, "Call reset() before step()"
        s = self._state

        # clear everything that is per-round
        s.cfo_actions_current_round = []
        s.env_feedback = []
        s.reward_events = []
        reward_components: dict[str, float] = {}

        # restore any docs whose hide timer expired this step
        self._restore_expired_docs()

        # route to the right handler
        handlers = {
            ActionType.request_category:              self._handle_request_category,
            ActionType.cross_reference:               self._handle_cross_reference,
            ActionType.flag_anomaly:                  self._handle_flag_anomaly,
            ActionType.request_external_confirmation: self._handle_external_confirmation,
            ActionType.submit_findings:               self._handle_submit_findings,
        }
        handlers[action.action_type](action, reward_components)
        prev_status = {
            doc_id: doc.status for doc_id, doc in s.document_index.items()
        }

        # CFO reacts after agent acts
        self._cfo_react(action)

        changes = 0
        for doc_id, doc in s.document_index.items():
            if doc_id in prev_status and doc.status != prev_status[doc_id]:
                changes += 1
                s.reward_events.append(f"status_change:{doc_id}")

        if changes > 0:
            reward_components["status_change"] = min(0.10 * changes, 0.15)

        # snapshot status BEFORE incrementing step
        # so next round's delta compares against end-of-this-round state
        s.last_known_status = {
            doc_id: doc.status for doc_id, doc in s.document_index.items()
        }

        s.step += 1
        done = s.step >= s.max_steps or s.findings_submitted is not None

        reward_info = RewardInfo(
            total=sum(reward_components.values()),
            components=reward_components,
            events=list(s.reward_events),
        )

        return self._build_observation(), reward_info, done, {}

    def state(self) -> AuditState:
        assert self._state is not None, "Call reset() before state()"
        return self._state

    # ─────────────────────────────────────────
    # Action Handlers
    # ─────────────────────────────────────────

    def _handle_request_category(self, action: AuditAction, rewards: dict):
        s = self._state
        category = action.params.get("category")

        if isinstance(category, DocumentCategory):
            category = category.value
    
        if not category:
            s.env_feedback.append("ERROR: missing 'category' param")
            return

        if category in s.requested_categories:
            rewards["repeat_request_penalty"] = -0.04
            s.reward_events.append("repeat_request_penalty")
            s.env_feedback.append(f"'{category}' already requested — wasted action")
            return

        s.requested_categories.append(category)

        doc_ids_here = s.document_location_index.get(category, [])
        returned = []

        for doc_id in doc_ids_here:
            doc = s.document_index[doc_id]

            if doc.status == DocumentStatus.available:
                # use a set check to avoid duplicates if agent somehow gets same doc twice
                if doc_id not in s.received_doc_ids:
                    s.received_doc_ids.append(doc_id)
                doc.history.append(f"step {s.step}: received by agent")
                returned.append(doc_id)

                if doc.evidence_type == EvidenceType.smoking_gun:
                    rewards["smoking_gun_received"] = rewards.get("smoking_gun_received", 0) + 0.08
                    s.reward_events.append(f"smoking_gun_received:{doc_id}")
                elif doc.evidence_type == EvidenceType.circumstantial:
                    rewards["circumstantial_received"] = rewards.get("circumstantial_received", 0) + 0.04
                    s.reward_events.append(f"circumstantial_received:{doc_id}")

            else:
                # doc exists but is concealed — the agent learns something is being withheld
                s.env_feedback.append(f"'{doc_id}' in {category} unavailable ({doc.status.value})")

        if not returned:
            s.env_feedback.append(f"No documents returned from '{category}'")
            rewards["empty_category_penalty"] = -0.06
            s.reward_events.append("empty_category_penalty")

    def _handle_cross_reference(self, action: AuditAction, rewards: dict):
        s = self._state
        doc_a_id = action.params.get("doc_a")
        doc_b_id = action.params.get("doc_b")

        if not doc_a_id or not doc_b_id:
            s.env_feedback.append("ERROR: cross_reference requires 'doc_a' and 'doc_b'")
            return

        for doc_id in (doc_a_id, doc_b_id):
            if doc_id not in s.received_doc_ids:
                s.env_feedback.append(f"ERROR: '{doc_id}' not in received documents")
                return

        doc_a = s.document_index[doc_a_id]
        doc_b = s.document_index[doc_b_id]

        # a contradiction exists when both docs have signals AND come from
        # different categories AND are both evidential (not clean filler)
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
            s.env_feedback.append(f"CONTRADICTION: {doc_a_id} and {doc_b_id} conflict")
        else:
            s.env_feedback.append(f"No contradiction found between {doc_a_id} and {doc_b_id}")

    def _handle_flag_anomaly(self, action: AuditAction, rewards: dict):
        s = self._state
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
            rewards["false_positive"] = rewards.get("false_positive", 0) - 0.08
            s.reward_events.append(f"false_positive:{doc_id}")

        s.env_feedback.append(f"Anomaly recorded on '{doc_id}'")

    def _handle_external_confirmation(self, action: AuditAction, rewards: dict):
        s = self._state
        category = action.params.get("category")
        
        if isinstance(category, DocumentCategory):
            category = category.value

        if not hasattr(s, "external_confirmations_used"):
            s.external_confirmations_used = 0

        if s.external_confirmations_used >= self.config.max_external_confirmations:
            s.env_feedback.append("External confirmation limit reached")
            return

        s.external_confirmations_used += 1

        if not category:
            s.env_feedback.append("ERROR: request_external_confirmation requires 'category'")
            return

        # burn extra steps upfront — step() will add 1 more at end
        s.step = min(
            s.step + self.config.external_confirmation_cost - 1,
            s.max_steps
        )

        doc_ids_here = s.document_location_index.get(category, [])
        returned = []

        for doc_id in doc_ids_here:
            doc = s.document_index[doc_id]
            # destroyed is permanent — everything else can be bypassed
            if doc.status != DocumentStatus.destroyed:
                if doc_id not in s.received_doc_ids:
                    s.received_doc_ids.append(doc_id)
                doc.history.append(f"step {s.step}: obtained via external confirmation")
                returned.append(doc_id)

                if doc.status in (DocumentStatus.hidden, DocumentStatus.reclassified, DocumentStatus.delayed):
                    rewards["external_bypass"] = rewards.get("external_bypass", 0) + 0.15
                    s.reward_events.append(f"external_bypass:{doc_id}")

        s.env_feedback.append(
            f"External confirmation '{category}': {len(returned)} doc(s) obtained"
        )

    def _handle_submit_findings(self, action: AuditAction, rewards: dict):
        s = self._state
        s.findings_submitted = action.params
        s.env_feedback.append("Findings submitted — episode ending")

    # ─────────────────────────────────────────
    # CFO Adversary
    # ─────────────────────────────────────────

    def _cfo_reclassify(self, doc_id: str):
        s = self._state
        doc = s.document_index[doc_id]

        old_location = doc.location
        # remove from old location
        if old_location in s.document_location_index:
            if doc_id in s.document_location_index[old_location]:
                s.document_location_index[old_location].remove(doc_id)


        # new location
        doc.location = "misc_ops"
        doc.status = DocumentStatus.reclassified

        # add to new location
        s.document_location_index.setdefault(doc.location, []).append(doc_id)



        doc.history.append(f"step {s.step}: CFO reclassified from {old_location} → misc_ops")

        entry = f"step {s.step}: CFO reclassified '{doc_id}'"
        s.cfo_actions_current_round.append(entry)
        s.cfo_actions_history.append(entry)

    def _cfo_react(self, agent_action: AuditAction):
        s = self._state

        if s.cfo_strategy == "passive":
            return

        # budget resets every round
        budget = self.config.cfo_budget_per_round

        if s.cfo_strategy == "reactive":
            # only acts on the category the agent just touched
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
                    if random.random() < 0.5:
                        self._cfo_hide(doc_id)
                    else:
                        self._cfo_reclassify(doc_id)
                    budget -= 1

        elif s.cfo_strategy == "proactive":
            # always prioritises smoking gun docs regardless of what agent did
            for doc_id in s.critical_docs:
                if budget <= 0:
                    break
                doc = s.document_index[doc_id]
                if (
                    doc.evidence_type == EvidenceType.smoking_gun
                    and doc.status == DocumentStatus.available
                ):
                    if random.random() < 0.5:
                        self._cfo_hide(doc_id)
                    else:
                        self._cfo_reclassify(doc_id)
                    budget -= 1

    def _cfo_hide(self, doc_id: str):
        s = self._state
        doc = s.document_index[doc_id]
        doc.status = DocumentStatus.hidden
        doc.hidden_until_step = s.step + self.config.hide_duration_rounds
        doc.history.append(
            f"step {s.step}: CFO hid — expires step {doc.hidden_until_step}"
        )
        entry = f"step {s.step}: CFO hid '{doc_id}'"
        s.cfo_actions_current_round.append(entry)
        s.cfo_actions_history.append(entry)

    def _restore_expired_docs(self):
        s = self._state
        for doc_id, doc in s.document_index.items():
            if (
                doc.status == DocumentStatus.hidden
                and doc.hidden_until_step is not None
                and s.step >= doc.hidden_until_step
            ):
                doc.status = DocumentStatus.available
                doc.hidden_until_step = None
                doc.history.append(f"step {s.step}: auto-restored")
                s.env_feedback.append(f"'{doc_id}' is available again")

    # ─────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────

    def _build_observation(self) -> AuditObservation:
        s = self._state

        # only docs the agent has touched
        document_status = {
            doc_id: s.document_index[doc_id].status
            for doc_id in s.received_doc_ids
        }

        # what changed since last observation — core of the hard task mechanic
        document_status_delta = [
            f"{doc_id}: {s.last_known_status[doc_id].value} → {s.document_index[doc_id].status.value}"
            for doc_id in s.received_doc_ids
            if doc_id in s.last_known_status
            and s.document_index[doc_id].status != s.last_known_status[doc_id]
        ]

        # agent only sees CFO actions that touched docs it knows about
        cfo_visible_actions = [
            entry for entry in s.cfo_actions_current_round
            if any(doc_id in entry for doc_id in s.received_doc_ids)
        ]

        documents_received = {}
        for doc_id in s.received_doc_ids:
            doc = s.document_index[doc_id]
            anomaly_count = sum(1 for a in s.anomalies_flagged if a.startswith(doc_id))
            documents_received[doc_id] = DocumentSummary(
                id=doc_id,
                category=doc.category,
                status=doc.status,
                anomalies_flagged=anomaly_count,
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
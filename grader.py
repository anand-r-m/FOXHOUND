"""
grader.py — FOXHOUND Final Episode Grader
==========================================
Called once per episode, after the agent calls submit_findings (or the step
budget is exhausted).  Reads the full AuditState that the agent never had
access to and produces a deterministic RewardInfo strictly in (0, 1).

Usage (from server or demo):
    from grader import grade_submission
    reward_info = grade_submission(env.state())
    print(reward_info.total)          # strictly between 0 and 1
    print(reward_info.components)     # per-component breakdown
    print(reward_info.events)         # human-readable audit trail

Design principles
-----------------
- Every weight and threshold is a named constant at the top — easy to tune.
- Components map directly to the YAML spec in the roadmap.
- Cross-reference quality is inferred from the evidence chain (the agent's
  cross-reference history isn't stored in state, so we proxy it from what
  the agent PUT into its evidence chain — if they connected dots, the chain
  will span multiple categories and evidence types).
- Total is clamped to (0, 1) via models.clamp_task_score — never 0.0 or 1.0.
"""

from __future__ import annotations

import math
from typing import Any

from models import AuditState, EvidenceType, FraudType, RewardInfo, clamp_task_score


def _normalize_evidence_chain(raw: Any) -> list[str]:
    """
    Params sometimes arrive as a list of ids, a single id string, or missing.
    A bare string must become one id — otherwise Python iterates characters.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _normalize_obstruction_notes(raw: Any) -> list[str]:
    """Notes should be a list; a single string is treated as one note (not N characters)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _normalize_fraud_type_value(raw: Any) -> str:
    """Accept enum or string so API/Python clients both compare correctly."""
    if raw is None:
        return ""
    if isinstance(raw, FraudType):
        return raw.value
    return str(raw).strip()

# ─────────────────────────────────────────────────────────────────────────────
# Weight constants — all numbers live here, nowhere else
# Max achievable score if you add only the positive weights:
#   0.08 + 0.12 + 0.08 + 0.35 + 0.15 + 0.08 + 0.07 + 0.05  = 0.98
# (contradiction_found and weak_link_found are mutually exclusive, so the real
#  ceiling is ~0.98 which is then capped at 1.0 anyway.)
# ─────────────────────────────────────────────────────────────────────────────

# Document requesting
W_USEFUL_REQUESTS        =  0.08   # max reward for requesting productive categories
W_REDUNDANT_PER_EMPTY    = -0.03   # per empty-category request (no docs exist there)

# Cross-referencing (inferred from evidence chain)
W_CONTRADICTION_FULL     =  0.12   # smoking gun + multi-category + circumstantial in chain
W_CONTRADICTION_PARTIAL  =  0.06   # smoking gun + at least one other category in chain
W_WEAK_LINK              =  0.04   # multi-category evidence received (no smoking gun found)
W_USELESS_CHAIN          = -0.05   # evidence chain submitted but contains ONLY clean docs

# Anomaly flagging
W_CORRECT_FLAG_RECALL    =  0.08   # scaled by (correct_flags / total_evidence_docs)
W_FALSE_POSITIVE_PER     = -0.05   # per clean-doc anomaly flag

# Findings submission
W_FRAUD_TYPE_CORRECT     =  0.35   # biggest single weight — identifying the fraud type
W_EVIDENCE_COMPLETENESS  =  0.15   # scaled by (correct_chain_docs / total_evidence_docs)
W_NO_EVIDENCE_PENALTY    = -0.05   # submitted findings with an empty evidence chain

# External confirmation
W_EXT_BYPASSED_PER_DOC   =  0.04   # per concealed doc retrieved via external confirmation
W_EXT_MAX                =  0.08   # ceiling on external confirmation reward
W_EXT_WASTE_PER          = -0.02   # per confirmation used that retrieved nothing concealed
W_EXT_MISSED_HARD        = -0.03   # did not use it at all on hard difficulty

# Obstruction detection (hard only)
W_OBSTRUCTION            =  0.07   # scaled by (notes / cfo_actions), hard only
W_OBSTRUCTION_FREE       =  0.05   # free points on easy / medium

# Efficiency bonus (only awarded on correct fraud type)
W_EFFICIENCY             =  0.05   # scaled by (remaining_steps / max_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def grade_submission(state: AuditState) -> RewardInfo:
    """
    Compute the final grade for a completed episode.

    Parameters
    ----------
    state : AuditState
        The env's internal state at episode end (call env.state()).

    Returns
    -------
    RewardInfo
        .total      — float strictly in (0, 1)
        .components — dict matching the YAML reward spec
        .events     — human-readable explanation of scoring decisions
    """
    s = state
    submission: dict[str, Any] = s.findings_submitted or {}
    components: dict[str, float] = {}
    events: list[str] = []

    # ── Pre-compute shared lookups used by multiple components ────────────────

    # All non-clean docs in the game (ground truth)
    evidence_docs = {
        doc_id: doc
        for doc_id, doc in s.document_index.items()
        if doc.evidence_type != EvidenceType.clean
    }
    total_evidence = len(evidence_docs)

    evidence_chain = _normalize_evidence_chain(submission.get("evidence_chain"))

    # Chain docs that the agent DID include and that are actually evidence
    chain_evidence_docs = [
        s.document_index[doc_id]
        for doc_id in evidence_chain
        if doc_id in s.document_index
        and s.document_index[doc_id].evidence_type != EvidenceType.clean
    ]

    # Chain docs that the agent included (whether evidence or not)
    chain_all_docs = [
        s.document_index[doc_id]
        for doc_id in evidence_chain
        if doc_id in s.document_index
    ]

    fraud_type_submitted = _normalize_fraud_type_value(submission.get("fraud_type"))
    fraud_correct: bool = fraud_type_submitted == s.true_fraud_type.value

    # ── 1. DOCUMENT REQUESTING ────────────────────────────────────────────────
    # Measures how purposefully the agent explored the document space.

    # Categories that held at least one evidence doc (regardless of CFO action)
    productive_categories = {
        doc.category.value
        for doc in evidence_docs.values()
    }

    if s.requested_categories:
        useful = [c for c in s.requested_categories if c in productive_categories]
        useful_ratio = len(useful) / len(s.requested_categories)
        components["useful_document_requests"] = round(useful_ratio * W_USEFUL_REQUESTS, 4)

        # Empty categories = categories the agent requested that have NO documents at all
        # (separate from "docs exist but are hidden" — that's useful intel, not a waste)
        empty = [
            c for c in s.requested_categories
            if not s.document_location_index.get(c)  # no docs registered at this location
        ]
        components["redundant_document_requests"] = round(W_REDUNDANT_PER_EMPTY * len(empty), 4)
        if empty:
            events.append(f"redundant_document_requests: empty categories requested = {empty}")
    else:
        # Agent never requested anything — penalize
        components["useful_document_requests"] = 0.0
        components["redundant_document_requests"] = W_REDUNDANT_PER_EMPTY
        events.append("redundant_document_requests: no categories were requested at all")

    # ── 2. CROSS-REFERENCING ──────────────────────────────────────────────────
    # The agent's raw cross_reference actions aren't stored in AuditState, so
    # we infer cross-reference quality from the evidence chain:
    # — if the chain spans multiple categories AND includes a smoking gun,
    #   the agent must have connected evidence across document types.
    # — weaker signal: multiple categories of evidence received (even if chain
    #   doesn't include the smoking gun).

    chain_categories = {d.category for d in chain_evidence_docs}
    chain_evidence_types = {d.evidence_type for d in chain_evidence_docs}
    has_smoking_gun_in_chain = EvidenceType.smoking_gun in chain_evidence_types
    has_circumstantial_in_chain = EvidenceType.circumstantial in chain_evidence_types
    has_multi_category_in_chain = len(chain_categories) >= 2

    # Received docs with evidence (what the agent *could* have cross-referenced)
    received_evidence_categories = {
        s.document_index[doc_id].category
        for doc_id in s.received_doc_ids
        if doc_id in evidence_docs
    }

    if has_smoking_gun_in_chain and has_multi_category_in_chain and has_circumstantial_in_chain:
        # Full contradiction: smoking gun corroborated by circumstantial from a different category
        components["contradiction_found"] = W_CONTRADICTION_FULL
        events.append(
            "contradiction_found: chain has smoking gun + circumstantial across "
            f"{len(chain_categories)} categories"
        )
        components["weak_link_found"] = 0.0  # don't double-count

    elif has_smoking_gun_in_chain and has_multi_category_in_chain:
        # Partial: found the smoking gun and linked it to at least one other category
        components["contradiction_found"] = W_CONTRADICTION_PARTIAL
        events.append(
            "contradiction_found (partial): smoking gun + one other category in chain"
        )
        components["weak_link_found"] = 0.0

    else:
        components["contradiction_found"] = 0.0

        # Weak link: agent received evidence from ≥2 categories, even if chain is incomplete
        if len(received_evidence_categories) >= 2:
            components["weak_link_found"] = W_WEAK_LINK
            events.append(
                f"weak_link_found: evidence received from {len(received_evidence_categories)} "
                "categories (no smoking gun in chain)"
            )
        else:
            components["weak_link_found"] = 0.0

    # Useless chain: agent submitted an evidence chain but every doc in it is clean
    chain_is_noisy = bool(chain_all_docs) and all(
        d.evidence_type == EvidenceType.clean for d in chain_all_docs
    )
    if chain_is_noisy:
        components["useless_cross_reference"] = W_USELESS_CHAIN
        events.append("useless_cross_reference: submitted chain contains only clean documents")
    else:
        components["useless_cross_reference"] = 0.0

    # ── 3. ANOMALY FLAGGING ───────────────────────────────────────────────────
    # anomalies_flagged entries are stored as "doc_id: description"

    correct_flags = 0
    false_positives = 0

    for flag in s.anomalies_flagged:
        # extract doc_id from "doc_id: description" format
        doc_id = flag.split(":")[0].strip()
        if doc_id in s.document_index:
            if s.document_index[doc_id].evidence_type != EvidenceType.clean:
                correct_flags += 1
            else:
                false_positives += 1

    if total_evidence > 0:
        flag_recall = correct_flags / total_evidence
        components["correct_anomaly_flag"] = round(flag_recall * W_CORRECT_FLAG_RECALL, 4)
    else:
        components["correct_anomaly_flag"] = 0.0

    components["false_positive_anomaly_flag"] = round(W_FALSE_POSITIVE_PER * false_positives, 4)

    if correct_flags:
        events.append(f"correct_anomaly_flag: {correct_flags}/{total_evidence} evidence docs flagged")
    if false_positives:
        events.append(f"false_positive_anomaly_flag: {false_positives} clean doc(s) incorrectly flagged")

    # ── 4. FINDINGS SUBMISSION ────────────────────────────────────────────────
    # The most important component.  Fraud type identification is 35% on its own.

    # 4a. Fraud type
    if fraud_correct:
        components["submit_findings"] = W_FRAUD_TYPE_CORRECT
        events.append(f"submit_findings: correct fraud type ({fraud_type_submitted})")
    else:
        components["submit_findings"] = 0.0
        events.append(
            f"submit_findings: WRONG fraud type "
            f"(submitted={fraud_type_submitted!r}, true={s.true_fraud_type.value!r})"
        )

    # 4b. Evidence completeness within the chain
    if total_evidence > 0:
        completeness = len(chain_evidence_docs) / total_evidence
        components["submit_findings_evidence_completeness"] = round(
            completeness * W_EVIDENCE_COMPLETENESS, 4
        )
        events.append(
            f"submit_findings_evidence_completeness: "
            f"{len(chain_evidence_docs)}/{total_evidence} evidence docs in chain"
        )
    else:
        # No evidence docs exist in this episode — nothing to be "complete" about
        components["submit_findings_evidence_completeness"] = 0.0
        events.append(
            "submit_findings_evidence_completeness: no evidence docs in episode (0.0)"
        )

    # 4c. Penalty for submitting with no evidence at all
    if not evidence_chain:
        components["submit_findings_no_evidence_penalty"] = W_NO_EVIDENCE_PENALTY
        events.append("submit_findings_no_evidence_penalty: empty evidence chain submitted")
    else:
        components["submit_findings_no_evidence_penalty"] = 0.0

    # ── 5. EXTERNAL CONFIRMATION ──────────────────────────────────────────────
    # Rewards smart use of the costly bypass action.
    # "Smart use" = it retrieved a doc that was actively concealed at the time.
    # We detect this by checking each received doc's history for the confirmation marker.

    ext_used = s.external_confirmations_used

    if ext_used == 0:
        if s.difficulty == "hard":
            # On hard the CFO proactively hides smoking guns — not using external
            # confirmation at all is a meaningful missed opportunity
            components["request_external_confirmation"] = W_EXT_MISSED_HARD
            events.append(
                "request_external_confirmation: not used on hard difficulty "
                "(CFO was proactively concealing)"
            )
        else:
            # Easy / medium: passive or reactive CFO — not using it is fine
            components["request_external_confirmation"] = 0.0
    else:
        # Count concealed evidence docs the agent retrieved via external confirmation
        # env.py stamps history with "obtained via external confirmation" when this happens
        concealed_retrieved = [
            doc_id for doc_id in s.received_doc_ids
            if doc_id in evidence_docs
            and any(
                "via external confirmation" in entry
                for entry in s.document_index[doc_id].history
            )
        ]

        if concealed_retrieved:
            reward = min(W_EXT_BYPASSED_PER_DOC * len(concealed_retrieved), W_EXT_MAX)
            components["request_external_confirmation"] = round(reward, 4)
            events.append(
                f"request_external_confirmation: retrieved {len(concealed_retrieved)} "
                "concealed evidence doc(s)"
            )
        else:
            # Used but got no concealed evidence — wasteful spend of step budget
            components["request_external_confirmation"] = round(W_EXT_WASTE_PER * ext_used, 4)
            events.append(
                f"request_external_confirmation: {ext_used} confirmation(s) used "
                "but no concealed evidence retrieved"
            )

    # ── 6. OBSTRUCTION DETECTION ──────────────────────────────────────────────
    # Did the agent notice and document CFO interference?
    # Only scored on hard (where CFO was proactive).  Free points on easy/medium.

    obstruction_notes = _normalize_obstruction_notes(submission.get("obstruction_notes"))
    cfo_actions = s.cfo_actions_history  # cumulative across all steps

    if s.difficulty == "hard" and cfo_actions:
        if obstruction_notes:
            detection_ratio = min(len(obstruction_notes) / max(len(cfo_actions), 1), 1.0)
            components["obstruction_detection"] = round(detection_ratio * W_OBSTRUCTION, 4)
            events.append(
                f"obstruction_detection: {len(obstruction_notes)} note(s) vs "
                f"{len(cfo_actions)} CFO action(s)"
            )
        else:
            components["obstruction_detection"] = 0.0
            events.append(
                "obstruction_detection: CFO was active but agent reported no obstruction"
            )
    else:
        # Easy: no CFO activity.  Medium: reactive, low impact.  Both get free points.
        components["obstruction_detection"] = W_OBSTRUCTION_FREE

    # ── 7. INVESTIGATION EFFICIENCY ───────────────────────────────────────────
    # Small bonus for solving correctly without burning the whole step budget.
    # Only awarded when fraud type is correct — we don't reward quitting early
    # with a wrong answer.

    steps_used = s.step
    if s.max_steps > 0:
        step_efficiency = max(0.0, 1.0 - (steps_used / s.max_steps))
    else:
        step_efficiency = 0.0

    if fraud_correct and step_efficiency > 0:
        components["investigation_efficiency"] = round(step_efficiency * W_EFFICIENCY, 4)
        events.append(
            f"investigation_efficiency: {s.max_steps - steps_used}/{s.max_steps} "
            "steps remaining"
        )
    else:
        components["investigation_efficiency"] = 0.0

    # ── TOTAL ─────────────────────────────────────────────────────────────────

    raw_total = sum(components.values())
    _safe = raw_total if math.isfinite(raw_total) else 0.01
    total = round(min(max(_safe, 0.01), 0.99), 3)

    return RewardInfo(total=total, components=components, events=events)


def final_grade(state: AuditState) -> float:
    """Convenience wrapper — returns just the float score."""
    return grade_submission(state).total
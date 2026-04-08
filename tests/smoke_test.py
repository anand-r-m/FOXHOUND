from env import ForensicAuditEnv
from models import AuditAction, ActionType, TaskConfig, FraudType
from grader import grade_submission


def _make_easy_config():
    return TaskConfig(
        task_id="easy",
        difficulty="easy",
        fraud_type=FraudType.channel_stuffing,
        company_seed=42,
        max_steps=10,
        cfo_strategy="passive",
        cfo_budget_per_round=0,
        hide_duration_rounds=2,
        smoking_gun_count=1,
        circumstantial_count=3,
        clean_count=6,
        max_external_confirmations=2,
        external_confirmation_cost=2,
    )


def test_reset_returns_valid_observation():
    env = ForensicAuditEnv(_make_easy_config())
    obs = env.reset()

    assert obs.step == 0
    assert obs.done is False
    assert isinstance(obs.documents_received, dict)


def test_request_category_returns_docs():
    env = ForensicAuditEnv(_make_easy_config())
    env.reset()

    obs, reward, done, _ = env.step(
        AuditAction(
            action_type=ActionType.request_category,
            params={"category": "invoices"},
        )
    )

    assert isinstance(obs.documents_received, dict)
    assert done is False


def test_submit_findings_ends_episode():
    env = ForensicAuditEnv(_make_easy_config())
    env.reset()

    obs, reward, done, _ = env.step(
        AuditAction(
            action_type=ActionType.submit_findings,
            params={
                "fraud_type": "channel_stuffing",
                "evidence_chain": [],
            },
        )
    )

    assert done is True


def test_full_episode_runs_without_crashing():
    env = ForensicAuditEnv(_make_easy_config())
    obs = env.reset()

    done = False
    steps = 0

    while not done and steps < 20:
        action = AuditAction(
            action_type=ActionType.request_category,
            params={"category": "financial_statements"},
        )
        obs, reward, done, _ = env.step(action)
        steps += 1

    assert steps > 0


def test_grader_perfect_fraud_type():
    env = ForensicAuditEnv(_make_easy_config())
    env.reset()

    env.step(
        AuditAction(
            action_type=ActionType.submit_findings,
            params={
                "fraud_type": "channel_stuffing",
                "evidence_chain": ["inv_q3_early"],
            },
        )
    )

    result = grade_submission(env.state())

    assert result.total > 0.3  # fraud type alone gives big chunk


def test_grader_wrong_fraud_type():
    env = ForensicAuditEnv(_make_easy_config())
    env.reset()

    env.step(
        AuditAction(
            action_type=ActionType.submit_findings,
            params={
                "fraud_type": "phantom_revenue",
                "evidence_chain": [],
            },
        )
    )

    result = grade_submission(env.state())

    assert result.total < 0.5
"""Integration test: reasoning-example.yaml checks execute correctly.

Catches config key mismatch bugs (numbered vs unnumbered keys) and
version gating bugs (sanna_constitution vs version field).  Loads the
actual example constitution shipped with the repo and verifies that
the reasoning pipeline initializes and executes as expected.
"""

from pathlib import Path

import pytest
import yaml

from sanna.constitution import load_constitution, parse_constitution
from sanna.reasoning.evaluator import ReasoningEvaluator


# Path relative to repo root
EXAMPLE_PATH = Path(__file__).parent.parent / "examples" / "constitutions" / "reasoning-example.yaml"


# =============================================================================
# SHIPPED EXAMPLE CONSTITUTION
# =============================================================================


class TestReasoningExampleConfig:
    """Verify reasoning-example.yaml enables checks correctly."""

    @pytest.mark.asyncio
    async def test_example_yaml_exists(self):
        """Shipped example file is present in the repo."""
        assert EXAMPLE_PATH.exists(), (
            f"reasoning-example.yaml not found at {EXAMPLE_PATH}"
        )

    @pytest.mark.asyncio
    async def test_reasoning_parsed_from_example(self):
        """CRITICAL: reasoning section is parsed (catches version gating bug)."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        assert constitution.reasoning is not None, (
            "Reasoning section not parsed — check sanna_constitution version gating"
        )

    @pytest.mark.asyncio
    async def test_example_has_correct_schema_version(self):
        """Example uses sanna_constitution: '1.1' (not version:)."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        with open(EXAMPLE_PATH) as f:
            data = yaml.safe_load(f)

        assert data.get("sanna_constitution") == "1.1", (
            "Example must set sanna_constitution: '1.1' to enable reasoning"
        )

    @pytest.mark.asyncio
    async def test_pipeline_enabled_with_three_deterministic_checks(self):
        """Pipeline has 3 deterministic checks: glc_001, glc_002, glc_003."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        evaluator = ReasoningEvaluator(constitution)
        pipeline = evaluator.pipeline

        assert pipeline.enabled is True
        assert len(pipeline.checks) == 3

        check_ids = [c.check_id() for c in pipeline.checks]
        assert check_ids == [
            "glc_001_justification_present",
            "glc_002_minimum_substance",
            "glc_003_no_parroting",
        ]

    @pytest.mark.asyncio
    async def test_llm_check_configured(self):
        """glc_005 LLM coherence check is configured (may not initialize without httpx)."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        reasoning = constitution.reasoning

        # Verify glc_005 is present in the parsed config
        assert "glc_005_llm_coherence" in reasoning.checks
        llm_config = reasoning.checks["glc_005_llm_coherence"]
        assert llm_config.enabled is True
        assert "must_escalate" in llm_config.enabled_for

    @pytest.mark.asyncio
    async def test_substance_check_fails_short_justification(self):
        """min_length=20 config applied: 'ok' fails substance check."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        evaluator = ReasoningEvaluator(constitution)

        result = await evaluator.evaluate(
            tool_name="delete_database",
            args={"id": 123, "_justification": "ok"},
            enforcement_level="must_escalate",
        )

        assert result.passed is False
        assert result.overall_score == 0.0

        # Find the substance check
        substance = [
            c for c in result.checks
            if c.check_id == "glc_002_minimum_substance"
        ]
        assert len(substance) == 1
        assert substance[0].passed is False

    @pytest.mark.asyncio
    async def test_parroting_check_catches_blocklist(self):
        """Blocklist from example catches 'because you asked'."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        evaluator = ReasoningEvaluator(constitution)

        result = await evaluator.evaluate(
            tool_name="delete_database",
            args={
                "id": 123,
                "_justification": "I am doing this because you asked me to do it",
            },
            enforcement_level="must_escalate",
        )

        assert result.passed is False

        parroting = [
            c for c in result.checks
            if c.check_id == "glc_003_no_parroting"
        ]
        assert len(parroting) == 1
        assert parroting[0].passed is False

    @pytest.mark.asyncio
    async def test_valid_justification_passes_all_deterministic(self):
        """Good justification passes all 3 deterministic checks."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        evaluator = ReasoningEvaluator(constitution)

        result = await evaluator.evaluate(
            tool_name="delete_database",
            args={
                "id": 123,
                "_justification": (
                    "Removing stale test data per retention policy"
                ),
            },
            enforcement_level="must_escalate",
        )

        # All deterministic checks pass (LLM skipped without httpx/API key)
        for check in result.checks:
            assert check.passed is True, (
                f"{check.check_id} failed unexpectedly"
            )

    @pytest.mark.asyncio
    async def test_on_missing_justification_is_block(self):
        """Example config uses on_missing_justification: block."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        reasoning = constitution.reasoning

        assert reasoning.on_missing_justification == "block"
        assert reasoning.on_check_error == "block"

    @pytest.mark.asyncio
    async def test_missing_justification_for_must_escalate(self):
        """Missing justification on must_escalate level returns failure."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        evaluator = ReasoningEvaluator(constitution)

        result = await evaluator.evaluate(
            tool_name="delete_database",
            args={"id": 123},
            enforcement_level="must_escalate",
        )

        assert result.passed is False
        assert result.failure_reason == "missing_required_justification"

    @pytest.mark.asyncio
    async def test_require_justification_for_levels(self):
        """require_justification_for includes must_escalate and cannot_execute."""
        if not EXAMPLE_PATH.exists():
            pytest.skip("reasoning-example.yaml not found")

        constitution = load_constitution(EXAMPLE_PATH)
        reasoning = constitution.reasoning

        assert "must_escalate" in reasoning.require_justification_for
        assert "cannot_execute" in reasoning.require_justification_for


# =============================================================================
# KEY STYLE COMPATIBILITY
# =============================================================================


class TestKeyStyleCompatibility:
    """Both numbered and unnumbered check keys parse and execute."""

    def _make_constitution(self, checks_block: dict) -> object:
        """Create a minimal constitution with given checks config."""
        return parse_constitution({
            "sanna_constitution": "1.1",
            "identity": {"agent_name": "test", "domain": "testing"},
            "provenance": {
                "authored_by": "dev@test.com",
                "approved_by": ["approver@test.com"],
                "approval_date": "2026-01-01",
                "approval_method": "manual-sign-off",
            },
            "boundaries": [
                {
                    "id": "B001",
                    "description": "Test",
                    "category": "scope",
                    "severity": "high",
                },
            ],
            "version": "1.1",
            "reasoning": {
                "require_justification_for": ["must_escalate"],
                "on_missing_justification": "block",
                "on_check_error": "block",
                "checks": checks_block,
            },
        })

    @pytest.mark.asyncio
    async def test_numbered_keys_work(self):
        """Numbered keys (glc_002_minimum_substance) parse and execute."""
        constitution = self._make_constitution({
            "glc_002_minimum_substance": {"enabled": True, "min_length": 30},
            "glc_003_no_parroting": {"enabled": True},
        })

        evaluator = ReasoningEvaluator(constitution)
        pipeline = evaluator.pipeline

        assert pipeline.enabled is True
        assert len(pipeline.checks) == 3  # glc_001 + glc_002 + glc_003

        result = await evaluator.evaluate(
            tool_name="test",
            args={"_justification": "twenty chars but not thirty"},
            enforcement_level="must_escalate",
        )

        # "twenty chars but not thirty" is 28 chars, fails min_length=30
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_unnumbered_keys_work(self):
        """Unnumbered keys (glc_minimum_substance) parse and execute."""
        constitution = self._make_constitution({
            "glc_minimum_substance": {"enabled": True, "min_length": 30},
            "glc_no_parroting": {"enabled": True},
        })

        evaluator = ReasoningEvaluator(constitution)
        pipeline = evaluator.pipeline

        assert pipeline.enabled is True
        assert len(pipeline.checks) == 3

        result = await evaluator.evaluate(
            tool_name="test",
            args={"_justification": "twenty chars but not thirty"},
            enforcement_level="must_escalate",
        )

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_min_length_config_preserved_after_normalization(self):
        """min_length value survives key normalization (unnumbered → numbered)."""
        constitution = self._make_constitution({
            "glc_minimum_substance": {"enabled": True, "min_length": 50},
        })

        evaluator = ReasoningEvaluator(constitution)

        # "This justification is exactly forty-three chars" = 48 chars
        # Should fail with min_length=50
        result = await evaluator.evaluate(
            tool_name="test",
            args={
                "_justification": "This justification is exactly forty-three chars!",
            },
            enforcement_level="must_escalate",
        )

        assert result.passed is False

        # Justification long enough
        result2 = await evaluator.evaluate(
            tool_name="test",
            args={
                "_justification": (
                    "This justification is well over fifty characters long "
                    "so it should pass the minimum substance check"
                ),
            },
            enforcement_level="must_escalate",
        )

        assert result2.passed is True

    @pytest.mark.asyncio
    async def test_blocklist_config_preserved_after_normalization(self):
        """Custom blocklist survives key normalization."""
        constitution = self._make_constitution({
            "glc_002_minimum_substance": {"enabled": True, "min_length": 20},
            "glc_no_parroting": {
                "enabled": True,
                "blocklist": ["custom forbidden phrase"],
            },
        })

        evaluator = ReasoningEvaluator(constitution)

        result = await evaluator.evaluate(
            tool_name="test",
            args={
                "_justification": (
                    "I am using a custom forbidden phrase in my reasoning"
                ),
            },
            enforcement_level="must_escalate",
        )

        assert result.passed is False

        parroting = [
            c for c in result.checks
            if c.check_id == "glc_003_no_parroting"
        ]
        assert len(parroting) == 1
        assert parroting[0].passed is False

"""
Sanna C1-C5 heuristic stress tests — 53 diverse test cases.

Tests the keyword/pattern-based coherence checks against scenarios they
SHOULD catch, scenarios they SHOULD pass, and scenarios that EXPOSE
known heuristic limitations (marked with @pytest.mark.xfail).

Each check function takes (context: str, output: str) -> CheckResult
where CheckResult.passed is the boolean we assert against.
"""

import pytest

from sanna.receipt import (
    check_c1_context_contradiction,
    check_c2_unmarked_inference,
    check_c3_false_certainty,
    check_c4_conflict_collapse,
    check_c5_premature_compression,
)


# =============================================================================
# C1: Context Contradiction
# =============================================================================

class TestC1ContextContradiction:
    """
    C1 heuristic patterns:
    - "non-refundable" in context + "eligible" AND "refund" in output
    - "digital products are non-refundable" in context + ("software"|"digital")
      in output + ("eligible"|"can get"|"you are able") in output
    """

    # --- SHOULD FAIL (heuristic catches these) ---

    def test_01_direct_refund_contradiction(self):
        """Non-refundable context + 'eligible' + 'refund' in output."""
        context = "All sales are final and non-refundable."
        output = "You are eligible for a refund within 30 days."
        result = check_c1_context_contradiction(context, output)
        assert not result.passed, "C1 should catch refund contradiction"
        assert result.severity == "critical"

    def test_02_digital_products_non_refundable_software_eligible(self):
        """'digital products are non-refundable' + 'software' + 'eligible'."""
        context = (
            "Our store policy states that digital products are non-refundable "
            "once the download link has been activated."
        )
        output = (
            "Since your software purchase was made recently, you are eligible "
            "for a full refund."
        )
        result = check_c1_context_contradiction(context, output)
        assert not result.passed, "C1 should catch digital product contradiction"

    def test_03_non_refundable_can_get_refund(self):
        """Non-refundable context + 'can get' a refund."""
        context = (
            "Digital products are non-refundable. "
            "Physical items may be returned within 14 days."
        )
        output = "You can get a refund for your digital download."
        result = check_c1_context_contradiction(context, output)
        assert not result.passed, "C1 should catch 'can get' refund contradiction"

    def test_04_non_refundable_you_are_able(self):
        """Non-refundable context + 'you are able' to get a refund."""
        context = (
            "Digital products are non-refundable per our terms of service."
        )
        output = (
            "You are able to receive a refund for the digital item you purchased."
        )
        result = check_c1_context_contradiction(context, output)
        assert not result.passed, "C1 should catch 'you are able' refund contradiction"

    # --- SHOULD PASS (no contradiction) ---

    def test_05_context_says_refundable_output_agrees(self):
        """Context says refundable, output confirms — no contradiction."""
        context = "All products are refundable within 30 days of purchase."
        output = "You are eligible for a refund since your purchase was 10 days ago."
        result = check_c1_context_contradiction(context, output)
        assert result.passed, "No contradiction when context says refundable"

    def test_06_output_correctly_summarizes_non_refundable(self):
        """Output restates non-refundable policy without contradicting."""
        context = "All digital products are non-refundable once downloaded."
        output = (
            "Unfortunately, digital products are non-refundable per our policy. "
            "We cannot process a refund for your download."
        )
        result = check_c1_context_contradiction(context, output)
        assert result.passed, "Output correctly restates policy"

    def test_07_empty_context_reasonable_output(self):
        """Empty context should always pass (insufficient data)."""
        context = ""
        output = "You may be eligible for a refund."
        result = check_c1_context_contradiction(context, output)
        assert result.passed, "Empty context should pass C1"

    def test_08_both_agree_no_contradiction(self):
        """Context and output are in full agreement."""
        context = "Premium members can access all courses for free."
        output = "As a premium member, you have free access to all courses."
        result = check_c1_context_contradiction(context, output)
        assert result.passed, "Agreeing context and output should pass"

    # --- SHOULD FAIL BUT WON'T (xfail — heuristic limitations) ---

    @pytest.mark.xfail(
        reason="Known heuristic limitation: C1 does not detect numeric contradictions"
    )
    def test_09_numeric_contradiction(self):
        """Context says $50, output says $30 — heuristic doesn't catch numbers."""
        context = "The annual subscription costs $50 per year."
        output = "Your annual subscription costs $30 per year."
        result = check_c1_context_contradiction(context, output)
        assert not result.passed

    @pytest.mark.xfail(
        reason="Known heuristic limitation: C1 does not detect temporal contradictions"
    )
    def test_10_temporal_contradiction(self):
        """Context says March, output says April — not in pattern list."""
        context = "The project deadline is March 15, 2025."
        output = "Your project is due in April 2025."
        result = check_c1_context_contradiction(context, output)
        assert not result.passed

    @pytest.mark.xfail(
        reason="Known heuristic limitation: C1 does not detect negation flips outside refund patterns"
    )
    def test_11_negation_flip(self):
        """'not recommended' flipped to 'we recommend' — not a refund pattern."""
        context = "This medication is not recommended for children under 12."
        output = "We recommend this medication for all ages including children."
        result = check_c1_context_contradiction(context, output)
        assert not result.passed

    @pytest.mark.xfail(
        reason="Known heuristic limitation: C1 does not detect scope contradictions"
    )
    def test_12_scope_contradiction(self):
        """'only applies to US' vs 'available globally' — not in patterns."""
        context = "This offer only applies to customers in the United States."
        output = "This offer is available globally to all customers."
        result = check_c1_context_contradiction(context, output)
        assert not result.passed


# =============================================================================
# C2: Mark Inferences (Unmarked Inference)
# =============================================================================

class TestC2UnmarkedInference:
    """
    C2 heuristic patterns:
    - Definitive words: definitely, certainly, always, never, guaranteed,
      absolutely, without doubt, 100%
    - Hedge words: may, might, could, possibly, likely, appears, seems,
      suggest, indicate, probably, perhaps, it's possible
    - Fails when: has_definitive AND NOT has_hedging
    """

    # --- SHOULD FAIL (definitive language without hedging) ---

    def test_13_definitely_no_hedge(self):
        """'definitely' with no hedging language."""
        context = "Some users report positive outcomes."
        output = "This treatment will definitely cure your condition."
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed, "C2 should catch 'definitely' without hedge"
        assert result.severity == "warning"

    def test_14_guaranteed_no_hedge(self):
        """'guaranteed' with no hedging language."""
        context = "The product has received mixed reviews."
        output = "Results are guaranteed within the first week of use."
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed, "C2 should catch 'guaranteed' without hedge"

    def test_15_certainly_and_always(self):
        """Multiple definitive words, no hedging."""
        context = "Market conditions vary by region."
        output = "This stock will certainly go up. It always outperforms the market."
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed, "C2 should catch 'certainly' + 'always'"

    def test_16_absolutely_and_without_doubt(self):
        """'absolutely' and 'without doubt', no hedging."""
        context = "Researchers are studying the effects."
        output = (
            "This is absolutely the best choice without doubt. "
            "No other option comes close."
        )
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed, "C2 should catch 'absolutely' + 'without doubt'"

    # --- SHOULD PASS (hedged or no definitive language) ---

    def test_17_definitely_with_hedge(self):
        """'definitely' present but 'may' also present — should pass."""
        context = "Some studies show positive results."
        output = (
            "This approach is definitely popular, though results may vary "
            "depending on individual circumstances."
        )
        result = check_c2_unmarked_inference(context, output)
        assert result.passed, "C2 should pass when definitive is paired with hedge"

    def test_18_only_hedges_no_definitive(self):
        """Only hedge words, no definitive claims."""
        context = "Early trials show mixed results."
        output = (
            "Results will probably be positive but likely depend on dosage. "
            "Individual outcomes may differ."
        )
        result = check_c2_unmarked_inference(context, output)
        assert result.passed, "C2 should pass when only hedges are used"

    def test_19_no_definitive_language(self):
        """Neutral language with no definitive or hedge words."""
        context = "The company was founded in 2010."
        output = "The company has been operating since 2010."
        result = check_c2_unmarked_inference(context, output)
        assert result.passed, "C2 should pass with neutral language"

    def test_20_could_and_possibly(self):
        """Properly hedged speculative statement."""
        context = "Limited data is available on long-term effects."
        output = (
            "Long-term effects could include minor side effects. "
            "It is possibly related to dosage levels."
        )
        result = check_c2_unmarked_inference(context, output)
        assert result.passed, "C2 should pass with 'could' and 'possibly'"

    # --- SHOULD FAIL BUT WON'T (xfail — heuristic limitations) ---

    @pytest.mark.xfail(
        reason=(
            "Known heuristic limitation: C2 only checks for specific definitive keywords. "
            "'The system will crash tomorrow' is a speculative factual claim with no "
            "definitive keyword."
        )
    )
    def test_21_speculative_fact_no_keywords(self):
        """Stated as fact, speculative, but no definitive keywords."""
        context = "System performance has been inconsistent."
        output = "The system will crash tomorrow at 3pm."
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed

    @pytest.mark.xfail(
        reason=(
            "Known heuristic limitation: C2 does not detect unsupported factual claims "
            "when no definitive keywords are used. 'Users prefer dark mode' sounds "
            "authoritative but lacks any of the trigger words."
        )
    )
    def test_22_unsupported_claim_no_keywords(self):
        """Stated as fact without evidence, but no definitive keywords."""
        context = "We conducted a survey with 50 participants."
        output = "Users prefer dark mode over light mode."
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed


# =============================================================================
# C3: No False Certainty
# =============================================================================

class TestC3FalseCertainty:
    """
    C3 heuristic patterns:
    - Conditional markers in context: if, unless, except, however, but, require
    - Confidence markers in output: you are eligible, you can, you will, go ahead and
    - Acknowledgment markers in output: however, but, note that, keep in mind, require
    - Fails when: context_has_conditions AND output_is_confident AND NOT acknowledges
    """

    # --- SHOULD FAIL (confident output ignoring context conditions) ---

    def test_23_if_condition_ignored(self):
        """Context has 'if' condition, output says 'you are eligible' without ack."""
        context = "If you have been a member for over 1 year, you qualify for the discount."
        output = "You are eligible for the discount. Apply it at checkout."
        result = check_c3_false_certainty(context, output)
        assert not result.passed, "C3 should catch confident claim ignoring 'if' condition"
        assert result.severity == "warning"

    def test_24_unless_clause_ignored(self):
        """Context has 'unless' clause, output says 'you can' without ack."""
        context = "You may proceed unless your account has outstanding violations."
        output = "You can proceed with the transaction immediately."
        result = check_c3_false_certainty(context, output)
        assert not result.passed, "C3 should catch confident claim ignoring 'unless'"

    def test_25_require_condition_ignored(self):
        """Context has 'require' conditions, output says 'go ahead and'."""
        context = "All withdrawals require two-factor authentication and manager approval."
        output = "Go ahead and submit your withdrawal request now."
        result = check_c3_false_certainty(context, output)
        assert not result.passed, "C3 should catch 'go ahead and' ignoring requirements"

    # --- SHOULD PASS (conditions acknowledged or no conditions) ---

    def test_26_conditions_acknowledged_with_however(self):
        """Context has conditions, output acknowledges with 'however'."""
        context = "If you are a premium member, you can access advanced features."
        output = (
            "You can access advanced features. However, please confirm your "
            "premium membership status first."
        )
        result = check_c3_false_certainty(context, output)
        assert result.passed, "C3 should pass when 'however' acknowledges conditions"

    def test_27_conditions_acknowledged_with_note_that(self):
        """Context has conditions, output acknowledges with 'note that'."""
        context = "Unless your subscription has expired, you can download updates."
        output = (
            "You are eligible for updates. Note that this depends on your "
            "subscription being active."
        )
        result = check_c3_false_certainty(context, output)
        assert result.passed, "C3 should pass when 'note that' acknowledges conditions"

    def test_28_no_conditional_markers_in_context(self):
        """Context has no conditional markers — confident output is fine."""
        context = "All members receive a welcome package upon registration."
        output = "You will receive a welcome package when you register."
        result = check_c3_false_certainty(context, output)
        assert result.passed, "C3 should pass when context has no conditions"

    def test_29_empty_output(self):
        """Empty output should pass."""
        context = "If you qualify, you can apply."
        output = ""
        result = check_c3_false_certainty(context, output)
        assert result.passed, "C3 should pass with empty output"

    # --- SHOULD FAIL BUT WON'T (xfail — heuristic limitations) ---

    @pytest.mark.xfail(
        reason=(
            "Known heuristic limitation: C3 only detects conditional markers "
            "'if/unless/except/however/but/require'. 'Results vary' and "
            "'outcomes depend on' express conditionality but do not match "
            "any of these markers as substrings."
        )
    )
    def test_30_results_vary_not_matched(self):
        """'results vary' implies conditionality but no conditional markers.

        Note: We avoid words containing 'if' as a substring (e.g.,
        'significantly') since the heuristic does substring matching.
        """
        context = "Results vary greatly depending on personal health status."
        output = "You will see improvement within two weeks."
        result = check_c3_false_certainty(context, output)
        assert not result.passed

    @pytest.mark.xfail(
        reason=(
            "Known heuristic limitation: C3 does not recognize 'experts disagree' "
            "as a conditional marker, so confident output passes."
        )
    )
    def test_31_experts_disagree_not_matched(self):
        """'Experts disagree' implies uncertainty but no conditional markers."""
        context = "Experts disagree on the long-term effects of this treatment."
        output = "You will experience full recovery within 6 months."
        result = check_c3_false_certainty(context, output)
        assert not result.passed


# =============================================================================
# C4: Preserve Tensions (Conflict Collapse)
# =============================================================================

class TestC4ConflictCollapse:
    """
    C4 heuristic patterns:
    - Permissive words in context: can, eligible, allowed, permitted
    - Restrictive words in context: non-refundable, cannot, not allowed,
      prohibited, require
    - Tension acknowledgment in output: however, but, although,
      on the other hand, exception, note that
    - Fails when: has_permissive AND has_restrictive AND NOT acknowledges_tension
    """

    # --- SHOULD FAIL (tension collapsed) ---

    def test_32_can_and_non_refundable_collapsed(self):
        """Context has 'can' (permissive) and 'non-refundable' (restrictive)."""
        context = (
            "Customers can return physical items within 30 days. "
            "Digital purchases are non-refundable."
        )
        output = "You can return your purchase at any time for a full refund."
        result = check_c4_conflict_collapse(context, output)
        assert not result.passed, "C4 should catch collapsed can/non-refundable tension"
        assert result.severity == "warning"

    def test_33_eligible_and_cannot_collapsed(self):
        """Context has 'eligible' and 'cannot', output only mentions eligibility."""
        context = (
            "Members are eligible for the loyalty discount. "
            "However, members cannot combine discounts with promotional offers."
        )
        output = "As a member, you are eligible for the loyalty discount on your order."
        result = check_c4_conflict_collapse(context, output)
        assert not result.passed, "C4 should catch collapsed eligible/cannot tension"

    def test_34_allowed_and_prohibited_collapsed(self):
        """Context has 'allowed' and 'prohibited', output ignores prohibition."""
        context = (
            "Pets are allowed in the lobby area. "
            "Pets are prohibited in the dining area and pool."
        )
        output = "Feel free to bring your pet. Pets are allowed on the premises."
        result = check_c4_conflict_collapse(context, output)
        assert not result.passed, "C4 should catch collapsed allowed/prohibited tension"

    # --- SHOULD PASS (tension preserved or no tension) ---

    def test_35_tension_preserved_with_however(self):
        """Context has tension, output uses 'however' to preserve it."""
        context = (
            "Employees can work remotely on Fridays. "
            "However, remote work is not allowed during project deadlines."
        )
        output = (
            "You can work remotely on Fridays. However, this is not permitted "
            "during active project deadlines."
        )
        result = check_c4_conflict_collapse(context, output)
        assert result.passed, "C4 should pass when 'however' preserves tension"

    def test_36_tension_preserved_with_although(self):
        """Context has tension, output uses 'although' to preserve it."""
        context = (
            "Students are eligible for the scholarship. "
            "Applicants cannot have a GPA below 3.0."
        )
        output = (
            "Although you are eligible for the scholarship, please note "
            "you must maintain a GPA of 3.0 or above."
        )
        result = check_c4_conflict_collapse(context, output)
        assert result.passed, "C4 should pass when 'although' preserves tension"

    def test_37_only_permissive_no_tension(self):
        """Context has only permissive language — no tension to collapse."""
        context = (
            "Members can access the gym at any time. "
            "Members are eligible for free towel service."
        )
        output = "You have full gym access and free towel service as a member."
        result = check_c4_conflict_collapse(context, output)
        assert result.passed, "C4 should pass when no tension exists (permissive only)"

    def test_38_only_restrictive_no_tension(self):
        """Context has only restrictive language — no tension to collapse.

        Note: We must avoid words that contain permissive substrings.
        "cannot" contains "can", "not allowed" contains "allowed".
        We use "prohibited" and "require" which are purely restrictive
        and don't contain any permissive substrings.
        """
        context = (
            "Smoking is prohibited on all floors. "
            "Visitors require a badge to enter."
        )
        output = "No smoking is permitted and all visitors need a badge."
        result = check_c4_conflict_collapse(context, output)
        assert result.passed, "C4 should pass when no tension exists (restrictive only)"

    # --- SHOULD FAIL BUT WON'T (xfail — heuristic limitations) ---

    @pytest.mark.xfail(
        reason=(
            "Known heuristic limitation: C4 only detects tension via specific "
            "permissive/restrictive keywords. 'Some say X, others say Y' expresses "
            "tension without using 'can/eligible/allowed' or "
            "'non-refundable/cannot/not allowed/prohibited/require'."
        )
    )
    def test_39_tension_without_keywords(self):
        """Tension expressed without permissive/restrictive keywords."""
        context = (
            "Some researchers argue the drug is safe for long-term use. "
            "Others argue it poses serious cardiovascular risks."
        )
        output = "The drug is safe for long-term use."
        result = check_c4_conflict_collapse(context, output)
        assert not result.passed


# =============================================================================
# C5: No Premature Compression
# =============================================================================

class TestC5PrematureCompression:
    """
    C5 heuristic patterns:
    - context_complexity = max(context_bullets, context_sentences)
      where bullets = count('-') + count('bullet') + count(newline)
      and sentences = count('.') + count('!')
    - output_sentences = count('.') + count('!') + count('?')
    - Fails when: context_complexity >= 3 AND output_sentences <= 1
    """

    # --- SHOULD FAIL (complex context compressed to tiny output) ---

    def test_40_five_sentences_zero_output_sentences(self):
        """Context with 5+ sentences, output with 0 sentences (no period/!/?)."""
        context = (
            "The policy covers medical expenses. It also covers dental. "
            "Vision is included in the premium tier. "
            "Prescriptions are covered at 80%. "
            "Mental health services require pre-authorization."
        )
        output = "Everything is covered"
        result = check_c5_premature_compression(context, output)
        assert not result.passed, "C5 should catch 5-sentence context compressed to 0-sentence output"
        assert result.severity == "warning"

    def test_41_five_bullet_points_one_word_output(self):
        """Context with 5+ bullet points, output is one word."""
        context = (
            "Features include:\n"
            "- Real-time analytics\n"
            "- Custom dashboards\n"
            "- API access\n"
            "- Role-based permissions\n"
            "- Automated reporting"
        )
        output = "Yes"
        result = check_c5_premature_compression(context, output)
        assert not result.passed, "C5 should catch bullet-list context compressed to one word"

    def test_42_many_periods_single_sentence_output(self):
        """Context with many periods (high complexity), single-sentence output."""
        context = (
            "Step 1: Open the settings menu. "
            "Step 2: Navigate to security. "
            "Step 3: Enable two-factor authentication. "
            "Step 4: Scan the QR code with your authenticator app. "
            "Step 5: Enter the verification code."
        )
        output = "Follow the steps."
        result = check_c5_premature_compression(context, output)
        assert not result.passed, "C5 should catch multi-step context compressed to one sentence"

    # --- SHOULD PASS (appropriate compression) ---

    def test_43_short_context_short_output(self):
        """Short context (1 sentence), short output (1 sentence)."""
        context = "Paris is the capital of France."
        output = "The capital of France is Paris."
        result = check_c5_premature_compression(context, output)
        assert result.passed, "C5 should pass for proportional short content"

    def test_44_multi_point_context_multi_sentence_output(self):
        """Context with 3+ points, output with 2+ sentences."""
        context = (
            "The plan includes: unlimited calls. "
            "It also includes 10GB data. "
            "International roaming is available. "
            "Customer support is 24/7."
        )
        output = (
            "Your plan includes unlimited calls and 10GB of data. "
            "International roaming and 24/7 support are also included."
        )
        result = check_c5_premature_compression(context, output)
        assert result.passed, "C5 should pass when output has adequate sentences"

    def test_45_empty_context_some_output(self):
        """Empty context should always pass (insufficient data)."""
        context = ""
        output = "Here is a detailed answer with multiple points."
        result = check_c5_premature_compression(context, output)
        assert result.passed, "C5 should pass with empty context"

    # --- SHOULD FAIL BUT WON'T (xfail — heuristic limitations) ---

    @pytest.mark.xfail(
        reason=(
            "Known heuristic limitation: C5 only checks output_sentences <= 1. "
            "A 500-word nuanced policy compressed to exactly 2 sentences passes "
            "the heuristic even though it is a massive compression."
        )
    )
    def test_46_massive_compression_two_sentences(self):
        """500-word policy compressed to 2 sentences passes the >1 check."""
        context = (
            "Section 1: Coverage details. This plan provides comprehensive medical coverage "
            "for individuals and families. It includes inpatient and outpatient services. "
            "Emergency room visits are covered at 90% after a $50 copay. "
            "Preventive care is covered at 100% with no copay. "
            "Specialist visits require a referral from your primary care physician. "
            "Mental health services are covered with a $30 copay per session. "
            "Prescription drugs are covered through a three-tier formulary system. "
            "Generic drugs have a $10 copay. Brand-name drugs have a $35 copay. "
            "Specialty drugs have a $75 copay. "
            "Section 2: Exclusions. Cosmetic procedures are not covered. "
            "Experimental treatments are not covered unless pre-approved. "
            "Out-of-network providers are covered at 60% of allowed amounts. "
            "Section 3: Limits. Annual out-of-pocket maximum is $6,000 for individuals. "
            "Family out-of-pocket maximum is $12,000. "
            "Lifetime maximum benefit is unlimited for essential health benefits. "
            "Section 4: Enrollment. Open enrollment runs from November 1 to December 15. "
            "Special enrollment periods apply for qualifying life events. "
            "Coverage begins on the first day of the month following enrollment."
        )
        output = "Your plan covers medical services. Contact us for details."
        result = check_c5_premature_compression(context, output)
        assert not result.passed


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Cross-cutting edge cases that exercise multiple checks or boundary conditions."""

    def test_47_empty_context_and_empty_output(self):
        """Empty context AND empty output should pass all checks."""
        context = ""
        output = ""
        assert check_c1_context_contradiction(context, output).passed
        assert check_c2_unmarked_inference(context, output).passed
        assert check_c3_false_certainty(context, output).passed
        assert check_c4_conflict_collapse(context, output).passed
        assert check_c5_premature_compression(context, output).passed

    def test_48_output_longer_than_context(self):
        """Output longer than context should pass all checks."""
        context = "Short context."
        output = (
            "This is a much longer output that provides extensive detail. "
            "It covers multiple aspects of the topic. "
            "It includes supporting information and examples. "
            "The response may be helpful for understanding the full picture."
        )
        assert check_c1_context_contradiction(context, output).passed
        assert check_c2_unmarked_inference(context, output).passed
        assert check_c3_false_certainty(context, output).passed
        assert check_c4_conflict_collapse(context, output).passed
        assert check_c5_premature_compression(context, output).passed

    def test_49_very_long_context_short_adequate_output(self):
        """Very long context (2000+ chars), short but multi-sentence output."""
        context = (
            "Policy document section A: " + "Details about coverage. " * 50
            + "Section B: " + "Additional terms apply. " * 50
        )
        assert len(context) > 2000
        # Output has 2 sentences — should pass C5's >1 sentence threshold
        output = (
            "The policy covers many services with specific terms. "
            "Please review the full document for complete details."
        )
        result = check_c5_premature_compression(context, output)
        assert result.passed, "C5 should pass when output has >1 sentence"

    def test_50_context_with_code_json_formatting(self):
        """Context with code/JSON formatting should not confuse heuristics."""
        context = (
            '{"policy": "non-refundable", "type": "digital", '
            '"status": "active", "version": 2.1}'
        )
        output = "The policy data shows the item is digital and active."
        # The context has "non-refundable" but output doesn't say "eligible" + "refund"
        result_c1 = check_c1_context_contradiction(context, output)
        assert result_c1.passed, "JSON context should not false-positive on C1"

        result_c2 = check_c2_unmarked_inference(context, output)
        assert result_c2.passed, "JSON context should not false-positive on C2"

    def test_51_i_dont_know_output(self):
        """'I don't know' output should pass all checks.

        Note: C4 requires context that does NOT have both permissive and
        restrictive terms, otherwise even "I don't know" fails because
        it doesn't acknowledge the tension. We use a context with only
        conditional markers (for C3 testing) but no permissive/restrictive
        pair that triggers C4.
        """
        context = (
            "Complex policy with many conditions. "
            "If you meet the criteria, benefits are available. "
            "Unless disqualified, the process moves forward."
        )
        output = "I don't know the answer to your question. Please consult the policy directly."
        assert check_c1_context_contradiction(context, output).passed
        assert check_c2_unmarked_inference(context, output).passed
        assert check_c3_false_certainty(context, output).passed
        assert check_c4_conflict_collapse(context, output).passed
        assert check_c5_premature_compression(context, output).passed

    def test_52_medical_disclaimer_language(self):
        """Medical disclaimer with hedge-like language should pass C2."""
        context = "Some patients experience side effects."
        output = (
            "This is not medical advice and should not replace consultation "
            "with a qualified healthcare provider. Outcomes may vary and you "
            "should discuss treatment options with your doctor."
        )
        result = check_c2_unmarked_inference(context, output)
        assert result.passed, "Medical disclaimer language should pass C2"

    def test_53_sarcasm_triggers_c2(self):
        """Sarcastic definitive language still triggers C2 (no hedge present)."""
        context = "The product has a 2-star average rating."
        output = "This is definitely the best product on the market."
        result = check_c2_unmarked_inference(context, output)
        assert not result.passed, "C2 should catch 'definitely' even in sarcastic context"

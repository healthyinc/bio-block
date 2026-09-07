"""Tests for the hypothesis tree decision engine."""

import numpy as np
import pandas as pd
import pytest

from app.models.demo_schemas import (
    ColumnType,
    CandidateHypothesis,
    NodeKind,
    TreeState,
    VariableRole,
)
from app.services.dataset_profiler import profile_dataset
from app.services.hypothesis_tree import (
    add_answer_node,
    create_initial_tree,
    fork_branch,
    get_available_starts,
    get_candidate_analyses,
    get_candidate_hypotheses,
    get_next_question,
    validate_branch,
)




@pytest.fixture
def clinical_df():
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "treatment": np.random.choice(["Drug", "Placebo"], n),
        "severity": np.random.choice(["Mild", "Moderate", "Severe"], n),
        "age": np.random.normal(55, 10, n).round(1),
        "blood_pressure_pre": np.random.normal(140, 15, n).round(1),
        "blood_pressure_post": np.random.normal(130, 15, n).round(1),
        "cholesterol": np.random.normal(200, 30, n).round(1),
        "sex": np.random.choice(["M", "F"], n),
    })


@pytest.fixture
def profile(clinical_df):
    return profile_dataset(clinical_df)


@pytest.fixture
def tree():
    return create_initial_tree()


def _answer(tree, profile, branch_id, answer_text, option_id=None, context=None):
    """Shorthand to add an answer to the tree."""
    parent_id = tree.branches[branch_id].node_ids[-1]
    q = get_next_question(profile, tree, branch_id)
    ctx = context or {}
    if q and q.category:
        ctx[q.category] = option_id or answer_text
    tree, node = add_answer_node(tree, branch_id, parent_id, answer_text, option_id, q, ctx)
    return tree




class TestQuestionGeneration:
    def test_first_question_is_start_mode(self, profile, tree):
        branch_id = tree.active_branch_id
        q = get_next_question(profile, tree, branch_id)
        assert q is not None
        assert q.category == "start_mode"
        assert len(q.options) == 4

    def test_goal_question_after_start(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        q = get_next_question(profile, tree, branch_id)
        assert q is not None
        assert q.category == "goal"

    def test_outcome_question_after_compare_goal(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        q = get_next_question(profile, tree, branch_id)
        assert q is not None
        assert "outcome" in q.category

    def test_independent_design_asks_for_group(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        q = get_next_question(profile, tree, branch_id)
        assert q is not None
        assert "group" in q.category

    def test_paired_design_asks_for_paired_column(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_paired", "design_paired",
                       {"design": "design_paired"})
        q = get_next_question(profile, tree, branch_id)
        assert q is not None
        assert "paired" in q.category

    def test_independent_never_asks_paired_questions(self, profile, tree):
        """Selecting independent design must never ask paired-only questions."""
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})

        # Walk remaining questions
        for _ in range(10):
            q = get_next_question(profile, tree, branch_id)
            if q is None:
                break
            assert q.category != "select_paired", "Independent design should never ask for paired column"
            # Auto-answer to progress
            if q.options:
                tree = _answer(tree, profile, branch_id, q.options[0].label,
                               q.options[0].id, {q.category: q.options[0].id})
            else:
                break




class TestForkBranch:
    def test_fork_creates_sibling(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})

        # Fork from the root
        fork_node = tree.root_node_id
        tree, new_branch_id = fork_branch(tree, fork_node)

        assert new_branch_id != branch_id
        assert new_branch_id in tree.branches
        assert tree.active_branch_id == new_branch_id

    def test_fork_preserves_original(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})

        original_nodes = list(tree.branches[branch_id].node_ids)
        fork_node = tree.root_node_id
        tree, new_branch_id = fork_branch(tree, fork_node)

        # Original branch still has its nodes
        assert tree.branches[branch_id].node_ids == original_nodes

    def test_fork_does_not_mutate_original_answers(self, profile, tree):
        """Changing answers in one branch must not affect another."""
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})

        original_node_count = len(tree.branches[branch_id].node_ids)

        # Fork
        fork_node = tree.root_node_id
        tree, new_branch_id = fork_branch(tree, fork_node)

        # Add answer on new branch
        tree = _answer(tree, profile, new_branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, new_branch_id, "goal_relationship", "goal_relationship",
                       {"goal": "goal_relationship"})

        # Original branch should NOT have new nodes
        assert len(tree.branches[branch_id].node_ids) == original_node_count




class TestHypothesisGeneration:
    def test_two_group_hypothesis_generated(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment",
                       {"select_group": "treatment", "group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_two", "groups_two",
                       {"group_count": "groups_two"})

        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        assert len(hyps) >= 1
        assert any("blood_pressure_pre" in h.statement for h in hyps)
        assert any("treatment" in h.statement for h in hyps)

    def test_hypotheses_have_null_and_alternative(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment",
                       {"select_group": "treatment", "group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_two", "groups_two",
                       {"group_count": "groups_two"})

        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        for h in hyps:
            assert h.null_hypothesis
            assert h.alternative_hypothesis




class TestAnalysisAlternatives:
    def _build_two_group_tree(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment",
                       {"select_group": "treatment", "group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_two", "groups_two",
                       {"group_count": "groups_two"})
        return tree

    def test_two_group_offers_parametric_and_nonparametric(self, profile, tree):
        tree = self._build_two_group_tree(profile, tree)
        branch_id = tree.active_branch_id
        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        assert len(hyps) >= 1

        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)
        test_names = [a.test_name for a in analyses]
        assert "independent_ttest" in test_names
        assert "mann_whitney_u" in test_names

    def test_analyses_include_tradeoffs(self, profile, tree):
        tree = self._build_two_group_tree(profile, tree)
        branch_id = tree.active_branch_id
        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)

        for a in analyses:
            assert len(a.tradeoffs) > 0

    def test_one_analysis_is_suggested(self, profile, tree):
        tree = self._build_two_group_tree(profile, tree)
        branch_id = tree.active_branch_id
        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)

        suggested = [a for a in analyses if a.is_suggested]
        assert len(suggested) >= 1

    def test_non_normal_suggests_non_parametric(self, profile, tree):
        # Set the normality hint to non-normal for the outcome column
        for col in profile.columns:
            if col.name == "blood_pressure_pre":
                col.normality_hint = "likely_non_normal"
        
        tree = self._build_two_group_tree(profile, tree)
        branch_id = tree.active_branch_id
        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)

        t_test = next(a for a in analyses if a.test_name == "independent_ttest")
        mw_u = next(a for a in analyses if a.test_name == "mann_whitney_u")

        assert mw_u.is_suggested is True
        assert t_test.is_suggested is False
        assert mw_u.suggestion_reason is not None
        assert "non-normally distributed" in mw_u.suggestion_reason

    def test_approx_normal_suggests_parametric_ttest(self, profile, tree):
        # Even with mild skewness (|skew| <= 1.0) and normality_hint="appears_normal", suggest parametric
        for col in profile.columns:
            if col.name == "blood_pressure_pre":
                col.normality_hint = "appears_normal"
                col.skewness = 0.6
                col.kurtosis = 2.2

        tree = self._build_two_group_tree(profile, tree)
        branch_id = tree.active_branch_id
        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)

        t_test = next(a for a in analyses if a.test_name == "independent_ttest")
        mw_u = next(a for a in analyses if a.test_name == "mann_whitney_u")

        assert t_test.is_suggested is True
        assert mw_u.is_suggested is False

    def test_multi_group_normal_suggests_anova(self, profile, tree):
        for col in profile.columns:
            if col.name == "blood_pressure_pre":
                col.normality_hint = "appears_normal"
                col.skewness = 0.1
                col.kurtosis = 0.0

        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal", {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare", {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment",
                       {"select_group": "treatment", "group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_multi", "groups_multi",
                       {"group_count": "groups_multi"})

        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)

        anova = next(a for a in analyses if a.test_name == "one_way_anova")
        kruskal = next(a for a in analyses if a.test_name == "kruskal_wallis")

        assert anova.is_suggested is True
        assert kruskal.is_suggested is False

    def test_multi_group_non_normal_suggests_kruskal(self, profile, tree):
        for col in profile.columns:
            if col.name == "blood_pressure_pre":
                col.normality_hint = "likely_non_normal"

        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal", {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare", {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre",
                       {"select_outcome": "blood_pressure_pre", "outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment",
                       {"select_group": "treatment", "group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_multi", "groups_multi",
                       {"group_count": "groups_multi"})

        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)

        anova = next(a for a in analyses if a.test_name == "one_way_anova")
        kruskal = next(a for a in analyses if a.test_name == "kruskal_wallis")

        assert anova.is_suggested is False
        assert kruskal.is_suggested is True

    def test_large_dataset_suggests_independent_ttest(self, tree):
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            "weight": np.concatenate([np.random.normal(180, 20, 5000), np.random.normal(140, 15, 5000)]),
            "sex": ["M"] * 5000 + ["F"] * 5000,
        })
        large_profile = profile_dataset(df)

        branch_id = tree.active_branch_id
        tree = _answer(tree, large_profile, branch_id, "start_goal", "start_goal", {"start_mode": "start_goal"})
        tree = _answer(tree, large_profile, branch_id, "goal_compare", "goal_compare", {"goal": "goal_compare"})
        tree = _answer(tree, large_profile, branch_id, "weight", "col_weight",
                       {"select_outcome": "weight", "outcome": "weight"})
        tree = _answer(tree, large_profile, branch_id, "design_independent", "design_independent",
                       {"design": "design_independent"})
        tree = _answer(tree, large_profile, branch_id, "sex", "col_sex",
                       {"select_group": "sex", "group": "sex"})
        tree = _answer(tree, large_profile, branch_id, "groups_two", "groups_two",
                       {"group_count": "groups_two"})

        hyps = get_candidate_hypotheses(large_profile, tree, branch_id)
        analyses = get_candidate_analyses(large_profile, hyps[0], tree, branch_id)

        t_test = next(a for a in analyses if a.test_name == "independent_ttest")
        mw_u = next(a for a in analyses if a.test_name == "mann_whitney_u")

        assert t_test.is_suggested is True
        assert mw_u.is_suggested is False
        assert "Standard parametric test" in t_test.suggestion_reason
        assert "Levene's test" in t_test.suggestion_reason




class TestBranchValidation:
    def test_valid_branch(self, profile, tree):
        branch_id = tree.active_branch_id
        validation = validate_branch(profile, tree, branch_id)
        assert validation.is_valid

    def test_missing_column_contradiction(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal",
                       {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare",
                       {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "nonexistent_column", None,
                       {"outcome": "nonexistent_column"})

        validation = validate_branch(profile, tree, branch_id)
        assert not validation.is_valid
        assert len(validation.contradictions) > 0




class TestFollowUpQuestions:
    def test_followup_covariate_asks_for_covariate_column(self, profile, tree):
        branch_id = tree.active_branch_id
        # Build 2-group tree
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal", {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare", {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre", {"outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent", {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment", {"group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_two", "groups_two", {"group_count": "groups_two"})
        
        # Add follow-up answer: followup_covariate
        tree = _answer(tree, profile, branch_id, "Add a covariate and re-analyze", "followup_covariate")
        
        q = get_next_question(profile, tree, branch_id)
        assert q is not None
        assert q.category == "select_covariate"
        assert "covariate" in q.prompt.lower()


    def test_followup_nonparametric_promotes_mann_whitney(self, profile, tree):
        branch_id = tree.active_branch_id
        tree = _answer(tree, profile, branch_id, "start_goal", "start_goal", {"start_mode": "start_goal"})
        tree = _answer(tree, profile, branch_id, "goal_compare", "goal_compare", {"goal": "goal_compare"})
        tree = _answer(tree, profile, branch_id, "blood_pressure_pre", "col_blood_pressure_pre", {"outcome": "blood_pressure_pre"})
        tree = _answer(tree, profile, branch_id, "design_independent", "design_independent", {"design": "design_independent"})
        tree = _answer(tree, profile, branch_id, "treatment", "col_treatment", {"group": "treatment"})
        tree = _answer(tree, profile, branch_id, "groups_two", "groups_two", {"group_count": "groups_two"})
        
        # Add follow-up answer: followup_nonparametric
        tree = _answer(tree, profile, branch_id, "Compare with a non-parametric alternative", "followup_nonparametric")
        
        hyps = get_candidate_hypotheses(profile, tree, branch_id)
        analyses = get_candidate_analyses(profile, hyps[0], tree, branch_id)
        
        mw_u = next(a for a in analyses if a.test_name == "mann_whitney_u")
        assert mw_u.is_suggested is True
        assert "non-parametric alternative requested" in mw_u.suggestion_reason




from __future__ import annotations

import hashlib
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.models.demo_schemas import (
    AnalysisTradeoff,
    BranchStatus,
    BranchValidation,
    CandidateAnalysis,
    CandidateHypothesis,
    ColumnType,
    DatasetProfile,
    ForkResponse,
    HypothesisOrigin,
    NodeKind,
    Question,
    QuestionOption,
    StartMode,
    TreeBranch,
    TreeNode,
    TreeState,
    VariableRole,
    VariableRoleAssignment,
)


def _uid() -> str:
    return uuid.uuid4().hex[:12]

def _stable_id(prefix: str, *args: Any) -> str:
    """Stable deterministic ID from args."""
    m = hashlib.md5()
    for arg in args:
        m.update(str(arg).encode('utf-8'))
    return f"{prefix}_{m.hexdigest()[:12]}"

def _now() -> str:
    return datetime.utcnow().isoformat()



def _cols_by_type(profile: DatasetProfile, dtype: ColumnType) -> List[str]:
    return [c.name for c in profile.columns if c.dtype == dtype]


def _cols_with_role_suggestion(profile: DatasetProfile, role: VariableRole) -> List[str]:
    return [c.name for c in profile.columns if role in c.suggested_roles]


def _branch_nodes(tree: TreeState, branch_id: str) -> List[TreeNode]:
    """Ordered nodes in a branch."""
    branch = tree.branches.get(branch_id)
    if not branch:
        return []
    return [tree.nodes[nid] for nid in branch.node_ids if nid in tree.nodes]


def _branch_answers(tree: TreeState, branch_id: str) -> Dict[str, str]:
    """Answer values keyed by category / option id."""
    answers: Dict[str, str] = {}
    for node in _branch_nodes(tree, branch_id):
        if node.kind == NodeKind.ANSWER and node.answer:
            key = node.answer_option_id or node.answer
            answers[key] = node.answer
        # Also track context data
        if node.context:
            answers.update({str(k): str(v) for k, v in node.context.items()})
    return answers


def get_resolved_answers(profile: DatasetProfile, tree: TreeState, branch_id: str) -> Dict[str, str]:
    """Answers with auto-mapping for focus_variable."""
    answers = _branch_answers(tree, branch_id)
    

    if "focus_variable" in answers:
        col = answers["focus_variable"]
        col_profile = next((c for c in profile.columns if c.name == col), None)
        if col_profile:
            if col_profile.dtype == ColumnType.NUMERIC:
                answers.setdefault("outcome", col)
            elif col_profile.dtype == ColumnType.CATEGORICAL:
                answers.setdefault("group", col)
                

    if "goal" in answers:
        answers["start_mode"] = "goal"
        
    return answers


def _has_results_in_branch(tree: TreeState, branch_id: str) -> bool:
    return any(
        n.kind == NodeKind.RESULT for n in _branch_nodes(tree, branch_id)
    )



def resolve_variable_roles(
    tree: TreeState, branch_id: str
) -> Dict[str, VariableRoleAssignment]:
    """Column→role assignments from branch history."""
    roles: Dict[str, VariableRoleAssignment] = {}
    for node in _branch_nodes(tree, branch_id):
        ctx = node.context
        if "role_assignments" in ctx:
            for col, role_str in ctx["role_assignments"].items():
                roles[col] = VariableRoleAssignment(
                    column=col,
                    role=VariableRole(role_str),
                    branch_id=branch_id,
                )
    return roles



def get_available_starts(profile: DatasetProfile) -> Question:
    """Opening question: how to begin."""
    options = [
        QuestionOption(
            id="start_goal",
            label="Start with a research goal",
            description="Compare groups, study a relationship, or investigate change over time.",
        ),
        QuestionOption(
            id="start_variable",
            label="Start with an interesting variable",
            description="Choose a column and explore what can be investigated with it.",
        ),
        QuestionOption(
            id="start_observation",
            label="Start with an observation",
            description='For example: "Group A appears higher than Group B."',
        ),
        QuestionOption(
            id="start_free_text",
            label="Write my own hypothesis",
            description="State a hypothesis and map it to your dataset columns.",
        ),
    ]
    return Question(
        id=f"q_{_uid()}",
        prompt="How would you like to begin exploring this dataset?",
        explanation=(
            "You can approach your data from different angles. "
            "There is no single correct starting point."
        ),
        options=options,
        allows_custom=False,
        category="start_mode",
    )



def _goal_question(profile: DatasetProfile) -> Question:
    options = [
        QuestionOption(
            id="goal_compare",
            label="Compare outcomes across groups",
            description="Are outcomes different between two or more groups?",
        ),
        QuestionOption(
            id="goal_relationship",
            label="Study a relationship between variables",
            description="Is there a correlation or predictive link?",
        ),
        QuestionOption(
            id="goal_change",
            label="Study change over time",
            description="Do measurements change across repeated observations?",
        ),
        QuestionOption(
            id="goal_reference",
            label="Compare to a known reference value",
            description="Is the sample mean different from a known standard?",
        ),
    ]
    return Question(
        id=f"q_{_uid()}",
        prompt="What is your research goal?",
        explanation="Select the broad type of question you want to answer.",
        options=options,
        allows_custom=True,
        category="goal",
    )


def _variable_select_question(
    profile: DatasetProfile, role: str, exclude: List[str] | None = None
) -> Question:
    """Column-selection question for a variable role."""
    exclude = exclude or []

    if role == "outcome":
        candidates = _cols_by_type(profile, ColumnType.NUMERIC)
        prompt = "Which numeric column is your outcome (dependent variable)?"
        explanation = "This is the variable you want to measure or compare."
    elif role == "group":
        candidates = [
            c.name for c in profile.columns
            if c.dtype == ColumnType.CATEGORICAL and 2 <= (c.cardinality or c.unique_count) <= 20
        ]
        prompt = "Which column defines your groups?"
        explanation = "This column splits your data into comparison groups."
    elif role == "predictor":
        candidates = _cols_by_type(profile, ColumnType.NUMERIC)
        prompt = "Which numeric column is your predictor (independent variable)?"
        explanation = "This is the variable you think might predict or explain the outcome."
    elif role == "paired":
        candidates = _cols_by_type(profile, ColumnType.NUMERIC)
        prompt = "Which column contains the second (paired) measurement?"
        explanation = "Select the matching measurement for a within-subjects comparison."
    elif role == "time":
        candidates = (
            _cols_with_role_suggestion(profile, VariableRole.TIME)
            or _cols_by_type(profile, ColumnType.DATETIME)
        )
        prompt = "Which column represents time or measurement order?"
        explanation = "This defines the sequence of observations."
    elif role == "subject":
        candidates = (
            _cols_with_role_suggestion(profile, VariableRole.SUBJECT)
            or _cols_by_type(profile, ColumnType.IDENTIFIER)
        )
        prompt = "Which column identifies individual subjects or units?"
        explanation = "This links repeated measurements to the same participant."
    elif role == "covariate":
        candidates = [c.name for c in profile.columns]
        prompt = "Which column would you like to add as a covariate?"
        explanation = "This variable will be controlled for during analysis."
    elif role == "subgroup":
        candidates = [
            c.name for c in profile.columns
            if c.dtype == ColumnType.CATEGORICAL and 2 <= (c.cardinality or c.unique_count) <= 20
        ]
        prompt = "Which column defines the subgroup you want to analyze?"
        explanation = "The analysis will filter or stratify data based on this column."
    else:
        candidates = [c.name for c in profile.columns]
        prompt = f"Select a column for the '{role}' role."
        explanation = None

    candidates = [c for c in candidates if c not in exclude]

    options = [
        QuestionOption(id=f"col_{c}", label=c)
        for c in candidates
    ]

    if not options:
        options = [
            QuestionOption(
                id="no_columns",
                label="No suitable columns found",
                disabled=True,
                disabled_reason=f"The dataset has no columns suitable for the {role} role.",
            )
        ]

    return Question(
        id=f"q_{_uid()}",
        prompt=prompt,
        explanation=explanation,
        options=options,
        allows_custom=False,
        category=f"select_{role}",
    )


def _group_count_question(profile: DatasetProfile, group_col: str) -> Question:
    """Group count question (auto-detected from data)."""
    col_profile = next((c for c in profile.columns if c.name == group_col), None)
    n_groups = col_profile.unique_count if col_profile else 0

    options = []
    if n_groups == 2:
        options.append(QuestionOption(
            id="groups_two",
            label=f"Two groups (detected: {n_groups} levels)",
            description="Use a two-sample test.",
        ))
    elif n_groups >= 3:
        options.append(QuestionOption(
            id="groups_multi",
            label=f"Three or more groups (detected: {n_groups} levels)",
            description="Use an ANOVA-family test.",
        ))
    else:
        options.append(QuestionOption(
            id="groups_two",
            label="Two groups",
        ))
        options.append(QuestionOption(
            id="groups_multi",
            label="Three or more groups",
        ))

    return Question(
        id=f"q_{_uid()}",
        prompt=f"How many groups does '{group_col}' define?",
        explanation=f"Column '{group_col}' has {n_groups} unique value(s).",
        options=options,
        allows_custom=False,
        category="group_count",
    )


def _design_question() -> Question:
    return Question(
        id=f"q_{_uid()}",
        prompt="Are the measurements independent or paired/repeated?",
        explanation=(
            "Independent: different participants in each group. "
            "Paired: same participants measured twice. "
            "Repeated: same participants measured 3+ times."
        ),
        options=[
            QuestionOption(
                id="design_independent",
                label="Independent groups (different participants)",
            ),
            QuestionOption(
                id="design_paired",
                label="Paired measurements (same participants, two time points)",
            ),
            QuestionOption(
                id="design_repeated",
                label="Repeated measures (same participants, 3+ time points)",
            ),
        ],
        allows_custom=False,
        category="design",
    )


def _relationship_type_question() -> Question:
    return Question(
        id=f"q_{_uid()}",
        prompt="What type of relationship are you investigating?",
        options=[
            QuestionOption(
                id="rel_numeric_numeric",
                label="Relationship between two numeric variables",
                description="Correlation or regression.",
            ),
            QuestionOption(
                id="rel_categorical_categorical",
                label="Association between two categorical variables",
                description="Chi-square or Fisher's exact test.",
            ),
            QuestionOption(
                id="rel_with_covariates",
                label="Relationship while controlling for other variables",
                description="Regression with covariates.",
            ),
        ],
        allows_custom=True,
        category="relationship_type",
    )


def _not_sure_explanation(category: str) -> str:
    explanations = {
        "goal": (
            "• 'Compare outcomes' is for testing whether groups differ.\n"
            "• 'Study a relationship' is for correlations and predictions.\n"
            "• 'Change over time' is for longitudinal/repeated data.\n"
            "• 'Reference value' is for comparing your sample to a known number."
        ),
        "design": (
            "• 'Independent' means each participant appears in only one group.\n"
            "• 'Paired' means each participant is measured twice (e.g., before/after).\n"
            "• 'Repeated' means 3 or more measurements per participant."
        ),
    }
    return explanations.get(category, "Review the options and their descriptions.")



def _get_latest_followup(tree: TreeState, branch_id: str) -> Optional[str]:
    """Most recent follow-up action on the branch (after last RESULT node)."""
    nodes = _branch_nodes(tree, branch_id)


    last_result_idx = -1
    for i, n in enumerate(nodes):
        if n.kind == NodeKind.RESULT:
            last_result_idx = i


    search_nodes = nodes[last_result_idx + 1:] if last_result_idx >= 0 else nodes
    for n in reversed(search_nodes):
        if n.kind == NodeKind.ANSWER:
            opt = n.answer_option_id or n.answer or ""
            opt_lower = opt.lower()

            if opt.startswith("followup_"):
                return opt  # e.g. "followup_covariate"
            if "covariate" in opt_lower and "col_" not in opt:
                return "followup_covariate"
            if "nonparametric" in opt_lower or "non-parametric" in opt_lower:
                return "followup_nonparametric"
            if "subgroup" in opt_lower:
                return "followup_subgroup"
            if "related hypothesis" in opt_lower:
                return "followup_new_hypothesis"
            if "effect size" in opt_lower:
                return "followup_effect_size"
            if "export" in opt_lower:
                return "followup_export"
            return None  # non-followup answer — stop scanning
    return None



def _done_or_followup(profile: DatasetProfile, tree: TreeState, branch_id: str) -> Optional[Question]:
    """Return follow-up question if results exist, else None."""
    if _has_results_in_branch(tree, branch_id):
        return get_follow_up_question(profile, tree, branch_id)
    return None


def get_next_question(
    profile: DatasetProfile,
    tree: TreeState,
    branch_id: str,
) -> Optional[Question]:
    """Next unresolved question for the branch."""
    answers = get_resolved_answers(profile, tree, branch_id)


    latest_followup = _get_latest_followup(tree, branch_id)
    if latest_followup == "followup_covariate" and "select_covariate" not in answers:
        exclude = [c for c in [answers.get("outcome"), answers.get("group"), answers.get("predictor")] if c]
        return _variable_select_question(profile, "covariate", exclude=exclude)

    if latest_followup == "followup_subgroup" and "select_subgroup" not in answers:
        exclude = [c for c in [answers.get("outcome"), answers.get("group")] if c]
        return _variable_select_question(profile, "subgroup", exclude=exclude)

    if latest_followup == "followup_new_hypothesis" and "new_outcome" not in answers:
        exclude = [c for c in [answers.get("outcome")] if c]
        q = _variable_select_question(profile, "outcome", exclude=exclude)
        q.prompt = "Select a new outcome variable for the related hypothesis."
        q.category = "new_outcome"
        return q

    # start mode
    if "start_mode" not in answers:
        return get_available_starts(profile)

    start = answers.get("start_mode", "")

    # -- goal path --
    if start in ("start_goal", "goal"):
        if "goal" not in answers:
            return _goal_question(profile)

        goal = answers.get("goal", "")

        if goal in ("goal_compare", "compare"):

            if "outcome" not in answers:
                return _variable_select_question(profile, "outcome")

            outcome = answers.get("outcome", "")


            if "design" not in answers:
                return _design_question()

            design = answers.get("design", "")

            if design in ("design_independent", "independent"):
                if "group" not in answers:
                    return _variable_select_question(
                        profile, "group", exclude=[outcome]
                    )
                group = answers.get("group", "")
                if "group_count" not in answers:
                    return _group_count_question(profile, group)


                return _done_or_followup(profile, tree, branch_id)

            elif design in ("design_paired", "paired"):
                if "paired" not in answers:
                    return _variable_select_question(
                        profile, "paired", exclude=[outcome]
                    )
                return _done_or_followup(profile, tree, branch_id)

            elif design in ("design_repeated", "repeated"):
                if "subject" not in answers:
                    return _variable_select_question(profile, "subject")
                if "time" not in answers:
                    return _variable_select_question(profile, "time")
                return _done_or_followup(profile, tree, branch_id)

        elif goal in ("goal_relationship", "relationship"):
            if "relationship_type" not in answers:
                return _relationship_type_question()

            rel_type = answers.get("relationship_type", "")

            if rel_type in ("rel_numeric_numeric", "numeric_numeric"):
                if "outcome" not in answers:
                    return _variable_select_question(profile, "outcome")
                outcome = answers.get("outcome", "")
                if "predictor" not in answers:
                    return _variable_select_question(
                        profile, "predictor", exclude=[outcome]
                    )
                return _done_or_followup(profile, tree, branch_id)

            elif rel_type in ("rel_categorical_categorical", "categorical_categorical"):
                if "outcome" not in answers:
                    return _variable_select_question(profile, "group")
                if "predictor" not in answers:
                    cat_cols = [
                        c.name for c in profile.columns
                        if c.dtype == ColumnType.CATEGORICAL
                    ]
                    q = _variable_select_question(profile, "group")
                    q.prompt = "Select the second categorical variable."
                    q.category = "predictor"
                    return q
                return _done_or_followup(profile, tree, branch_id)

            elif rel_type in ("rel_with_covariates", "with_covariates"):
                if "outcome" not in answers:
                    return _variable_select_question(profile, "outcome")
                if "predictor" not in answers:
                    outcome = answers.get("outcome", "")
                    return _variable_select_question(
                        profile, "predictor", exclude=[outcome]
                    )
                return _done_or_followup(profile, tree, branch_id)

        elif goal in ("goal_change", "change"):
            if "outcome" not in answers:
                return _variable_select_question(profile, "outcome")
            if "subject" not in answers:
                return _variable_select_question(profile, "subject")
            if "time" not in answers:
                return _variable_select_question(profile, "time")
            return _done_or_followup(profile, tree, branch_id)

        elif goal in ("goal_reference", "reference"):
            if "outcome" not in answers:
                return _variable_select_question(profile, "outcome")
            # population mean as custom answer
            if "population_mean" not in answers:
                return Question(
                    id=f"q_{_uid()}",
                    prompt="What is the known reference (population) mean?",
                    explanation="Enter the value you want to compare your sample against.",
                    options=[],
                    allows_custom=True,
                    allows_skip=False,
                    category="population_mean",
                )
            return _done_or_followup(profile, tree, branch_id)

    # -- variable path --
    elif start in ("start_variable", "variable"):
        if "focus_variable" not in answers:
            all_cols = [c.name for c in profile.columns]
            return Question(
                id=f"q_{_uid()}",
                prompt="Which variable interests you?",
                explanation="Select a column and we'll explore what analyses are possible.",
                options=[QuestionOption(id=f"col_{c}", label=c) for c in all_cols],
                allows_custom=False,
                category="focus_variable",
            )

        if "goal" not in answers:
            return _goal_question(profile)
        # then follow goal path
        return get_next_question(profile, tree, branch_id)

    # -- observation path --
    elif start in ("start_observation", "observation"):
        if "observation_text" not in answers:
            return Question(
                id=f"q_{_uid()}",
                prompt="Describe what you observed in the data.",
                explanation='For example: "Treated patients seem to have higher scores."',
                options=[],
                allows_custom=True,
                category="observation_text",
            )
        if "goal" not in answers:
            return _goal_question(profile)
        return get_next_question(profile, tree, branch_id)

    # -- free text path --
    elif start in ("start_free_text", "free_text"):
        if "hypothesis_text" not in answers:
            return Question(
                id=f"q_{_uid()}",
                prompt="Write your hypothesis in plain language.",
                explanation="We'll help map it to your dataset columns.",
                options=[],
                allows_custom=True,
                category="hypothesis_text",
            )
        if "outcome" not in answers:
            return _variable_select_question(profile, "outcome")
        if "goal" not in answers:
            return _goal_question(profile)
        return get_next_question(profile, tree, branch_id)

    return _done_or_followup(profile, tree, branch_id)




def get_candidate_hypotheses(
    profile: DatasetProfile,
    tree: TreeState,
    branch_id: str,
) -> List[CandidateHypothesis]:
    """Generate hypotheses from branch state."""
    answers = get_resolved_answers(profile, tree, branch_id)
    hypotheses: List[CandidateHypothesis] = []
    is_exploratory = _has_results_in_branch(tree, branch_id)

    goal = answers.get("goal", "")
    outcome = answers.get("outcome", "")
    group = answers.get("group", "")
    design = answers.get("design", "")

    if goal in ("goal_compare", "compare") and outcome:
        if design in ("design_independent", "independent") and group:
            group_count = answers.get("group_count", "")
            if group_count in ("groups_two", "two"):
                hypotheses.append(CandidateHypothesis(
                    id=_stable_id("hyp", branch_id, outcome, group, "compare_two"),
                    branch_id=branch_id,
                    statement=f"The mean '{outcome}' differs between the two groups defined by '{group}'.",
                    null_hypothesis=f"There is no difference in '{outcome}' between groups of '{group}'.",
                    alternative_hypothesis=f"There is a significant difference in '{outcome}' between groups of '{group}'.",
                    variables={
                        outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
                        group: VariableRoleAssignment(column=group, role=VariableRole.GROUP, branch_id=branch_id),
                    },
                    origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
                ))
            elif group_count in ("groups_multi", "multi"):
                hypotheses.append(CandidateHypothesis(
                    id=_stable_id("hyp", branch_id, outcome, group, "compare_multi"),
                    branch_id=branch_id,
                    statement=f"The mean '{outcome}' differs across the groups defined by '{group}'.",
                    null_hypothesis=f"All group means of '{outcome}' are equal across '{group}' levels.",
                    alternative_hypothesis=f"At least one group mean of '{outcome}' differs.",
                    variables={
                        outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
                        group: VariableRoleAssignment(column=group, role=VariableRole.GROUP, branch_id=branch_id),
                    },
                    origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
                ))

        elif design in ("design_paired", "paired"):
            paired_col = answers.get("paired", "")
            if paired_col:
                hypotheses.append(CandidateHypothesis(
                    id=_stable_id("hyp", branch_id, outcome, paired_col, "compare_paired"),
                    branch_id=branch_id,
                    statement=f"There is a difference between '{outcome}' and '{paired_col}' (paired measurements).",
                    null_hypothesis=f"The mean difference between '{outcome}' and '{paired_col}' is zero.",
                    alternative_hypothesis=f"The mean difference between '{outcome}' and '{paired_col}' is not zero.",
                    variables={
                        outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
                        paired_col: VariableRoleAssignment(column=paired_col, role=VariableRole.PAIRED, branch_id=branch_id),
                    },
                    origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
                ))

        elif design in ("design_repeated", "repeated"):
            hypotheses.append(CandidateHypothesis(
                id=_stable_id("hyp", branch_id, outcome, "compare_repeated"),
                branch_id=branch_id,
                statement=f"'{outcome}' changes significantly across repeated measurements.",
                null_hypothesis=f"There is no change in '{outcome}' across time points.",
                alternative_hypothesis=f"'{outcome}' differs across at least two time points.",
                variables={
                    outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
                },
                origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
            ))

    elif goal in ("goal_relationship", "relationship") and outcome:
        rel_type = answers.get("relationship_type", "")
        predictor = answers.get("predictor", "")

        if rel_type in ("rel_numeric_numeric", "numeric_numeric") and predictor:
            hypotheses.append(CandidateHypothesis(
                id=_stable_id("hyp", branch_id, outcome, predictor, "rel_numeric"),
                branch_id=branch_id,
                statement=f"There is a relationship between '{predictor}' and '{outcome}'.",
                null_hypothesis=f"There is no correlation between '{predictor}' and '{outcome}'.",
                alternative_hypothesis=f"There is a significant correlation between '{predictor}' and '{outcome}'.",
                variables={
                    outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
                    predictor: VariableRoleAssignment(column=predictor, role=VariableRole.PREDICTOR, branch_id=branch_id),
                },
                origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
            ))

        elif rel_type in ("rel_categorical_categorical", "categorical_categorical") and predictor:
            hypotheses.append(CandidateHypothesis(
                id=_stable_id("hyp", branch_id, outcome, predictor, "rel_categorical"),
                branch_id=branch_id,
                statement=f"There is an association between '{outcome}' and '{predictor}'.",
                null_hypothesis=f"'{outcome}' and '{predictor}' are independent.",
                alternative_hypothesis=f"'{outcome}' and '{predictor}' are associated.",
                variables={
                    outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
                    predictor: VariableRoleAssignment(column=predictor, role=VariableRole.PREDICTOR, branch_id=branch_id),
                },
                origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
            ))

    elif goal in ("goal_reference", "reference") and outcome:
        pop_mean = answers.get("population_mean", "?")
        hypotheses.append(CandidateHypothesis(
            id=_stable_id("hyp", branch_id, outcome, pop_mean, "reference"),
            branch_id=branch_id,
            statement=f"The mean of '{outcome}' differs from the reference value {pop_mean}.",
            null_hypothesis=f"The population mean of '{outcome}' equals {pop_mean}.",
            alternative_hypothesis=f"The population mean of '{outcome}' does not equal {pop_mean}.",
            variables={
                outcome: VariableRoleAssignment(column=outcome, role=VariableRole.OUTCOME, branch_id=branch_id),
            },
            origin=HypothesisOrigin.EXPLORATORY if is_exploratory else HypothesisOrigin.CONFIRMATORY,
        ))

    return hypotheses




def _is_column_non_normal(profile: DatasetProfile, col_name: str) -> bool:
    """Check if column looks non-normal using evidence-based heuristics.

    Parametric tests (t-tests, ANOVA) are robust to moderate departures from normality,
    especially in moderate to large samples by the Central Limit Theorem (CLT).
    Therefore, parametric tests (which auto-handle heteroscedasticity via Welch's correction)
    are recommended by default unless:
      - The sample size is small (< 500) and formal tests/heuristics show non-normality.
      - The sample is substantially skewed (|skewness| > 1.5).
      - The user explicitly requested a non-parametric alternative.
    """
    if not col_name:
        return False
    
    n_rows = profile.row_count

    for col in profile.columns:
        if col.name == col_name:
            # Large samples (n >= 500): CLT guarantees asymptotic normality of sample means.
            # Only extreme skewness (|skew| > 1.5) warrants a non-parametric default suggestion.
            if n_rows >= 500:
                if col.skewness is not None and abs(col.skewness) > 1.5:
                    return True
                return False

            # Moderate / small samples (n < 500)
            if col.normality_hint == "likely_non_normal":
                return True
            if col.normality_hint == "appears_normal":
                return False

            # Substantial skewness threshold (|skewness| > 1.0)
            # Bulmer (1979): |skew| < 0.5 is approx symmetric, 0.5-1.0 is moderate, > 1.0 is highly skewed.
            if col.skewness is not None and abs(col.skewness) > 1.0:
                return True

            # Substantial kurtosis threshold (|excess kurtosis| > 3.0)
            if col.kurtosis is not None and abs(col.kurtosis) > 3.0:
                return True

            return False
    return False


def _normality_confidence(profile: DatasetProfile, col_name: str) -> str:
    """Return normality confidence: 'normal', 'borderline', or 'non_normal'."""
    if not col_name:
        return "normal"
    n_rows = profile.row_count
    for col in profile.columns:
        if col.name == col_name:
            if n_rows >= 500:
                if col.skewness is not None and abs(col.skewness) > 1.5:
                    return "non_normal"
                if col.skewness is not None and abs(col.skewness) > 0.8:
                    return "borderline"
                return "normal"
            if col.normality_hint == "likely_non_normal":
                return "non_normal"
            if col.normality_hint == "appears_normal":
                return "normal"
            if col.skewness is not None and abs(col.skewness) > 1.0:
                return "non_normal"
            if col.kurtosis is not None and abs(col.kurtosis) > 3.0:
                return "non_normal"
            if (col.skewness is not None and abs(col.skewness) > 0.5) or (col.kurtosis is not None and abs(col.kurtosis) > 2.0):
                return "borderline"
            return "normal"
    return "normal"



def get_candidate_analyses(
    profile: DatasetProfile,
    hypothesis: CandidateHypothesis,
    tree: TreeState,
    branch_id: str,
) -> List[CandidateAnalysis]:
    """Competing analysis approaches for a hypothesis."""
    answers = get_resolved_answers(profile, tree, branch_id)
    analyses: List[CandidateAnalysis] = []

    latest_followup = _get_latest_followup(tree, branch_id)
    prefer_nonparametric = (
        latest_followup == "followup_nonparametric"
        or "followup_nonparametric" in answers
    )

    goal = answers.get("goal", "")
    design = answers.get("design", "")
    group_count = answers.get("group_count", "")
    rel_type = answers.get("relationship_type", "")

    # two independent groups
    if goal in ("goal_compare", "compare") and design in ("design_independent", "independent"):
        if group_count in ("groups_two", "two"):
            outcome_col = answers.get("outcome", "")
            non_normal = _is_column_non_normal(profile, outcome_col)
            use_nonparametric = non_normal or prefer_nonparametric

            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "independent_ttest"),
                hypothesis_id=hypothesis.id,
                test_name="independent_ttest",
                display_name="Independent samples t-test",
                description="Compares means of two groups. Automatically runs Levene's test to apply Student's t-test (equal variances) or Welch's t-test (unequal variances).",
                is_suggested=not use_nonparametric,
                suggestion_reason=None if use_nonparametric else "Standard parametric test for two-group comparisons. Evaluates equality of variances via Levene's test to select Student's or Welch's t-test.",
                tradeoffs=[
                    AnalysisTradeoff(label="Assumes normality & checks variance", description="Robust for moderate/large samples under CLT; auto-corrects for unequal variances via Welch's t-test."),
                    AnalysisTradeoff(label="More statistical power", description="Detects smaller differences than non-parametric alternatives."),
                ],
            ))
            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "mann_whitney_u"),
                hypothesis_id=hypothesis.id,
                test_name="mann_whitney_u",
                display_name="Mann-Whitney U test",
                description="Non-parametric alternative that compares distributions without assuming normality.",
                is_suggested=use_nonparametric,
                suggestion_reason=(
                    "Suggested non-parametric alternative requested by researcher."
                    if prefer_nonparametric
                    else ("Suggested fallback because the outcome variable appears to be non-normally distributed." if non_normal else None)
                ),
                tradeoffs=[
                    AnalysisTradeoff(label="No normality assumption", description="Works with skewed or ordinal data."),
                    AnalysisTradeoff(label="Slightly less power", description="May miss small differences that a t-test would detect."),
                ],
            ))

        elif group_count in ("groups_multi", "multi"):
            outcome_col = answers.get("outcome", "")
            non_normal = _is_column_non_normal(profile, outcome_col)
            use_nonparametric = non_normal or prefer_nonparametric

            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "one_way_anova"),
                hypothesis_id=hypothesis.id,
                test_name="one_way_anova",
                display_name="One-way ANOVA",
                description="Compares means across three or more groups. Automatically applies standard Fisher's ANOVA or Welch's ANOVA based on Levene's test.",
                is_suggested=not use_nonparametric,
                suggestion_reason=None if use_nonparametric else "Standard parametric test for multi-group comparisons with post-hoc testing (auto-handles unequal variances via Welch's ANOVA).",
                tradeoffs=[
                    AnalysisTradeoff(label="Assumes normality & checks variance", description="Auto-falls back to Welch's ANOVA if variances are unequal across groups."),
                    AnalysisTradeoff(label="Post-hoc tests available", description="Can identify which specific groups differ via Tukey HSD or Games-Howell."),
                ],
            ))
            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "kruskal_wallis"),
                hypothesis_id=hypothesis.id,
                test_name="kruskal_wallis",
                display_name="Kruskal-Wallis H test",
                description="Non-parametric alternative for comparing distributions across 3+ groups.",
                is_suggested=use_nonparametric,
                suggestion_reason=(
                    "Suggested non-parametric alternative requested by researcher."
                    if prefer_nonparametric
                    else ("Suggested fallback because the outcome variable appears to be non-normally distributed." if non_normal else None)
                ),
                tradeoffs=[
                    AnalysisTradeoff(label="No normality assumption", description="Works with skewed or ordinal data."),
                    AnalysisTradeoff(label="Less specific", description="Tests whether distributions differ, not specifically means."),
                ],
            ))

    # paired
    elif goal in ("goal_compare", "compare") and design in ("design_paired", "paired"):
        outcome_col = answers.get("outcome", "")
        paired_col = answers.get("paired", "")
        non_normal = _is_column_non_normal(profile, outcome_col) or _is_column_non_normal(profile, paired_col)
        use_nonparametric = non_normal or prefer_nonparametric

        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "paired_ttest"),
            hypothesis_id=hypothesis.id,
            test_name="paired_ttest",
            display_name="Paired t-test",
            description="Compares two related measurements assuming normal distribution of differences.",
            is_suggested=not use_nonparametric,
            suggestion_reason=None if use_nonparametric else "Standard test for before/after or matched-pairs designs.",
            tradeoffs=[
                AnalysisTradeoff(label="Assumes normal differences", description="The differences (not raw scores) must be approximately normal."),
            ],
        ))
        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "wilcoxon_signed_rank"),
            hypothesis_id=hypothesis.id,
            test_name="wilcoxon_signed_rank",
            display_name="Wilcoxon signed-rank test",
            description="Non-parametric paired comparison.",
            is_suggested=use_nonparametric,
            suggestion_reason=(
                "Suggested non-parametric alternative requested by researcher."
                if prefer_nonparametric
                else ("Suggested fallback because one or both paired variables appear to be non-normally distributed." if non_normal else None)
            ),
            tradeoffs=[
                AnalysisTradeoff(label="No normality assumption", description="Uses ranks rather than raw values."),
                AnalysisTradeoff(label="Slightly less power", description="May miss small effects."),
            ],
        ))

    # repeated measures
    elif goal in ("goal_compare", "compare") and design in ("design_repeated", "repeated"):
        outcome_col = answers.get("outcome", "")
        non_normal = _is_column_non_normal(profile, outcome_col)
        use_nonparametric = non_normal or prefer_nonparametric

        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "repeated_measures_anova"),
            hypothesis_id=hypothesis.id,
            test_name="repeated_measures_anova",
            display_name="Repeated-measures ANOVA",
            description="Tests for differences across 3+ time points with the same subjects.",
            is_suggested=not use_nonparametric,
            suggestion_reason=None if use_nonparametric else "Standard approach for within-subjects longitudinal designs.",
            tradeoffs=[
                AnalysisTradeoff(label="Assumes sphericity", description="Requires similar variances of differences between conditions."),
            ],
        ))
        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "friedman"),
            hypothesis_id=hypothesis.id,
            test_name="friedman",
            display_name="Friedman test",
            description="Non-parametric alternative for repeated measures.",
            is_suggested=use_nonparametric,
            suggestion_reason=(
                "Suggested non-parametric alternative requested by researcher."
                if prefer_nonparametric
                else ("Suggested fallback because the outcome variable appears to be non-normally distributed." if non_normal else None)
            ),
            tradeoffs=[
                AnalysisTradeoff(label="Rank-based", description="No distributional assumptions."),
                AnalysisTradeoff(label="Less power", description="May miss subtle changes."),
            ],
        ))

    # reference value
    elif goal in ("goal_reference", "reference"):
        outcome_col = answers.get("outcome", "")
        non_normal = _is_column_non_normal(profile, outcome_col)
        use_nonparametric = non_normal or prefer_nonparametric

        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "one_sample_ttest"),
            hypothesis_id=hypothesis.id,
            test_name="one_sample_ttest",
            display_name="One-sample t-test",
            description="Tests whether the sample mean differs from a known value.",
            is_suggested=not use_nonparametric,
            suggestion_reason=None if use_nonparametric else "Standard test for comparing a sample to a reference.",
            tradeoffs=[
                AnalysisTradeoff(label="Assumes normality", description="Best with roughly symmetric data."),
            ],
        ))
        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "one_sample_wilcoxon"),
            hypothesis_id=hypothesis.id,
            test_name="one_sample_wilcoxon",
            display_name="One-sample Wilcoxon signed-rank test",
            description="Non-parametric alternative using signed ranks.",
            is_suggested=use_nonparametric,
            suggestion_reason=(
                "Suggested non-parametric alternative requested by researcher."
                if prefer_nonparametric
                else ("Suggested fallback because the outcome variable appears to be non-normally distributed." if non_normal else None)
            ),
            tradeoffs=[
                AnalysisTradeoff(label="No normality assumption", description="Robust to skewed data."),
            ],
        ))

    # numeric-numeric relationship
    elif goal in ("goal_relationship", "relationship"):
        if rel_type in ("rel_numeric_numeric", "numeric_numeric"):
            outcome_col = answers.get("outcome", "")
            predictor_col = answers.get("predictor", "")
            non_normal = _is_column_non_normal(profile, outcome_col) or _is_column_non_normal(profile, predictor_col)
            use_nonparametric = non_normal or prefer_nonparametric

            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "pearson_correlation"),
                hypothesis_id=hypothesis.id,
                test_name="pearson_correlation",
                display_name="Pearson correlation",
                description="Measures linear relationship between two numeric variables.",
                is_suggested=not use_nonparametric,
                suggestion_reason=None if use_nonparametric else "Most common for bivariate linear associations.",
                tradeoffs=[
                    AnalysisTradeoff(label="Assumes linearity", description="Only detects linear relationships."),
                    AnalysisTradeoff(label="Sensitive to outliers", description="Extreme values can distort the coefficient."),
                ],
            ))
            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "spearman_correlation"),
                hypothesis_id=hypothesis.id,
                test_name="spearman_correlation",
                display_name="Spearman rank correlation",
                description="Non-parametric monotonic relationship measure.",
                is_suggested=use_nonparametric,
                suggestion_reason=(
                    "Suggested non-parametric alternative requested by researcher."
                    if prefer_nonparametric
                    else ("Suggested fallback because one or both variables appear to be non-normally distributed." if non_normal else None)
                ),
                tradeoffs=[
                    AnalysisTradeoff(label="Detects monotonic relationships", description="Not limited to linear patterns."),
                    AnalysisTradeoff(label="Robust to outliers", description="Uses ranks instead of raw values."),
                ],
            ))

        elif rel_type in ("rel_categorical_categorical", "categorical_categorical"):
            analyses.append(CandidateAnalysis(
                id=_stable_id("ana", hypothesis.id, "chi_square"),
                hypothesis_id=hypothesis.id,
                test_name="chi_square",
                display_name="Chi-square test of independence",
                description="Tests whether two categorical variables are independent.",
                is_suggested=True,
                suggestion_reason="Standard test for categorical associations.",
                tradeoffs=[
                    AnalysisTradeoff(label="Requires adequate cell counts", description="Expected counts should be ≥5 in most cells."),
                ],
            ))

    # change over time
    elif goal in ("goal_change", "change"):
        outcome_col = answers.get("outcome", "")
        non_normal = _is_column_non_normal(profile, outcome_col)
        use_nonparametric = non_normal or prefer_nonparametric

        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "repeated_measures_anova"),
            hypothesis_id=hypothesis.id,
            test_name="repeated_measures_anova",
            display_name="Repeated-measures ANOVA",
            description="Tests for change across time points within subjects.",
            is_suggested=not use_nonparametric,
            suggestion_reason=None if use_nonparametric else "Standard for longitudinal within-subjects designs.",
            tradeoffs=[
                AnalysisTradeoff(label="Assumes sphericity", description="Requires similar variances of differences."),
            ],
        ))
        analyses.append(CandidateAnalysis(
            id=_stable_id("ana", hypothesis.id, "friedman"),
            hypothesis_id=hypothesis.id,
            test_name="friedman",
            display_name="Friedman test",
            description="Non-parametric repeated-measures test.",
            is_suggested=use_nonparametric,
            suggestion_reason=(
                "Suggested non-parametric alternative requested by researcher."
                if prefer_nonparametric
                else ("Suggested fallback because the outcome variable appears to be non-normally distributed." if non_normal else None)
            ),
            tradeoffs=[
                AnalysisTradeoff(label="Rank-based", description="No distributional assumptions."),
            ],
        ))

    return analyses



def validate_branch(
    profile: DatasetProfile,
    tree: TreeState,
    branch_id: str,
) -> BranchValidation:
    """Check branch for contradictions and data issues."""
    answers = get_resolved_answers(profile, tree, branch_id)
    contradictions: List[str] = []
    warnings: List[str] = []
    missing_roles: List[VariableRole] = []
    dq_issues: List[str] = []

    goal = answers.get("goal", "")
    design = answers.get("design", "")
    outcome = answers.get("outcome", "")
    group = answers.get("group", "")

    # Check outcome column exists and is numeric
    if outcome:
        col = next((c for c in profile.columns if c.name == outcome), None)
        if col is None:
            contradictions.append(f"Outcome column '{outcome}' not found in dataset.")
        elif col.dtype != ColumnType.NUMERIC and goal not in ("goal_relationship",):
            warnings.append(
                f"Outcome '{outcome}' is {col.dtype.value}, not numeric. "
                "Some tests may not apply."
            )
        if col and col.missing_pct > 20:
            dq_issues.append(
                f"Outcome '{outcome}' has {col.missing_pct}% missing values."
            )

    # Check group column
    if group:
        col = next((c for c in profile.columns if c.name == group), None)
        if col is None:
            contradictions.append(f"Group column '{group}' not found in dataset.")
        elif col.dtype == ColumnType.NUMERIC and col.unique_count > 20:
            warnings.append(
                f"Group column '{group}' has {col.unique_count} unique numeric values. "
                "Consider using a categorical variable."
            )

    # Check design consistency
    if design in ("design_paired", "paired") and not answers.get("paired"):
        missing_roles.append(VariableRole.PAIRED)

    if design in ("design_repeated", "repeated"):
        if not answers.get("subject"):
            missing_roles.append(VariableRole.SUBJECT)
        if not answers.get("time"):
            missing_roles.append(VariableRole.TIME)

    return BranchValidation(
        is_valid=len(contradictions) == 0,
        contradictions=contradictions,
        warnings=warnings,
        missing_roles=missing_roles,
        data_quality_issues=dq_issues,
    )



def create_initial_tree() -> TreeState:
    """Create a fresh tree with a root node and one branch."""
    root_id = f"node_{_uid()}"
    branch_id = f"branch_{_uid()}"

    root = TreeNode(
        id=root_id,
        branch_id=branch_id,
        kind=NodeKind.ROOT,
        prompt="Dataset uploaded. Ready to explore.",
    )
    branch = TreeBranch(
        id=branch_id,
        name="Main exploration",
        node_ids=[root_id],
        is_primary=True,
    )

    return TreeState(
        nodes={root_id: root},
        branches={branch_id: branch},
        active_branch_id=branch_id,
        root_node_id=root_id,
    )


def add_answer_node(
    tree: TreeState,
    branch_id: str,
    parent_node_id: str,
    answer: str,
    option_id: Optional[str] = None,
    question: Optional[Question] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[TreeState, TreeNode]:
    """Add question+answer node pair. Returns (tree, answer_node)."""
    tree = deepcopy(tree)
    branch = tree.branches[branch_id]

    # Create question node
    q_id = f"node_{_uid()}"
    q_node = TreeNode(
        id=q_id,
        parent_id=parent_node_id,
        branch_id=branch_id,
        kind=NodeKind.QUESTION,
        prompt=question.prompt if question else "Question",
        options=question.options if question else [],
        context=context or {},
    )
    tree.nodes[q_id] = q_node
    branch.node_ids.append(q_id)

    # Create answer node
    a_id = f"node_{_uid()}"
    a_node = TreeNode(
        id=a_id,
        parent_id=q_id,
        branch_id=branch_id,
        kind=NodeKind.ANSWER,
        answer=answer,
        answer_option_id=option_id,
        context=context or {},
        is_exploratory=_has_results_in_branch(tree, branch_id),
    )
    tree.nodes[a_id] = a_node
    branch.node_ids.append(a_id)

    return tree, a_node


def fork_branch(
    tree: TreeState,
    source_node_id: str,
    new_answer: Optional[str] = None,
) -> Tuple[TreeState, str]:
    """Fork from a prior node, creating a sibling branch.

    Returns the updated tree and the new branch ID.
    """
    tree = deepcopy(tree)
    source_node = tree.nodes.get(source_node_id)
    if not source_node:
        raise ValueError(f"Node '{source_node_id}' not found.")

    old_branch_id = source_node.branch_id
    old_branch = tree.branches[old_branch_id]

    # Find the index of the source node in the old branch
    try:
        idx = old_branch.node_ids.index(source_node_id)
    except ValueError:
        idx = 0

    # New branch shares ancestry up to (and including) source_node
    shared_ids = old_branch.node_ids[: idx + 1]

    new_branch_id = f"branch_{_uid()}"
    new_branch = TreeBranch(
        id=new_branch_id,
        name=f"Fork from '{old_branch.name}'",
        node_ids=list(shared_ids),
    )
    tree.branches[new_branch_id] = new_branch
    tree.active_branch_id = new_branch_id

    return tree, new_branch_id


def add_result_node(
    tree: TreeState,
    branch_id: str,
    parent_node_id: str,
    result_data: Dict[str, Any],
) -> Tuple[TreeState, TreeNode]:
    """Append a result node."""
    tree = deepcopy(tree)
    branch = tree.branches[branch_id]

    r_id = f"node_{_uid()}"
    r_node = TreeNode(
        id=r_id,
        parent_id=parent_node_id,
        branch_id=branch_id,
        kind=NodeKind.RESULT,
        prompt="Analysis complete",
        context=result_data,
    )
    tree.nodes[r_id] = r_node
    branch.node_ids.append(r_id)

    return tree, r_node


def get_follow_up_question(
    profile: DatasetProfile,
    tree: TreeState,
    branch_id: str,
) -> Optional[Question]:
    """Post-result follow-up options."""
    latest_followup = _get_latest_followup(tree, branch_id)

    prompt = "What would you like to do next?"
    explanation = "Pick a follow-up path."

    if latest_followup == "followup_effect_size":
        # Extract active analysis result from tree if available
        res_node = None
        for n in reversed(_branch_nodes(tree, branch_id)):
            if n.kind == NodeKind.RESULT and n.context:
                res_node = n.context
                break

        if res_node:
            es = res_node.get("effect_size", {})
            r = res_node.get("result", {})
            ci = es.get("ci_95") or r.get("ci_95_difference") or r.get("ci_95")
            metric = es.get("metric") or "effect size"
            val = es.get("value")
            mag = es.get("magnitude") or ""

            ci_str = f"95% CI [{', '.join(str(x) for x in ci)}]" if ci else ""
            val_str = f"{metric}: {val}" if val is not None else ""
            mag_str = f" ({mag} magnitude)" if mag else ""

            prompt = f"📊 Effect Size & Confidence Interval: {val_str}{mag_str}"
            if ci_str:
                explanation = f"Detailed Breakdown — {ci_str}. Evaluates effect magnitude beyond p-value."
            elif val_str:
                explanation = f"Detailed Breakdown — {val_str}{mag_str} calculated for this active test."
            else:
                explanation = "Effect sizes show how big the effect actually is, independent of sample size."
        else:
            prompt = "📊 Effect Size & Confidence Interval Breakdown"
            explanation = "Effect sizes show how big the effect actually is, independent of sample size."

    elif latest_followup == "followup_nonparametric":
        prompt = "🔄 Non-parametric Alternative Selected"
        explanation = "Updated recommendations to rank-based tests (Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Spearman). Pick one below."

    elif latest_followup == "followup_export":
        prompt = "📄 Branch Rationale & Audit Trail Exported"
        explanation = "Decision trail, stats, effect sizes, and provenance for this branch have been logged."

    options = [
        QuestionOption(
            id="followup_effect_size",
            label="Inspect effect size and confidence interval",
            description="Check Cohen's d / η² / r with 95% CIs.",
        ),
        QuestionOption(
            id="followup_covariate",
            label="Add a covariate and re-analyze",
            description="Control for a confounding variable via ANCOVA / regression.",
        ),
        QuestionOption(
            id="followup_nonparametric",
            label="Compare with a non-parametric alternative",
            description="Try rank-based tests when normality assumptions are violated.",
        ),
        QuestionOption(
            id="followup_subgroup",
            label="Test a subgroup or interaction",
            description="Stratify by a categorical variable to check for interaction effects.",
        ),
        QuestionOption(
            id="followup_new_hypothesis",
            label="Create a related hypothesis with a different outcome",
            description="Same groups, different outcome variable.",
        ),
        QuestionOption(
            id="followup_export",
            label="Stop and export this branch's rationale",
            description="Export full decision trail, stats, and provenance.",
        ),
    ]

    return Question(
        id=f"q_{_uid()}",
        prompt=prompt,
        explanation=explanation,
        options=options,
        allows_custom=True,
        category="follow_up",
    )


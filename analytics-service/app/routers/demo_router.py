"""Hypothesis Lab demo router. No auth required."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.demo_schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisRunResponse,
    AnswerRequest,
    AnswerResponse,
    ForkRequest,
    ForkResponse,
    HypothesisAction,
    HypothesisRequest,
    SessionResponse,
    CandidateHypothesis,
    HypothesisOrigin,
    VariableRoleAssignment,
    VariableRole,
    NodeKind,
)
from app.services.dataset_profiler import profile_dataset
from app.services.demo_sessions import session_store
from app.services.hypothesis_tree import (
    add_answer_node,
    add_result_node,
    create_initial_tree,
    fork_branch,
    get_available_starts,
    get_candidate_analyses,
    get_candidate_hypotheses,
    get_follow_up_question,
    get_next_question,
    get_resolved_answers,
    _branch_nodes,
    validate_branch,
)

from app.utils.csv_parser import parse_csv


from app.services.inferential import (
    run_two_group_test,
    run_paired_test,
    run_one_sample_test,
    run_multi_group_test,
    _TEST_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo/hypothesis-tree", tags=["Demo – Hypothesis Lab"])



def _get_session(session_id: str):
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found or expired.")
    return session


def _get_active_analysis_result(session, branch_id: str) -> Optional[AnalysisResult]:
    """Reconstruct AnalysisResult from branch RESULT node."""
    if not branch_id:
        return None
    branch = session.tree.branches.get(branch_id)
    if not branch or not branch.node_ids:
        return None

    # Find the last RESULT node and collect answer nodes after it
    nodes = [session.tree.nodes[nid] for nid in branch.node_ids if nid in session.tree.nodes]
    last_res_idx = -1
    for i, n in enumerate(nodes):
        if n.kind == NodeKind.RESULT:
            last_res_idx = i

    if last_res_idx < 0:
        return None

    res_node = nodes[last_res_idx]
    result_data = res_node.context
    test_code = result_data.get("test_used", "")
    test_used = _TEST_DISPLAY_NAMES.get(test_code, test_code) or "Statistical Analysis"

    # Collect follow-up actions selected after the last RESULT node
    applied_followups = []
    for n in nodes[last_res_idx + 1:]:
        if n.kind == NodeKind.ANSWER:
            opt_id = n.answer_option_id or ""
            text = n.answer or ""
            ctx = n.context or {}

            if opt_id == "followup_effect_size" or "effect size" in text.lower():
                applied_followups.append("Inspected effect size & 95% confidence interval")
            elif opt_id == "followup_nonparametric" or "non-parametric" in text.lower():
                applied_followups.append("Prioritized non-parametric alternative methods")
            elif opt_id == "followup_export" or "export" in text.lower():
                applied_followups.append("Exported branch exploration rationale")
            elif "select_covariate" in ctx:
                applied_followups.append(f"Added covariate: {ctx['select_covariate']}")
            elif "select_subgroup" in ctx:
                applied_followups.append(f"Tested subgroup: {ctx['select_subgroup']}")
            elif "new_outcome" in ctx:
                applied_followups.append(f"Explored related outcome: {ctx['new_outcome']}")
            elif opt_id.startswith("followup_"):
                applied_followups.append(opt_id.replace("followup_", "").replace("_", " ").title())
            elif text and not text.startswith("col_"):
                applied_followups.append(text)

    return AnalysisResult(
        analysis_id=res_node.id,
        hypothesis_id=result_data.get("hypothesis_id"),
        test_used=test_used,
        result=result_data.get("result", {}),
        effect_size=result_data.get("effect_size", {}),
        assumptions=result_data.get("assumptions", {}),
        interpretation=result_data.get("interpretation", ""),
        warnings=result_data.get("warnings", []),
        follow_up_options=[
            "Inspect effect size",
            "Add a covariate",
            "Compare with non-parametric alternative",
            "Test a subgroup",
            "Create a related hypothesis",
        ],
        applied_followups=applied_followups,
    )



def _build_session_response(session) -> SessionResponse:
    """Build session response with question, hypotheses, etc."""
    branch_id = session.tree.active_branch_id
    current_q = None
    hypotheses = []
    analyses = []
    validation = None

    if branch_id:
        current_q = get_next_question(session.profile, session.tree, branch_id)
        hypotheses = get_candidate_hypotheses(session.profile, session.tree, branch_id)
        validation = validate_branch(session.profile, session.tree, branch_id)

        # Also include saved hypotheses
        for hyp in session.hypotheses.values():
            if hyp.branch_id == branch_id and hyp not in hypotheses:
                hypotheses.append(hyp)

        # Generate analyses for each hypothesis
        for hyp in hypotheses:
            hyp_analyses = get_candidate_analyses(
                session.profile, hyp, session.tree, branch_id
            )
            analyses.extend(hyp_analyses)

    analysis_result = _get_active_analysis_result(session, branch_id)

    return SessionResponse(
        session_id=session.session_id,
        profile=session.profile,
        tree=session.tree,
        current_question=current_q,
        candidate_hypotheses=hypotheses,
        candidate_analyses=analyses,
        validation=validation,
        analysis_result=analysis_result,
    )


# ── endpoints ─────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionResponse)
async def create_session(file: UploadFile = File(...)):
    """Upload CSV, profile it, create session."""
    contents = await file.read()
    df = parse_csv(contents)
    profile = profile_dataset(df)
    tree = create_initial_tree()

    session_id = session_store.create(df=df, profile=profile, tree=tree)
    session = session_store.get(session_id)

    return _build_session_response(session)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, active_branch_id: Optional[str] = None):
    """Get current session state."""
    session = _get_session(session_id)
    if active_branch_id:
        if active_branch_id in session.tree.branches:
            session.tree.active_branch_id = active_branch_id
        else:
            raise HTTPException(400, f"Branch '{active_branch_id}' not found in tree.")
    return _build_session_response(session)


@router.post("/sessions/{session_id}/answers", response_model=AnswerResponse)
async def submit_answer(session_id: str, body: AnswerRequest):
    """Submit answer, advance tree."""
    session = _get_session(session_id)
    branch_id = session.tree.active_branch_id

    if not branch_id:
        raise HTTPException(400, "No active branch.")

    parent_node = session.tree.nodes.get(body.parent_node_id)
    if parent_node is None:
        raise HTTPException(404, f"Node '{body.parent_node_id}' not found.")

    # Determine the answer text and option ID
    answer_text = body.custom_answer or body.option_id or ""
    option_id = body.option_id

    # Get the current question for context
    current_q = get_next_question(session.profile, session.tree, branch_id)

    # Build context from the answer
    context = {}
    if current_q:
        category = current_q.category
        if category:
            context[category] = answer_text

    # Handle column selections that set variable roles
    if body.selected_columns:
        context["role_assignments"] = body.selected_columns

    # If user selected a column (option starts with "col_"), extract the column name
    if option_id and option_id.startswith("col_"):
        col_name = option_id[4:]  # strip "col_" prefix
        answer_text = col_name
        if current_q and current_q.category:
            role = current_q.category.replace("select_", "")
            context[role] = col_name
            context.setdefault("role_assignments", {})[col_name] = role

    # If the category is a known context key, store the answer under it
    if current_q and current_q.category:
        context[current_q.category] = answer_text

    # For start_mode, store as "start_mode"
    if current_q and current_q.category == "start_mode":
        context["start_mode"] = option_id or answer_text

    tree, answer_node = add_answer_node(
        tree=session.tree,
        branch_id=branch_id,
        parent_node_id=body.parent_node_id,
        answer=answer_text,
        option_id=option_id,
        question=current_q,
        context=context,
    )
    session.tree = tree
    session_store.update_tree(session_id, tree)

    # Automatically execute re-analysis if a follow-up parameter was submitted
    answers = get_resolved_answers(session.profile, session.tree, branch_id)
    outcome_col = answers.get("outcome") or answers.get("select_outcome")
    group_col = answers.get("group") or answers.get("select_group")

    # Recover outcome/group from previous RESULT node if missing
    for n in reversed(_branch_nodes(session.tree, branch_id)):
        if n.kind == NodeKind.RESULT and n.context:
            res_ctx = n.context
            if not outcome_col and res_ctx.get("outcome_col"):
                outcome_col = res_ctx["outcome_col"]
            if not group_col and res_ctx.get("group_col"):
                group_col = res_ctx["group_col"]

    new_result_data = None

    if "select_covariate" in context and outcome_col and group_col:
        covariate_col = context["select_covariate"]
        try:
            from app.services.inferential import run_ancova
            new_result_data = run_ancova(session.df, outcome_col, group_col, covariate_col)
            new_result_data["outcome_col"] = outcome_col
            new_result_data["group_col"] = group_col
            new_result_data["hypothesis_id"] = f"hyp_{outcome_col}_{group_col}"
        except Exception as err:
            logger.warning(f"Covariate re-analysis failed: {err}")

    elif "select_subgroup" in context and outcome_col and group_col:
        subgroup_col = context["select_subgroup"]
        try:
            from app.services.inferential import run_subgroup_analysis
            new_result_data = run_subgroup_analysis(session.df, outcome_col, group_col, subgroup_col)
            new_result_data["outcome_col"] = outcome_col
            new_result_data["group_col"] = group_col
            new_result_data["hypothesis_id"] = f"hyp_{outcome_col}_{group_col}"
        except Exception as err:
            logger.warning(f"Subgroup re-analysis failed: {err}")

    elif "new_outcome" in context and group_col:
        new_outcome_col = context["new_outcome"]
        try:
            new_result_data = run_two_group_test(session.df, new_outcome_col, group_col)
            new_result_data["outcome_col"] = new_outcome_col
            new_result_data["group_col"] = group_col
            new_result_data["hypothesis_id"] = f"hyp_{new_outcome_col}_{group_col}"
        except Exception as err:
            logger.warning(f"New outcome re-analysis failed: {err}")

    elif option_id == "followup_nonparametric" and outcome_col and group_col:
        try:
            new_result_data = run_two_group_test(session.df, outcome_col, group_col, force_test="mann_whitney_u")
            new_result_data["outcome_col"] = outcome_col
            new_result_data["group_col"] = group_col
            new_result_data["hypothesis_id"] = f"hyp_{outcome_col}_{group_col}"
        except Exception as err:
            logger.warning(f"Non-parametric re-analysis failed: {err}")

    if new_result_data:
        session.tree, r_node = add_result_node(
            tree=session.tree,
            branch_id=branch_id,
            parent_node_id=answer_node.id,
            result_data=new_result_data,
        )
        session_store.update_tree(session_id, session.tree)

    # Build response
    next_q = get_next_question(session.profile, session.tree, branch_id)
    hypotheses = get_candidate_hypotheses(session.profile, session.tree, branch_id)
    validation = validate_branch(session.profile, session.tree, branch_id)

    analyses = []
    for hyp in hypotheses:
        hyp_analyses = get_candidate_analyses(
            session.profile, hyp, session.tree, branch_id
        )
        analyses.extend(hyp_analyses)

    analysis_result = _get_active_analysis_result(session, branch_id)

    return AnswerResponse(
        tree=session.tree,
        current_question=next_q,
        candidate_hypotheses=hypotheses,
        candidate_analyses=analyses,
        validation=validation,
        analysis_result=analysis_result,
    )



@router.post("/sessions/{session_id}/forks", response_model=ForkResponse)
async def fork_from_node(session_id: str, body: ForkRequest):
    """Fork from any prior node."""
    session = _get_session(session_id)

    try:
        tree, new_branch_id = fork_branch(
            tree=session.tree,
            source_node_id=body.node_id,
            new_answer=body.new_answer,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    session.tree = tree
    session_store.update_tree(session_id, tree)

    next_q = get_next_question(session.profile, tree, new_branch_id)

    return ForkResponse(
        new_branch_id=new_branch_id,
        tree=tree,
        current_question=next_q,
    )


@router.post("/sessions/{session_id}/hypotheses", response_model=SessionResponse)
async def manage_hypothesis(session_id: str, body: HypothesisRequest):
    """Manage hypothesis: save, rename, annotate, duplicate, archive."""
    session = _get_session(session_id)
    branch_id = body.branch_id or session.tree.active_branch_id

    if body.action == HypothesisAction.SAVE:
        if not body.statement:
            raise HTTPException(400, "statement is required for save.")
        from app.services.hypothesis_tree import _uid
        hyp_id = f"hyp_{_uid()}"
        hyp = CandidateHypothesis(
            id=hyp_id,
            branch_id=branch_id or "",
            statement=body.statement,
            annotation=body.annotation,
        )
        session.hypotheses[hyp_id] = hyp

    elif body.action == HypothesisAction.RENAME:
        if not body.hypothesis_id or not body.statement:
            raise HTTPException(400, "hypothesis_id and statement are required.")
        hyp = session.hypotheses.get(body.hypothesis_id)
        if not hyp:
            raise HTTPException(404, "Hypothesis not found.")
        hyp.statement = body.statement

    elif body.action == HypothesisAction.ANNOTATE:
        if not body.hypothesis_id:
            raise HTTPException(400, "hypothesis_id is required.")
        hyp = session.hypotheses.get(body.hypothesis_id)
        if not hyp:
            raise HTTPException(404, "Hypothesis not found.")
        hyp.annotation = body.annotation

    elif body.action == HypothesisAction.DUPLICATE:
        if not body.hypothesis_id:
            raise HTTPException(400, "hypothesis_id is required.")
        original = session.hypotheses.get(body.hypothesis_id)
        if not original:
            raise HTTPException(404, "Hypothesis not found.")
        from app.services.hypothesis_tree import _uid
        dup_id = f"hyp_{_uid()}"
        dup = original.model_copy(update={"id": dup_id, "is_primary": False})
        session.hypotheses[dup_id] = dup

    elif body.action == HypothesisAction.SET_PRIMARY:
        if not body.hypothesis_id:
            raise HTTPException(400, "hypothesis_id is required.")
        for hyp in session.hypotheses.values():
            hyp.is_primary = False
        hyp = session.hypotheses.get(body.hypothesis_id)
        if hyp:
            hyp.is_primary = True

    elif body.action == HypothesisAction.ARCHIVE:
        if not body.hypothesis_id:
            raise HTTPException(400, "hypothesis_id is required.")
        hyp = session.hypotheses.get(body.hypothesis_id)
        if hyp:
            hyp.origin = HypothesisOrigin.EXPLORATORY

    elif body.action == HypothesisAction.DELETE:
        if not body.hypothesis_id:
            raise HTTPException(400, "hypothesis_id is required.")
        session.hypotheses.pop(body.hypothesis_id, None)

    return _build_session_response(session)


@router.post("/sessions/{session_id}/analyses", response_model=AnalysisRunResponse)
async def run_analysis(session_id: str, body: AnalysisRequest):
    """Run selected analysis."""
    session = _get_session(session_id)
    branch_id = session.tree.active_branch_id

    if not branch_id:
        raise HTTPException(400, "No active branch.")

    # Find the hypothesis
    hyp = session.hypotheses.get(body.hypothesis_id)
    if not hyp:
        # Check generated hypotheses
        generated = get_candidate_hypotheses(session.profile, session.tree, branch_id)
        hyp = next((h for h in generated if h.id == body.hypothesis_id), None)
    if not hyp:
        raise HTTPException(404, "Hypothesis not found.")

    # Find the analysis
    analyses = get_candidate_analyses(
        session.profile, hyp, session.tree, branch_id
    )
    analysis = next((a for a in analyses if a.id == body.analysis_id), None)
    if not analysis:
        raise HTTPException(404, "Analysis not found.")

    # Extract variable info from hypothesis
    df = session.df
    outcome_col = None
    group_col = None
    paired_col = None
    predictor_col = None

    for var_name, role_assign in hyp.variables.items():
        if role_assign.role == VariableRole.OUTCOME:
            outcome_col = role_assign.column
        elif role_assign.role == VariableRole.GROUP:
            group_col = role_assign.column
        elif role_assign.role == VariableRole.PAIRED:
            paired_col = role_assign.column
        elif role_assign.role == VariableRole.PREDICTOR:
            predictor_col = role_assign.column

    # Run the appropriate test
    try:
        result_data = _execute_analysis(
            df=df,
            test_name=analysis.test_name,
            outcome_col=outcome_col,
            group_col=group_col,
            paired_col=paired_col,
            predictor_col=predictor_col,
            answers=None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Analysis execution failed")
        raise HTTPException(500, f"Analysis failed: {exc}")

    # Add result node to tree
    result_data["hypothesis_id"] = body.hypothesis_id
    result_data["outcome_col"] = outcome_col
    result_data["group_col"] = group_col
    last_node_id = session.tree.branches[branch_id].node_ids[-1]
    tree, result_node = add_result_node(
        tree=session.tree,
        branch_id=branch_id,
        parent_node_id=last_node_id,
        result_data=result_data,
    )

    session.tree = tree
    session_store.update_tree(session_id, tree)

    # Build result
    test_code = result_data.get("test_used", analysis.test_name)
    test_used = _TEST_DISPLAY_NAMES.get(test_code, test_code)
    analysis_result = AnalysisResult(
        analysis_id=body.analysis_id,
        hypothesis_id=body.hypothesis_id,
        test_used=test_used,
        result=result_data.get("result", {}),
        effect_size=result_data.get("effect_size", {}),
        assumptions=result_data.get("assumptions", {}),
        interpretation=result_data.get("interpretation", ""),
        warnings=result_data.get("warnings", []),
        follow_up_options=[
            "Inspect effect size",
            "Add a covariate",
            "Compare with non-parametric alternative",
            "Test a subgroup",
            "Create a related hypothesis",
        ],
    )

    follow_up = get_follow_up_question(session.profile, tree, branch_id)

    return AnalysisRunResponse(
        result=analysis_result,
        tree=tree,
        follow_up_question=follow_up,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session."""
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {"detail": "Session deleted."}


# ── analysis execution bridge ─────────────────────────────────────────────

def _execute_analysis(
    df,
    test_name: str,
    outcome_col: Optional[str],
    group_col: Optional[str],
    paired_col: Optional[str],
    predictor_col: Optional[str],
    answers: Optional[dict],
) -> dict:
    """Bridge between hypothesis-tree IDs and inferential functions."""

    if test_name in ("independent_ttest", "mann_whitney_u"):
        if not outcome_col or not group_col:
            raise ValueError("outcome and group columns are required.")
        return run_two_group_test(df, outcome_col, group_col, force_test=test_name)


    elif test_name in ("one_way_anova", "kruskal_wallis"):
        if not outcome_col or not group_col:
            raise ValueError("outcome and group columns are required.")
        return run_multi_group_test(df, outcome_col, group_col)

    elif test_name in ("paired_ttest", "wilcoxon_signed_rank"):
        if not outcome_col or not paired_col:
            raise ValueError("Two paired columns are required.")
        return run_paired_test(df, outcome_col, paired_col)

    elif test_name in ("one_sample_ttest", "one_sample_wilcoxon"):
        if not outcome_col:
            raise ValueError("outcome column is required.")
        # Extract population mean from context
        pop_mean = 0.0  # Default
        if answers and "population_mean" in answers:
            try:
                pop_mean = float(answers["population_mean"])
            except (ValueError, TypeError):
                pass
        return run_one_sample_test(df, outcome_col, pop_mean)

    elif test_name in ("pearson_correlation", "spearman_correlation"):
        if not outcome_col or not predictor_col:
            raise ValueError("Two numeric columns are required.")
        # Use scipy directly for correlation
        import numpy as np
        from scipy import stats as sp_stats

        x = df[predictor_col].dropna().astype(float)
        y = df[outcome_col].dropna().astype(float)
        # Align on shared indices
        shared = x.index.intersection(y.index)
        x, y = x.loc[shared].values, y.loc[shared].values

        if test_name == "pearson_correlation":
            stat, p = sp_stats.pearsonr(x, y)
            method = "Pearson correlation"
        else:
            stat, p = sp_stats.spearmanr(x, y)
            method = "Spearman rank correlation"

        return {
            "test_used": method,
            "result": {
                "correlation": round(float(stat), 6),
                "p_value": round(float(p), 6),
                "n": len(x),
                "significant": bool(p < 0.05),
            },
            "effect_size": {
                "metric": "r",
                "value": round(float(stat), 4),
                "magnitude": (
                    "large" if abs(stat) >= 0.5
                    else "medium" if abs(stat) >= 0.3
                    else "small" if abs(stat) >= 0.1
                    else "negligible"
                ),
            },
            "assumptions": {},
            "interpretation": (
                f"The {method.lower()} between '{predictor_col}' and '{outcome_col}' "
                f"is r = {stat:.4f} (p = {p:.4f}). "
                f"{'This is statistically significant at α = 0.05.' if p < 0.05 else 'This is not statistically significant at α = 0.05.'}"
            ),
            "warnings": [],
            "reason": f"{method} selected.",
            "group_stats": {},
        }

    elif test_name == "chi_square":
        if not outcome_col or not predictor_col:
            raise ValueError("Two categorical columns are required.")
        import pandas as pd_local
        contingency = pd_local.crosstab(df[outcome_col], df[predictor_col])
        from scipy import stats as sp_stats
        chi2, p, dof, expected = sp_stats.chi2_contingency(contingency)

        n = contingency.sum().sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))) if n > 0 else 0

        return {
            "test_used": "Chi-square test of independence",
            "result": {
                "chi2": round(float(chi2), 6),
                "p_value": round(float(p), 6),
                "dof": int(dof),
                "significant": bool(p < 0.05),
            },
            "effect_size": {
                "metric": "cramers_v",
                "value": round(cramers_v, 4),
                "magnitude": (
                    "large" if cramers_v >= 0.5
                    else "medium" if cramers_v >= 0.3
                    else "small" if cramers_v >= 0.1
                    else "negligible"
                ),
            },
            "assumptions": {
                "min_expected": round(float(expected.min()), 2),
                "adequate_expected": bool(expected.min() >= 5),
            },
            "interpretation": (
                f"Chi-square test: χ² = {chi2:.4f}, df = {dof}, p = {p:.4f}. "
                f"{'Significant association detected.' if p < 0.05 else 'No significant association.'}"
            ),
            "warnings": (
                ["Some expected cell counts are < 5. Consider Fisher's exact test."]
                if expected.min() < 5 else []
            ),
            "reason": "Chi-square test of independence selected.",
            "group_stats": {},
        }

    else:
        raise ValueError(f"Unsupported analysis: {test_name}")

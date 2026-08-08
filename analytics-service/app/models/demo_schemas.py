"""Pydantic schemas for the Hypothesis Lab Demo.

These models are intentionally separate from the production schemas in
``schemas.py`` so the demo can evolve independently without touching the
existing analytics contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"


class NodeKind(str, Enum):
    ROOT = "root"
    QUESTION = "question"
    ANSWER = "answer"
    HYPOTHESIS = "hypothesis"
    ASSUMPTION = "assumption"
    ANALYSIS = "analysis"
    RESULT = "result"


class BranchStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPLETED = "completed"


class StartMode(str, Enum):
    GOAL = "goal"
    VARIABLE = "variable"
    OBSERVATION = "observation"
    FREE_TEXT = "free_text"


class VariableRole(str, Enum):
    OUTCOME = "outcome"
    PREDICTOR = "predictor"
    GROUP = "group"
    SUBJECT = "subject"
    TIME = "time"
    PAIRED = "paired"
    COVARIATE = "covariate"
    EFFECT_MODIFIER = "effect_modifier"


class HypothesisOrigin(str, Enum):
    """Track whether a hypothesis was formed before or after viewing results."""
    CONFIRMATORY = "confirmatory"
    EXPLORATORY = "exploratory"


# ---------------------------------------------------------------------------
# Dataset Profile
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    name: str
    dtype: ColumnType
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    total_count: int = 0

    # Numeric-specific
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    normality_hint: Optional[str] = None

    # Categorical-specific
    top_values: Optional[List[Dict[str, Any]]] = None
    cardinality: Optional[int] = None

    # Datetime-specific
    date_min: Optional[str] = None
    date_max: Optional[str] = None

    # Role suggestions from profiler
    suggested_roles: List[VariableRole] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    warnings: List[str] = Field(default_factory=list)
    suggested_group_columns: List[str] = Field(default_factory=list)
    suggested_time_columns: List[str] = Field(default_factory=list)
    suggested_subject_columns: List[str] = Field(default_factory=list)
    suggested_outcome_columns: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tree Nodes and Branches
# ---------------------------------------------------------------------------

class QuestionOption(BaseModel):
    id: str
    label: str
    description: Optional[str] = None
    disabled: bool = False
    disabled_reason: Optional[str] = None


class Question(BaseModel):
    id: str
    prompt: str
    explanation: Optional[str] = None
    options: List[QuestionOption] = Field(default_factory=list)
    allows_custom: bool = True
    allows_skip: bool = False
    category: Optional[str] = None


class VariableRoleAssignment(BaseModel):
    column: str
    role: VariableRole
    branch_id: str


class TreeNode(BaseModel):
    id: str
    parent_id: Optional[str] = None
    branch_id: str
    kind: NodeKind
    prompt: Optional[str] = None
    answer: Optional[str] = None
    answer_option_id: Optional[str] = None
    options: List[QuestionOption] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    is_exploratory: bool = False


class TreeBranch(BaseModel):
    id: str
    name: str = "Untitled Branch"
    status: BranchStatus = BranchStatus.ACTIVE
    node_ids: List[str] = Field(default_factory=list)
    is_primary: bool = False
    annotation: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class TreeState(BaseModel):
    nodes: Dict[str, TreeNode] = Field(default_factory=dict)
    branches: Dict[str, TreeBranch] = Field(default_factory=dict)
    active_branch_id: Optional[str] = None
    root_node_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Candidate Hypothesis & Analysis
# ---------------------------------------------------------------------------

class CandidateHypothesis(BaseModel):
    id: str
    branch_id: str
    statement: str
    null_hypothesis: Optional[str] = None
    alternative_hypothesis: Optional[str] = None
    variables: Dict[str, VariableRoleAssignment] = Field(default_factory=dict)
    origin: HypothesisOrigin = HypothesisOrigin.CONFIRMATORY
    is_primary: bool = False
    annotation: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AnalysisTradeoff(BaseModel):
    label: str
    description: str


class CandidateAnalysis(BaseModel):
    id: str
    hypothesis_id: str
    test_name: str
    display_name: str
    description: str
    is_suggested: bool = False
    suggestion_reason: Optional[str] = None
    tradeoffs: List[AnalysisTradeoff] = Field(default_factory=list)
    requirements: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    analysis_id: str
    hypothesis_id: Optional[str] = None
    test_used: str
    result: Dict[str, Any] = Field(default_factory=dict)
    effect_size: Dict[str, Any] = Field(default_factory=dict)
    assumptions: Dict[str, Any] = Field(default_factory=dict)
    interpretation: str = ""
    warnings: List[str] = Field(default_factory=list)
    follow_up_options: List[str] = Field(default_factory=list)
    applied_followups: List[str] = Field(default_factory=list)



class BranchValidation(BaseModel):
    is_valid: bool = True
    contradictions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_roles: List[VariableRole] = Field(default_factory=list)
    data_quality_issues: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    parent_node_id: str
    option_id: Optional[str] = None
    custom_answer: Optional[str] = None
    selected_columns: Optional[Dict[str, str]] = None  # column -> role


class ForkRequest(BaseModel):
    node_id: str
    new_answer: Optional[str] = None
    new_option_id: Optional[str] = None


class HypothesisAction(str, Enum):
    SAVE = "save"
    RENAME = "rename"
    ANNOTATE = "annotate"
    DUPLICATE = "duplicate"
    SET_PRIMARY = "set_primary"
    ARCHIVE = "archive"
    DELETE = "delete"


class HypothesisRequest(BaseModel):
    action: HypothesisAction
    hypothesis_id: Optional[str] = None
    statement: Optional[str] = None
    annotation: Optional[str] = None
    branch_id: Optional[str] = None


class AnalysisRequest(BaseModel):
    hypothesis_id: str
    analysis_id: str


class SessionResponse(BaseModel):
    session_id: str
    profile: DatasetProfile
    tree: TreeState
    current_question: Optional[Question] = None
    candidate_hypotheses: List[CandidateHypothesis] = Field(default_factory=list)
    candidate_analyses: List[CandidateAnalysis] = Field(default_factory=list)
    validation: Optional[BranchValidation] = None
    analysis_result: Optional[AnalysisResult] = None


class AnswerResponse(BaseModel):
    tree: TreeState
    current_question: Optional[Question] = None
    candidate_hypotheses: List[CandidateHypothesis] = Field(default_factory=list)
    candidate_analyses: List[CandidateAnalysis] = Field(default_factory=list)
    validation: Optional[BranchValidation] = None
    analysis_result: Optional[AnalysisResult] = None


class ForkResponse(BaseModel):
    new_branch_id: str
    tree: TreeState
    current_question: Optional[Question] = None


class AnalysisRunResponse(BaseModel):
    result: AnalysisResult
    tree: TreeState
    follow_up_question: Optional[Question] = None

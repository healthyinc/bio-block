

from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth.dependencies import AuthenticatedWallet, require_eip712_auth
from app.auth.rate_limiter import rate_limiter
from app.config import PINATA_GATEWAY_URL
from app.models.schemas import (
    DescriptiveResponse,
    HealthResponse,
    InferentialResponse,
    RegistryResultResponse,
    RegistryDatasetResponse,
    VisualizationResponse,
)
from app.services.descriptive import run_descriptive_analysis
from app.services.visualization import VALID_CHART_TYPES, generate_chart
from app.services.result_serializer import serialize_analytics_result
from app.services.ipfs_uploader import upload_result_to_ipfs
from app.services.chain_registry import register_on_chain, get_analytics_for_dataset
from app.services.audit_logger import AuditLogger
from app.services.inferential import (
    run_two_group_test,
    run_paired_test,
    run_one_sample_test,
    run_multi_group_test,
    run_chi_square_independence,
    run_chi_square_goodness_of_fit,
    run_pearson_correlation,
    run_spearman_correlation,
    run_correlation_analysis,
    run_correlation_matrix,
)
from app.utils.csv_parser import parse_csv

APP_VERSION = "0.1.0"
audit_logger = AuditLogger(log_path="audit_log.json")

app = FastAPI(
    title="Bio-Block Analytics API",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(version=APP_VERSION)


@app.post("/analytics/describe", response_model=DescriptiveResponse)
async def descriptive_analysis(
    auth: AuthenticatedWallet = Depends(require_eip712_auth),
    file: UploadFile = File(...),
    columns: Optional[str] = Form(None),
    store_on_ipfs: bool = Form(False, description="Upload result to IPFS"),
    register_on_chain_flag: bool = Form(
        False,
        alias="register_on_chain",
        description="Register result on-chain after IPFS upload",
    ),
):
    # Per-wallet rate limiting
    if not rate_limiter.check(auth.wallet_address):
        raise HTTPException(429, "Rate limit exceeded. Try again shortly.")

    contents = await file.read()
    df = parse_csv(contents)
    target_cols = None
    if columns and columns.strip() and columns.strip() != "string":
        target_cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() != "string"]
    analysis = run_descriptive_analysis(df, columns=target_cols or None)

    result_cid = None
    tx_hash = None

    if store_on_ipfs:
        result_doc = serialize_analytics_result(
            analysis_type="descriptive",
            source_cid=auth.dataset_cid,
            wallet_address=auth.wallet_address,
            results=analysis["results"],
            row_count=len(df),
            columns=analysis["columns_analyzed"],
            parameters={"columns": target_cols},
        )
        result_cid = await upload_result_to_ipfs(
            result_data=result_doc,
            analysis_type="descriptive",
            source_cid=auth.dataset_cid,
        )

        if register_on_chain_flag and result_cid:
            tx_hash = await register_on_chain(
                source_cid=auth.dataset_cid,
                result_cid=result_cid,
                analysis_type="descriptive",
                analyst_address=auth.wallet_address,
            )

    audit_logger.log_operation(
        operation="DESCRIBE",
        wallet_address=auth.wallet_address,
        dataset_cid=auth.dataset_cid,
        details=f"cols={analysis['columns_analyzed']}, rows={len(df)}",
    )

    return DescriptiveResponse(
        source_dataset_cid=auth.dataset_cid,
        row_count=len(df),
        columns_analyzed=analysis["columns_analyzed"],
        results=analysis["results"],
        result_cid=result_cid,
        tx_hash=tx_hash,
    )


@app.post("/analytics/visualize", response_model=VisualizationResponse)
async def visualize(
    auth: AuthenticatedWallet = Depends(require_eip712_auth),
    file: UploadFile = File(...),
    chart_type: str = Form(..., description=f"One of: {', '.join(VALID_CHART_TYPES)}"),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None),
    columns: Optional[str] = Form(None, description="Comma-separated column names for multi-column charts"),
    group_column: Optional[str] = Form(None),
    aggregation: Optional[str] = Form("count", description="Aggregation: count, mean, sum"),
    bins: Optional[int] = Form(20, description="Number of bins for histograms"),
    store_on_ipfs: bool = Form(False, description="Upload result to IPFS"),
    register_on_chain_flag: bool = Form(
        False,
        alias="register_on_chain",
        description="Register result on-chain after IPFS upload",
    ),
):
    # Per-wallet rate limiting
    if not rate_limiter.check(auth.wallet_address):
        raise HTTPException(429, "Rate limit exceeded. Try again shortly.")

    contents = await file.read()

    if chart_type not in VALID_CHART_TYPES:
        raise HTTPException(
            400,
            f"Unsupported chart type '{chart_type}'. Valid types: {VALID_CHART_TYPES}",
        )

    df = parse_csv(contents)

    # Swagger UI sends "string" as default placeholder — treat as empty
    def _clean(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        val = val.strip()
        if val == "" or val == "string":
            return None
        return val

    x_col = _clean(x_column)
    y_col = _clean(y_column)
    grp_col = _clean(group_column)
    cols_list = None
    if columns and _clean(columns):
        cols_list = [c.strip() for c in columns.split(",") if c.strip() and c.strip() != "string"]
        if not cols_list:
            cols_list = None

    try:
        result = generate_chart(
            df=df,
            chart_type=chart_type,
            x_column=x_col,
            y_column=y_col,
            columns=cols_list,
            group_column=grp_col,
            aggregation=_clean(aggregation) or "count",
            bins=bins or 20,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    result_cid = None
    tx_hash = None

    if store_on_ipfs:
        # Store chart_config in the IPFS result (NOT the base64 image —
        # images would be separate IPFS objects per the proposal).
        result_doc = serialize_analytics_result(
            analysis_type="graphical",
            source_cid=auth.dataset_cid,
            wallet_address=auth.wallet_address,
            results={"chart_config": result["chart_config"]},
            row_count=len(df),
            columns=[x_col, y_col] if x_col else list(df.columns[:5]),
            parameters={
                "chart_type": chart_type,
                "x_column": x_col,
                "y_column": y_col,
                "group_column": grp_col,
                "aggregation": _clean(aggregation) or "count",
                "bins": bins or 20,
            },
        )
        result_cid = await upload_result_to_ipfs(
            result_data=result_doc,
            analysis_type="graphical",
            source_cid=auth.dataset_cid,
        )

        if register_on_chain_flag and result_cid:
            tx_hash = await register_on_chain(
                source_cid=auth.dataset_cid,
                result_cid=result_cid,
                analysis_type="graphical",
                analyst_address=auth.wallet_address,
            )

    audit_logger.log_operation(
        operation="VISUALIZE",
        wallet_address=auth.wallet_address,
        dataset_cid=auth.dataset_cid,
        details=f"chart={chart_type}, rows={len(df)}",
    )

    return VisualizationResponse(
        source_dataset_cid=auth.dataset_cid,
        chart_type=chart_type,
        chart_config=result["chart_config"],
        image=result["image"],
        row_count=len(df),
        result_cid=result_cid,
        tx_hash=tx_hash,
    )


VALID_TEST_TYPES = {
    "t_test": ["independent", "paired", "one_sample"],
    "anova": ["one_way", "two_way", "repeated_measures"],
    "chi_square": ["independence", "goodness_of_fit"],
    "correlation": ["pearson", "spearman", "auto", "matrix"],
}
VALID_ALTERNATIVES = ["two-sided", "less", "greater"]


@app.post("/analytics/infer")
async def inferential_analysis(
    auth: AuthenticatedWallet = Depends(require_eip712_auth),
    file: UploadFile = File(...),
    test_type: str = Form(
        ...,
        description="Test category: t_test, anova, chi_square, correlation",
    ),
    test_subtype: Optional[str] = Form(
        None,
        description=(
            "Test subtype. For t_test: independent, paired, one_sample. "
            "For anova: one_way, two_way, repeated_measures. "
            "For chi_square: independence, goodness_of_fit. "
            "For correlation: pearson, spearman, auto, matrix."
        ),
    ),
    numeric_column: Optional[str] = Form(
        None,
        description="Target numeric column",
    ),
    group_column: Optional[str] = Form(
        None,
        description="Grouping column (for independent t-test, ANOVA)",
    ),
    numeric_column_2: Optional[str] = Form(
        None,
        description="Second numeric column (for paired test)",
    ),
    population_mean: Optional[float] = Form(
        None, description="Known population mean (for one_sample test)"
    ),
    factor_column_2: Optional[str] = Form(
        None, description="Second factor column (for two-way ANOVA)",
    ),
    repeated_columns: Optional[str] = Form(
        None,
        description="Comma-separated condition columns (for repeated measures)",
    ),
    column_1: Optional[str] = Form(
        None,
        description="First column (for chi-square independence or correlation)",
    ),
    column_2: Optional[str] = Form(
        None,
        description="Second column (for chi-square independence or correlation)",
    ),
    target_columns: Optional[str] = Form(
        None,
        description="Comma-separated columns (for correlation matrix)",
    ),
    correlation_method: Optional[str] = Form(
        "auto",
        description="Correlation method: pearson, spearman, auto",
    ),
    alpha: float = Form(0.05, description="Significance level"),
    alternative: str = Form(
        "two-sided",
        description="Alternative hypothesis: two-sided, less, greater",
    ),
    store_on_ipfs: bool = Form(False, description="Upload result to IPFS"),
    register_on_chain_flag: bool = Form(
        False,
        alias="register_on_chain",
        description="Register result on-chain after IPFS upload",
    ),
):
    """Run inferential statistical tests with automatic test selection.

    Supports t-tests (Student's, Welch's, paired, one-sample),
    Mann-Whitney U, Wilcoxon signed-rank, ANOVA (one-way, two-way,
    repeated measures), Kruskal-Wallis, and Friedman tests.

    The engine automatically selects the appropriate test based on
    normality (Shapiro-Wilk/KS) and equal variance (Levene's) checks.
    """
    # Per-wallet rate limiting
    if not rate_limiter.check(auth.wallet_address):
        raise HTTPException(429, "Rate limit exceeded. Try again shortly.")

    # Swagger UI sends "string" as default placeholder — treat as empty
    def _clean(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        val = val.strip()
        if val == "" or val == "string":
            return None
        return val

    cleaned_test_type = _clean(test_type)
    cleaned_subtype = _clean(test_subtype)
    cleaned_numeric = _clean(numeric_column)
    cleaned_group = _clean(group_column)
    cleaned_numeric_2 = _clean(numeric_column_2)
    cleaned_factor_2 = _clean(factor_column_2)
    cleaned_repeated = _clean(repeated_columns)
    cleaned_col1 = _clean(column_1)
    cleaned_col2 = _clean(column_2)
    cleaned_target_cols = _clean(target_columns)
    cleaned_corr_method = _clean(correlation_method) or "auto"

    # Validate test_type
    if cleaned_test_type not in VALID_TEST_TYPES:
        raise HTTPException(
            400,
            f"Unsupported test type '{cleaned_test_type}'. "
            f"Valid types: {list(VALID_TEST_TYPES.keys())}",
        )

    # Validate test_subtype
    valid_subtypes = VALID_TEST_TYPES[cleaned_test_type]
    if cleaned_subtype and cleaned_subtype not in valid_subtypes:
        raise HTTPException(
            400,
            f"Invalid test_subtype '{cleaned_subtype}' for test_type "
            f"'{cleaned_test_type}'. Valid subtypes: {valid_subtypes}",
        )

    # Validate alternative
    if alternative not in VALID_ALTERNATIVES:
        raise HTTPException(
            400,
            f"Invalid alternative '{alternative}'. "
            f"Valid values: {VALID_ALTERNATIVES}",
        )

    contents = await file.read()
    df = parse_csv(contents)

    try:
        if cleaned_test_type == "t_test":
            subtype = cleaned_subtype or "independent"
            if subtype == "independent":
                if not cleaned_numeric or not cleaned_group:
                    raise HTTPException(
                        400,
                        "Independent t-test requires 'numeric_column' and "
                        "'group_column'.",
                    )
                analysis = run_two_group_test(
                    df, cleaned_numeric, cleaned_group, alpha, alternative
                )
            elif subtype == "paired":
                col1 = cleaned_numeric
                col2 = cleaned_numeric_2
                if not col1 or not col2:
                    raise HTTPException(
                        400,
                        "Paired test requires 'numeric_column' and "
                        "'numeric_column_2'.",
                    )
                analysis = run_paired_test(
                    df, col1, col2, alpha, alternative
                )
            elif subtype == "one_sample":
                if not cleaned_numeric or population_mean is None:
                    raise HTTPException(
                        400,
                        "One-sample test requires 'numeric_column' and "
                        "'population_mean'.",
                    )
                analysis = run_one_sample_test(
                    df, cleaned_numeric, population_mean, alpha, alternative
                )
            else:
                raise HTTPException(
                    400, f"Unknown t_test subtype: {subtype}"
                )

        elif cleaned_test_type == "anova":
            subtype = cleaned_subtype or "one_way"
            if subtype == "one_way":
                if not cleaned_numeric or not cleaned_group:
                    raise HTTPException(
                        400,
                        "One-way ANOVA requires 'numeric_column' and "
                        "'group_column'.",
                    )
                analysis = run_multi_group_test(
                    df, cleaned_numeric, cleaned_group, alpha
                )
            elif subtype == "two_way":
                if not cleaned_numeric or not cleaned_group or not cleaned_factor_2:
                    raise HTTPException(
                        400,
                        "Two-way ANOVA requires 'numeric_column', "
                        "'group_column', and 'factor_column_2'.",
                    )
                from app.services.inferential import run_two_way_anova
                analysis = run_two_way_anova(
                    df, cleaned_numeric, cleaned_group, cleaned_factor_2, alpha
                )
            elif subtype == "repeated_measures":
                if not cleaned_repeated:
                    raise HTTPException(
                        400,
                        "Repeated-measures ANOVA requires 'repeated_columns' "
                        "(comma-separated list of condition columns).",
                    )
                condition_cols = [
                    c.strip() for c in cleaned_repeated.split(",")
                    if c.strip()
                ]
                if len(condition_cols) < 3:
                    raise HTTPException(
                        400,
                        "Repeated-measures ANOVA requires at least 3 "
                        "condition columns.",
                    )
                from app.services.inferential import run_repeated_measures_anova
                analysis = run_repeated_measures_anova(
                    df, condition_cols, alpha
                )
            else:
                raise HTTPException(
                    400, f"Unknown anova subtype: {subtype}"
                )

        elif cleaned_test_type == "chi_square":
            subtype = cleaned_subtype or "independence"
            if subtype == "independence":
                c1 = cleaned_col1 or cleaned_numeric
                c2 = cleaned_col2 or cleaned_group
                if not c1 or not c2:
                    raise HTTPException(
                        400,
                        "Chi-square independence test requires 'column_1' "
                        "and 'column_2' (or 'numeric_column' and "
                        "'group_column').",
                    )
                analysis = run_chi_square_independence(
                    df, c1, c2, alpha
                )
            elif subtype == "goodness_of_fit":
                col = cleaned_col1 or cleaned_numeric
                if not col:
                    raise HTTPException(
                        400,
                        "Chi-square goodness-of-fit requires 'column_1' "
                        "(or 'numeric_column').",
                    )
                analysis = run_chi_square_goodness_of_fit(
                    df, col, alpha=alpha
                )
            else:
                raise HTTPException(
                    400, f"Unknown chi_square subtype: {subtype}"
                )

        elif cleaned_test_type == "correlation":
            subtype = cleaned_subtype or "auto"
            if subtype == "matrix":
                cols_list = None
                if cleaned_target_cols:
                    cols_list = [
                        c.strip() for c in cleaned_target_cols.split(",")
                        if c.strip()
                    ]
                analysis = run_correlation_matrix(
                    df, columns=cols_list, method=cleaned_corr_method, alpha=alpha
                )
            else:
                c1 = cleaned_col1 or cleaned_numeric
                c2 = cleaned_col2 or cleaned_group
                if not c1 or not c2:
                    raise HTTPException(
                        400,
                        "Correlation analysis requires 'column_1' and "
                        "'column_2' (or 'numeric_column' and "
                        "'group_column').",
                    )
                if subtype == "pearson":
                    analysis = run_pearson_correlation(df, c1, c2, alpha)
                elif subtype == "spearman":
                    analysis = run_spearman_correlation(df, c1, c2, alpha)
                elif subtype == "auto":
                    analysis = run_correlation_analysis(df, c1, c2, alpha)
                else:
                    raise HTTPException(
                        400, f"Unknown correlation subtype: {subtype}"
                    )

        else:
            raise HTTPException(400, f"Unknown test type: {cleaned_test_type}")

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # Flatten analysis dict into the top-level response and add metadata
    response = {
        "analysis_type": "inferential",
        "source_dataset_cid": auth.dataset_cid,
        "test_category": cleaned_test_type,
        "test_subtype": cleaned_subtype or (
            "independent" if cleaned_test_type == "t_test" else "one_way"
        ),
        "row_count": len(df),
    }
    # Merge analysis results into the response (flattened)
    response.update(analysis)

    result_cid = None
    tx_hash = None

    if store_on_ipfs:
        result_doc = serialize_analytics_result(
            analysis_type="inferential",
            source_cid=auth.dataset_cid,
            wallet_address=auth.wallet_address,
            results=analysis,
            row_count=len(df),
            columns=[
                c for c in [
                    cleaned_numeric, cleaned_group,
                    cleaned_numeric_2, cleaned_factor_2,
                ] if c is not None
            ],
            parameters={
                "test_type": cleaned_test_type,
                "test_subtype": cleaned_subtype,
                "numeric_column": cleaned_numeric,
                "group_column": cleaned_group,
                "population_mean": population_mean,
                "alpha": alpha,
                "alternative": alternative,
            },
        )
        result_cid = await upload_result_to_ipfs(
            result_data=result_doc,
            analysis_type="inferential",
            source_cid=auth.dataset_cid,
        )

        if register_on_chain_flag and result_cid:
            tx_hash = await register_on_chain(
                source_cid=auth.dataset_cid,
                result_cid=result_cid,
                analysis_type="inferential",
                analyst_address=auth.wallet_address,
            )

    if result_cid:
        response["result_cid"] = result_cid
    if tx_hash:
        response["tx_hash"] = tx_hash

    audit_logger.log_operation(
        operation="INFER",
        wallet_address=auth.wallet_address,
        dataset_cid=auth.dataset_cid,
        details=f"test={cleaned_test_type}/{cleaned_subtype}, rows={len(df)}",
    )

    return response


@app.get("/analytics/results/{result_cid}", response_model=RegistryResultResponse)
async def get_result(result_cid: str):
    """Fetch an analytics result JSON from IPFS by its CID."""
    url = f"{PINATA_GATEWAY_URL}/{result_cid}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                raise HTTPException(404, f"Result CID not found on IPFS: {result_cid}")
            resp.raise_for_status()
            data = resp.json()
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            502, f"IPFS gateway error: {exc.response.status_code}"
        )
    except httpx.ConnectError:
        raise HTTPException(502, "IPFS gateway unreachable")
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch from IPFS: {exc}")

    return RegistryResultResponse(result_cid=result_cid, data=data)


@app.get("/analytics/dataset/{dataset_cid}", response_model=RegistryDatasetResponse)
async def get_dataset_results(dataset_cid: str):
    """Query the AnalyticsRegistry contract for all result CIDs linked to a dataset."""
    result_cids = get_analytics_for_dataset(dataset_cid)
    return RegistryDatasetResponse(dataset_cid=dataset_cid, result_cids=result_cids)


# -- Audit Logging Phase 2 endpoints --

@app.get("/audit/verify/{entry_id}")
async def verify_audit_entry(entry_id: str):
    """SHA-256 hash-chain validation for a single log entry."""
    result = audit_logger.verify_entry(entry_id)
    if not result.get("valid") and result.get("reason") == "entry not found":
        raise HTTPException(404, "Audit entry not found")
    return result


@app.get("/audit/verify")
async def verify_audit_chain():
    """Validate the entire hash-chain. Returns first broken link if any."""
    return audit_logger.verify_chain()


@app.get("/audit/logs")
async def get_audit_logs(
    wallet_address: Optional[str] = None,
    operation: Optional[str] = None,
    dataset_cid: Optional[str] = None,
    limit: int = 50,
):
    logs = audit_logger.query_logs(
        wallet_address=wallet_address,
        operation=operation,
        dataset_cid=dataset_cid,
        limit=limit,
    )
    return {"logs": logs, "total": len(logs)}


@app.get("/audit/logs/{entry_id}")
async def get_audit_entry(entry_id: str):
    entry = audit_logger.get_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Audit entry not found")
    verification = audit_logger.verify_entry(entry_id)
    entry["integrity_verified"] = verification["valid"]
    return entry


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3003)

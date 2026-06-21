

from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth.dependencies import AuthenticatedWallet, require_eip712_auth
from app.auth.rate_limiter import rate_limiter
from app.models.schemas import (
    DescriptiveResponse,
    HealthResponse,
    RegistryResultResponse,
    RegistryDatasetResponse,
    VisualizationResponse,
)
from app.services.descriptive import run_descriptive_analysis
from app.services.visualization import VALID_CHART_TYPES, generate_chart
from app.utils.csv_parser import parse_csv

APP_VERSION = "0.1.0"

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

    return DescriptiveResponse(
        source_dataset_cid=auth.dataset_cid,
        row_count=len(df),
        columns_analyzed=analysis["columns_analyzed"],
        results=analysis["results"],
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

    return VisualizationResponse(
        source_dataset_cid=auth.dataset_cid,
        chart_type=chart_type,
        chart_config=result["chart_config"],
        image=result["image"],
        row_count=len(df),
    )


@app.post("/analytics/infer")
async def infer():
    raise HTTPException(501, "Not yet implemented.")


@app.get("/analytics/results/{result_cid}", response_model=RegistryResultResponse)
async def get_result(result_cid: str):
    # Stub: will be integrated with IPFS controller
    return RegistryResultResponse(
        result_cid=result_cid,
        data={"status": "mock", "note": "IPFS fetch not yet implemented"}
    )


@app.get("/analytics/dataset/{dataset_cid}", response_model=RegistryDatasetResponse)
async def get_dataset_results(dataset_cid: str):
    # Stub: will be integrated with AnalyticsRegistry contract
    return RegistryDatasetResponse(
        dataset_cid=dataset_cid,
        result_cids=["QmPlaceholder1", "QmPlaceholder2"]
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3003)

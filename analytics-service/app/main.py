"""Bio-Block Analytics API — port 3003."""

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth.eip712 import verify_signature
from app.models.schemas import DescriptiveResponse, HealthResponse
from app.services.descriptive import run_descriptive_analysis
from app.utils.csv_parser import parse_csv

APP_VERSION = "0.1.0"

app = FastAPI(
    title="Bio-Block Analytics API",
    description="Decentralized analytics for the Bio-Block health data marketplace.",
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
    file: UploadFile = File(...),
    wallet_address: str = Form(...),
    dataset_cid: str = Form(...),
    signature: str = Form(...),
    timestamp: int = Form(...),
    nonce: int = Form(...),
    request_hash: str = Form(...),
    columns: Optional[str] = Form(None),
):
    """Descriptive statistics on an uploaded CSV. Requires EIP-712 signature."""
    contents = await file.read()

    if not verify_signature(
        wallet_address, dataset_cid, signature, timestamp, nonce, request_hash
    ):
        raise HTTPException(401, "Invalid or expired signature.")

    df = parse_csv(contents)
    target_cols = columns.split(",") if columns else None
    analysis = run_descriptive_analysis(df, columns=target_cols)

    return DescriptiveResponse(
        source_dataset_cid=dataset_cid,
        row_count=len(df),
        columns_analyzed=analysis["columns_analyzed"],
        results=analysis["results"],
    )


@app.post("/analytics/visualize")
async def visualize():
    """TODO: Chart generation with matplotlib/plotly."""
    raise HTTPException(501, "Not yet implemented.")


@app.post("/analytics/infer")
async def infer():
    """TODO: Hypothesis testing with scipy/statsmodels."""
    raise HTTPException(501, "Not yet implemented.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3003)

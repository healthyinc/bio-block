

from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth.dependencies import AuthenticatedWallet, require_eip712_auth
from app.auth.rate_limiter import rate_limiter
from app.models.schemas import DescriptiveResponse, HealthResponse
from app.services.descriptive import run_descriptive_analysis
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
    target_cols = columns.split(",") if columns else None
    analysis = run_descriptive_analysis(df, columns=target_cols)

    return DescriptiveResponse(
        source_dataset_cid=auth.dataset_cid,
        row_count=len(df),
        columns_analyzed=analysis["columns_analyzed"],
        results=analysis["results"],
    )


@app.post("/analytics/visualize")
async def visualize():
    raise HTTPException(501, "Not yet implemented.")


@app.post("/analytics/infer")
async def infer():
    raise HTTPException(501, "Not yet implemented.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3003)

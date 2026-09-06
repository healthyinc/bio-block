# Bio-Block Analytics Service

Decentralized analytics microservice for the [Bio-Block](https://github.com/healthyinc/bio-block) health data marketplace. Part of **GSoC 2026**.

## Architecture

```
Port 3003 — Analytics API (this service)
Port 3002 — Python Backend (ChromaDB search, image anonymization)
Port 3001 — JS Backend (file processing, IPFS uploads)
Port 3000 — Next.js Frontend
```

## Quick Start

### With Docker

```bash
cd analytics-service
docker-compose up --build
```

### Without Docker

```bash
cd analytics-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3003 --reload
```

API docs: [http://localhost:3003/docs](http://localhost:3003/docs)

## Endpoints

| Method | Endpoint               | Status       | Description            |
| ------ | ---------------------- | ------------ | ---------------------- |
| `GET`  | `/health`              | ✅ Done      | Service health check   |
| `POST` | `/analytics/describe`  | ✅ Done      | Descriptive statistics |
| `POST` | `/analytics/visualize` | ✅ Done      | Chart generation (7 types) |
| `POST` | `/analytics/infer`     | 🔜 Week 6-7 | Hypothesis testing     |
| `GET`  | `/analytics/results/{result_cid}` | ✅ Stub | Registry result lookup |
| `GET`  | `/analytics/dataset/{dataset_cid}` | ✅ Stub | Dataset results lookup |

## Testing

```bash
cd analytics-service
pip install -r requirements.txt
APP_ENV=test pytest tests/ -v
```

## Project Structure

```
analytics-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── auth/
│   │   ├── dependencies.py  # EIP-712 FastAPI dependency
│   │   ├── eip712.py        # EIP-712 signature verification
│   │   └── rate_limiter.py  # Per-wallet sliding-window rate limiter
│   ├── services/
│   │   ├── descriptive.py   # Descriptive statistics engine
│   │   └── visualization.py # Chart generation (histogram, scatter, box, heatmap, bar, line, pie)
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── utils/
│       └── csv_parser.py    # Robust CSV/Excel parsing with size limits
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_descriptive.py          # Stats engine tests
│   ├── test_csv_parser.py           # CSV parser tests
│   ├── test_auth.py                 # Auth verification tests
│   ├── test_auth_dependency.py      # Auth dependency integration tests
│   ├── test_rate_limiter.py         # Rate limiter tests
│   ├── test_visualization.py        # Visualization service tests
│   ├── test_visualization_categorical.py  # Categorical chart tests
│   └── fixtures/
│       └── patient_demographics.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

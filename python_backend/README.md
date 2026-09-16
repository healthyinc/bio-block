# Python backend

FastAPI service for ChromaDB search and image PHI anonymization.

## Setup (virtual environment required)

Do **not** install dependencies into the system Python. Always use a project venv.

### macOS / Linux

```bash
cd python_backend
chmod +x setup_venv.sh
./setup_venv.sh
source .venv/bin/activate
uvicorn main:app --reload --port 3002
```

### Windows (PowerShell)

```powershell
cd python_backend
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 3002
```

### Optional: use `uv`

If [uv](https://github.com/astral-sh/uv) is installed, the setup scripts use it automatically for faster installs. You can also run:

```bash
cd python_backend
uv venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## Tests

```bash
source .venv/bin/activate
# start the server in another terminal, then:
cd tests
python -m unittest test_api.py
```

## Notes

- Default port: `3002`
- ChromaDB data is stored under `./chroma_db` (gitignored)
- `.venv/` is gitignored — never commit the virtual environment

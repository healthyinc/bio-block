.PHONY: python-setup python-run python-test help

PYTHON_DIR := python_backend
VENV := $(PYTHON_DIR)/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Bio-Block make targets:"
	@echo "  make python-setup  Create .venv and install python_backend deps"
	@echo "  make python-run    Run FastAPI backend on port 3002"
	@echo "  make python-test   Run python_backend unit tests (server must be up)"

python-setup:
	@chmod +x $(PYTHON_DIR)/setup_venv.sh
	@$(PYTHON_DIR)/setup_venv.sh

python-run: $(VENV)
	@cd $(PYTHON_DIR) && . .venv/bin/activate && uvicorn main:app --reload --port 3002

python-test: $(VENV)
	@cd $(PYTHON_DIR)/tests && ../.venv/bin/python -m unittest test_api.py

$(VENV):
	@$(MAKE) python-setup

PYTHON = .venv/Scripts/python
PIP    = .venv/Scripts/pip

.PHONY: setup venv install generate-leads validate test clean

# Set up full environment
setup: venv install

venv:
	python -m venv .venv

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Section 4.1 — Lead generation
generate-leads:
	$(PYTHON) -m src.simulation.lead_generator --config configs/data_config.yaml

# Validation only (no generation)
validate:
	$(PYTHON) -m src.simulation.lead_generator --config configs/data_config.yaml --validate-only

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Run unit tests only (faster)
test-unit:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

# Clean generated data (keeps code)
clean-data:
	rm -f data/raw/*.parquet data/processed/**/*.parquet

# Clean everything including venv
clean:
	rm -rf .venv __pycache__ .pytest_cache

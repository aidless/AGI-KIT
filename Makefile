.PHONY: help install dev test test-cov lint format type-check docs serve clean

help:                   ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:                ## Install package in production mode
	pip install -e .

dev:                    ## Install with dev + optional deps
	pip install -e ".[dev,playwright]"

test:                   ## Run unit tests
	pytest tests/unit -v

test-integration:       ## Run integration tests (requires services)
	pytest tests/integration -v -m integration

test-cov:               ## Run tests with coverage
	pytest --cov=agi_kit --cov-report=html --cov-report=term tests/

lint:                   ## Run ruff linter
	ruff check src tests

format:                 ## Format code with black + isort
	black src tests && isort src tests

type-check:             ## Run mypy
	mypy src/agi_kit

docs:                   ## Build documentation
	mkdocs build --strict

docs-serve:             ## Live-preview docs
	mkdocs serve

serve:                  ## Start FastAPI server
	agi-kit serve --reload

clean:                  ## Clean build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
.PHONY: help install test test-integration test-quick lint format typecheck coverage clean pre-commit

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

test: ## Run unit tests
	uv run pytest tests/ -m "not integration" -v

test-integration: ## Run integration tests
	uv run pytest tests/ -m "integration" -v

test-quick: ## Run tests without coverage
	uv run pytest tests/ -v --no-cov

lint: ## Lint with ruff
	uv run ruff check src/ tests/

format: ## Format with ruff
	uv run ruff format src/ tests/

typecheck: ## Type check with ty
	uv run ty check src/

coverage: ## Run tests with coverage report
	uv run pytest tests/ -m "not integration" --cov --cov-report=term-missing

clean: ## Remove build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov __pycache__ **/__pycache__ *.pyc **/*.pyc

pre-commit: ## Install and run pre-commit hooks
	pre-commit install
	pre-commit run --all-files

.PHONY: install test lint format typecheck coverage clean

install:
	uv sync

test:
	uv run pytest tests/ -v

test-quick:
	uv run pytest tests/ -v --no-cov

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run ty check src/

coverage:
	uv run pytest tests/ --cov=src/fcp --cov-report=term-missing --cov-report=html --cov-fail-under=100

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov __pycache__ **/__pycache__ *.pyc **/*.pyc

pre-commit:
	pre-commit install
	pre-commit run --all-files

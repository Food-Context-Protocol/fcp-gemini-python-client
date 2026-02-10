.PHONY: install test test-integration lint format typecheck coverage clean

install:
	uv sync

test:
	uv run pytest tests/ -m "not integration" -v

test-integration:
	uv run pytest tests/ -m "integration" -v

test-quick:
	uv run pytest tests/ -v --no-cov

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run ty check src/

coverage:
	uv run pytest tests/ -m "not integration" --cov --cov-report=term-missing

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov __pycache__ **/__pycache__ *.pyc **/*.pyc

pre-commit:
	pre-commit install
	pre-commit run --all-files

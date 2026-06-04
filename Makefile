# Makefile for Epslionic-Colab Agent
# Patterns from huggingface/transformers (120k+ stars)

.PHONY: help install dev-install style lint typecheck test test-cov build clean distclean docker docker-run

help:  # Show this help
	@echo "Epslionic-Colab Agent - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install package in editable mode"
	@echo "  make dev-install     Install all dev dependencies"
	@echo "  make style           Run ruff formatter and isort"
	@echo "  make lint            Run ruff linter"
	@echo "  make typecheck       Run mypy type checking"
	@echo "  make test            Run tests (quick)"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make test-all        Run all tests (including slow)"
	@echo "  make build           Build wheel and sdist"
	@echo "  make clean           Remove build artifacts"
	@echo "  make distclean       Full cleanup (including node_modules, etc)"
	@echo "  make docker          Build Docker image"
	@echo "  make docker-run      Run in Docker container"

install:  # Install package in editable mode
	pip install -e .

dev-install:  # Install dev dependencies
	pip install -e ".[dev,dashboard]"

style:  # Run ruff formatter and isort
	ruff format epsionic tests
	ruff check --fix epsionic tests
	isort epsionic tests

lint:  # Run ruff linter
	ruff check epsionic tests
	ruff format --check epsionic tests

typecheck:  # Run mypy type checking
	mypy epsionic

test:  # Run quick tests
	python -m pytest tests/ -v --timeout=30 -x -q

test-cov:  # Run tests with coverage
	python -m pytest tests/ -v --timeout=30 --cov=epsionic --cov-report=term --cov-report=html

test-all:  # Run all tests (including slow)
	python -m pytest tests/ -v --timeout=120 --runslow

precommit:  # Install and run pre-commit
	pre-commit install
	pre-commit run --all-files

really-clean: distclean  # Nuclear cleanup
	rm -rf .tox/ *.egg-info/
	git clean -fdx -e .env

docs:  # Build documentation
	mkdocs build 2>/dev/null || echo "mkdocs not installed; install with: pip install mkdocs mkdocs-material"

build:  # Build wheel and sdist
	python -m build
	@echo "Build complete. Check dist/ directory."

clean:  # Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

distclean: clean  # Full cleanup
	rm -rf .mypy_cache/ .ruff_cache/
	rm -rf memory/ models/ datasets/ logs/ workspace*/

docker:  # Build Docker image
	docker build -t epsionic-agent:latest .

docker-run:  # Run in Docker container
	docker run --rm -it --gpus all \
		-e OPENAI_API_KEY=$${OPENAI_API_KEY:-""} \
		-e HF_TOKEN=$${HF_TOKEN:-""} \
		-v $${PWD}/data:/workspace/data \
		epsionic-agent:latest --autonomous

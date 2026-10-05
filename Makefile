.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help install test lint fmt check demo clean

help: ## show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## create the venv with runtime and dev dependencies
	$(UV) sync

test: ## run the unit tests (no network)
	$(UV) run pytest

lint: ## ruff check + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

fmt: ## auto-fix lint issues and reformat
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

check: lint test ## everything CI would run

demo: ## search, vet and confirm the field used while developing (Gemini, sector 72)
	$(UV) run varhunt search 100.0 20.0 --radius 0.2 --sector 72 --out runs/demo
	$(UV) run varhunt vet runs/demo
	$(UV) run varhunt confirm runs/demo

clean: ## remove caches and build artifacts
	rm -rf build dist src/*.egg-info .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

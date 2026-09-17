.PHONY: sync
sync:
	uv sync --all-extras --all-packages

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: test
test:
	uv run --extra cpu pytest

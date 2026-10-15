# Dev checks. `uv sync --group dev` once; then `make check` before every commit.
.PHONY: lint test test-live check check-pythons release-check

lint:
	uv run ruff check src tests bench

test:
	uv run pytest -q

test-live:
	uv run pytest -q -m live

check: lint test

# The macOS half of CI, run here: CI tests Linux only, by choice. Before a push,
# run the suite on the oldest and newest supported Pythons in their own venvs, with /Volumes hidden as on a CI runner.
PYTHONS ?= 3.10 3.14
check-pythons:
	@for v in $(PYTHONS); do \
		echo "== Python $$v =="; \
		UV_PROJECT_ENVIRONMENT=.venv-py$$v CALLO_TEST_HIDE_VOLUMES=1 uv run --python $$v --group dev pytest -q || exit 1; \
	done

# Before publishing (network; not part of `check`): refuses the placeholder BUILTIN_SET_REVISION, then fetches the
# built-in bias set from the pinned HF revision into a throwaway cache and verifies its sha256.
release-check:
	uv run python -m bench.release_check

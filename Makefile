run:
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb src

clean:
	rm -rf */*.pyc */__pycache__/ .mypy_cache/

lint:
lint-strict:
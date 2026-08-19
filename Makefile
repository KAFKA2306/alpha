.PHONY: help check

help:
	@echo "Available commands:"
	@echo "  check  Compile source, run the registry smoke test, and verify a clean checkout"

check:
	PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest -v tests.test_registry_smoke
	git diff --exit-code

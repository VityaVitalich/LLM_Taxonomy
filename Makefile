#* Formatters
.PHONY: codestyle
codestyle:
  isort ./DataInterface
  python -m black .
  ruff check .
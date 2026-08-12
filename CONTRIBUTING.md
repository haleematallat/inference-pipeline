# Contributing

Create a focused branch, add tests for behavior changes, and run:

~~~bash
ruff check .
mypy
pytest -q
~~~

Keep optional integrations isolated from core imports. New inference runners and backends
should use the existing registries and must not require external infrastructure in the default
test suite.

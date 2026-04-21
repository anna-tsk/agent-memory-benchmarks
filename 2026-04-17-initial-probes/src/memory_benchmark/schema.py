"""Schema-related helpers for benchmark files.

TODO:
- Add typed data models once the JSON schema settles.
- Expose the canonical schema path.
- Add lightweight validation helpers around `jsonschema` or another validator.
"""

from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schema" / "benchmark.schema.json"
"""Path to the benchmark JSON Schema file."""

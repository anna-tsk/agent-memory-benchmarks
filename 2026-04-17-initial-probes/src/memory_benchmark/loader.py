"""Loading utilities for benchmark trajectory files.

TODO:
- Load benchmark JSON from `data/examples` or user-provided paths.
- Validate loaded data against `data/schema/benchmark.schema.json`.
- Normalize timestamps into datetime objects if needed by future evaluation.
- Preserve raw provenance and scope metadata for downstream analysis.
"""

from pathlib import Path
from typing import Any


def load_benchmark(path: str | Path) -> dict[str, Any]:
    """Load a benchmark JSON file.

    TODO: Implement JSON parsing and schema validation.
    """
    raise NotImplementedError("Benchmark loading is not implemented yet.")

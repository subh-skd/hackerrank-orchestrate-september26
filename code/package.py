"""Build code.zip for submission.

    python code/package.py

Includes the solution, configuration, trained model, tests, docs and the
evaluation/ folder (reports and per-request traces). Excludes secrets,
virtualenvs, bytecode, logs and run caches (code/cache/ is recreated by the
next run).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
TARGET = ROOT / "code.zip"

EXCLUDE_PARTS = {"__pycache__", ".venv", ".pytest_cache", "cache"}
EXCLUDE_NAMES = {".env"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}


def included(path: Path) -> bool:
    rel = path.relative_to(CODE)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return False
    if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.is_file()


def main() -> int:
    files = sorted(p for p in CODE.rglob("*") if included(p))
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(CODE).as_posix())
    names = {p.relative_to(CODE).as_posix() for p in files}
    assert "evaluation/usage_report.md" in names, "evaluation/usage_report.md missing"
    assert not any(n.endswith(".env") for n in names), "secret file in archive"
    print(f"wrote {TARGET} ({len(files)} files, {TARGET.stat().st_size / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

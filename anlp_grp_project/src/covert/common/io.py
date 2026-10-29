"""JSONL/JSON helpers with optional schema validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import validate


def read_jsonl(path: str | Path, schema: str | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `covert status` to see which stage produces it.")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                if schema:
                    validate(schema, rec, where=f"{path}:{i+1}")
                out.append(rec)
    return out


def write_jsonl(path: str | Path, recs: Iterable[dict], schema: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, r in enumerate(recs):
            if schema:
                validate(schema, r, where=f"{path}:{i+1}")
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str | Path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

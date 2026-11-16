"""Sổ thí nghiệm: mỗi lần thử một dòng JSONL, kèm hash dữ liệu và code để truy vết và hiệu chỉnh theo số lần thử."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_CODE_FILES = ["trade_engine.py", "search/engine_nb.py", "search/families.py", "search/indicators.py"]


def _sha(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def data_fingerprint(paths: list[Path]) -> str:
    return _sha(sorted(map(Path, paths)))


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    return _sha([root / f for f in _CODE_FILES])


class Ledger:
    def __init__(self, path: Path, data_hash: str = "", code_hash: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data_hash = data_hash
        self.code_hash = code_fingerprint() if code_hash is None else code_hash

    def append(self, record: dict) -> None:
        rec = {**record, "data_hash": self.data_hash, "code_hash": self.code_hash,
               "ts": datetime.now(timezone.utc).isoformat()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def count(self, coin: str, family: str) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n += r.get("coin") == coin and r.get("family") == family
        return n

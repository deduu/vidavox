"""
Tiny JSON-based helper so you can persist
 arbitrary Python data (token counts, BM25 state, …).

Replace with a faster / transactional solution later (e.g. SQLite or Redis).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("StateManager will read/write under %s", self.save_dir)

       # ------------------------------------------------------------------ helpers
    def _path(self, name: str, ext: str) -> Path:
        return self.save_dir / f"{name}.{ext}"

    # ------------------------------------------------------------------ public
    def save(self, name: str, data: Any, *, fmt: str = "json") -> None:
        try:
            if fmt == "json":
                with self._path(name, "json").open("w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False, indent=2)
            elif fmt == "pickle":
                with self._path(name, "pkl").open("wb") as fp:
                    pickle.dump(data, fp)
            else:
                raise ValueError("fmt must be 'json' or 'pickle'")
            logger.info("Saved state %s (%s)", name, fmt)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not save state %s - %s", name, exc)

    def load(self, name: str, *, fmt: str = "json", default: Any | None = None):
        path = self._path(name, "json" if fmt == "json" else "pkl")
        if not path.exists():
            return default
        try:
            if fmt == "json":
                with path.open("r", encoding="utf-8") as fp:
                    return json.load(fp)
            with path.open("rb") as fp:  # pickle
                return pickle.load(fp)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load state %s - %s", name, exc)
            return default

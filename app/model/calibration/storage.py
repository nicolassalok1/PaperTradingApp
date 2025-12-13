from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.utils.paths import JSON_DIR


CALIB_RESULTS_DIR = JSON_DIR / "calibration_results"


def _safe_stem(name: str) -> str:
    raw = (name or "").strip()
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw)
    raw = raw.strip("._-")
    return raw or "calibration"


def _default_name(prefix: str = "heston") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.json"


def _resolve_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" and p.exists():
        return p
    name = _safe_stem(name_or_path)
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    return CALIB_RESULTS_DIR / name


def list_results(limit: int = 200) -> List[Dict[str, Any]]:
    CALIB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in CALIB_RESULTS_DIR.glob("*.json") if p.is_file()]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for p in files[: max(0, int(limit))]:
        try:
            st = p.stat()
            out.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "size": int(st.st_size),
                }
            )
        except Exception:
            out.append({"name": p.name, "path": str(p)})
    return out


def save_result(result: Dict[str, Any], name: str | None = None, overwrite: bool = False) -> Dict[str, Any]:
    CALIB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _default_name() if not name else name
    path = _resolve_path(filename)

    if path.exists() and not overwrite:
        stem = path.stem
        suffix = path.suffix
        i = 2
        while True:
            candidate = path.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                path = candidate
                break
            i += 1

    payload = dict(result or {})
    payload.setdefault("saved_at_utc", datetime.utcnow().isoformat(timespec="seconds") + "Z")

    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        return {"success": False, "message": f"Save failed: {exc}"}

    return {"success": True, "path": str(path), "name": path.name}


def load_result(name_or_path: str) -> Dict[str, Any]:
    path = _resolve_path(name_or_path)
    if not path.exists():
        return {"success": False, "message": f"Not found: {path}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "message": f"Load failed: {exc}"}
    if not isinstance(data, dict):
        return {"success": False, "message": "Invalid JSON content (expected object)."}
    return {"success": True, "data": data, "path": str(path), "name": path.name}


__all__ = ["CALIB_RESULTS_DIR", "list_results", "save_result", "load_result"]


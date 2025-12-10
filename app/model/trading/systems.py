from typing import Dict

from app.utils.io import load_json_file, save_json_file
from app.utils.paths import JSON_DIR

EQUITIES_FILE = JSON_DIR / "trading_systems.json"
TS_EXEC_LOG_FILE = JSON_DIR / "trading_systems_execs.json"


def load_equities() -> Dict[str, dict]:
    return load_json_file(EQUITIES_FILE, {})


def save_equities(equities: Dict[str, dict]) -> None:
    save_json_file(EQUITIES_FILE, equities)


def load_ts_exec_log() -> Dict[str, dict]:
    return load_json_file(TS_EXEC_LOG_FILE, {})


def save_ts_exec_log(log: Dict[str, dict]) -> None:
    save_json_file(TS_EXEC_LOG_FILE, log)

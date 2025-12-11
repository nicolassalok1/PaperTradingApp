"""
Automatic debug loop for PaperTradingApp.
Runs scripts/test_all.py, parses failures, applies targeted fixes, and repeats.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
TARGET_TEST = REPO_ROOT / "scripts" / "test_all.py"
MAX_ITERATIONS = 100


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_tests() -> tuple[int, str, str, str]:
    proc = subprocess.run(
        [PYTHON, str(TARGET_TEST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + ("\n" if stdout and stderr else "") + stderr
    return proc.returncode, combined, stdout, stderr


def tests_passed(returncode: int, output: str) -> bool:
    return "ALL TESTS PASSED" in output or (returncode == 0 and "[FAIL]" not in output)


def split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def remove_named_arg(arg_str: str, name: str) -> str:
    parts = split_top_level_commas(arg_str)
    kept = [p for p in parts if not re.match(rf"\s*{re.escape(name)}\s*=", p)]
    return ", ".join(kept)


def drop_base_url_args(text: str) -> tuple[str, bool]:
    changed = False
    for cls in ("TradingClient", "StockHistoricalDataClient"):
        pattern = re.compile(rf"{cls}\s*\((.*?)\)", re.S)

        def repl(match: re.Match[str]) -> str:
            nonlocal changed
            args = match.group(1)
            new_args = remove_named_arg(args, "base_url")
            if new_args.strip() != args.strip():
                changed = True
            return f"{cls}({new_args})"

        text = pattern.sub(repl, text)
    return text, changed


def strip_base_url_args_in_repo() -> bool:
    changed_any = False
    for path in REPO_ROOT.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        new_text, changed = drop_base_url_args(text)
        if changed:
            path.write_text(new_text, encoding="utf-8")
            print(f"[FIX] Removed base_url from {rel_path(path)}")
            changed_any = True
    return changed_any


def rewrite_imports_in_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    original = text
    text = re.sub(
        r"^(\s*)from\s+\.*model(\.[\w\.]*)\s+import",
        r"\1from app.model\2 import",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^(\s*)import\s+model(\.[\w\.]*)?(\s+as\s+\w+)?",
        r"\1import app.model\2\3",
        text,
        flags=re.M,
    )
    if text != original:
        if original.endswith("\n") and not text.endswith("\n"):
            text += "\n"
        path.write_text(text, encoding="utf-8")
        print(f"[FIX] Rewrote model imports in {rel_path(path)}")
        return True
    return False


def extract_trace_paths(output: str) -> List[Path]:
    paths: List[Path] = []
    for match in re.finditer(r'File "([^"]+)"', output):
        try:
            paths.append(Path(match.group(1)))
        except Exception:
            continue
    return paths


def fix_import_errors(output: str) -> bool:
    if "cannot import name" not in output and "No module named" not in output:
        return False
    changed = False
    trace_paths = extract_trace_paths(output)
    targets = [p for p in trace_paths if p.suffix == ".py" and p.exists()]
    if not targets:
        targets = [p for p in REPO_ROOT.rglob("*.py") if "venv" not in p.parts]
    for path in targets:
        changed |= rewrite_imports_in_file(path)
    return changed


def strip_mvc_imports(path: Path, violation: str) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    new_lines = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        if violation in ("controller_view", "model_view"):
            if "app.vue" in stripped or (violation == "model_view" and "streamlit" in stripped):
                changed = True
                continue
        if violation == "view_model":
            if "app.model" in stripped and "controller" not in stripped:
                changed = True
                continue
        new_lines.append(line)
    if changed:
        trailing = "\n" if text.endswith("\n") else ""
        path.write_text("\n".join(new_lines) + trailing, encoding="utf-8")
        print(f"[FIX] Removed MVC-offending imports in {rel_path(path)}")
    return changed


def fix_mvc_violations(output: str) -> bool:
    matches = list(re.finditer(r"MVC violation:\s*(.+)->\s*([^\n]+)", output))
    if not matches:
        return False
    changed = False
    for m in matches:
        desc = m.group(1)
        file_str = m.group(2).strip()
        path = Path(file_str)
        if not path.is_absolute():
            path = (REPO_ROOT / file_str).resolve()
        violation = "unknown"
        if "controller imports view" in desc:
            violation = "controller_view"
        elif "model imports view" in desc:
            violation = "model_view"
        elif "view imports model" in desc:
            violation = "view_model"
        changed |= strip_mvc_imports(path, violation)
    return changed


def ensure_load_dotenv_in_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    if "load_dotenv()" in text:
        return False
    lines = text.splitlines()
    insert_at = 0
    if lines and lines[0].lstrip().startswith(('"""', "'''")):
        quote = lines[0].lstrip()[:3]
        count = lines[0].count(quote)
        i = 0
        while i + 1 < len(lines) and count < 2:
            i += 1
            count += lines[i].count(quote)
        insert_at = i + 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
    while insert_at < len(lines) and lines[insert_at].startswith("from __future__"):
        insert_at += 1
    snippet = ["from dotenv import load_dotenv", "load_dotenv()"]
    new_lines = lines[:insert_at] + snippet + [""] + lines[insert_at:]
    path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"[FIX] Added load_dotenv to {rel_path(path)}")
    return True


def ensure_env_loading(output: str) -> bool:
    if not (REPO_ROOT / ".env").exists():
        return False
    if not any(token in output for token in ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL", "Alpaca"]):
        return False
    changed = False
    changed |= ensure_load_dotenv_in_file(TARGET_TEST)
    changed |= ensure_load_dotenv_in_file(REPO_ROOT / "app" / "controller" / "__init__.py")
    return changed


def extract_function_name(msg: str) -> str | None:
    m = re.search(r"TypeError:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", msg)
    return m.group(1) if m else None


def extract_missing_args(msg: str) -> List[str]:
    m = re.search(r"missing [\d]+ required positional argument[s]?: (.+)", msg)
    if not m:
        return []
    return re.findall(r"'([^']+)'", m.group(1))


def extract_unexpected_kwarg(msg: str) -> str | None:
    m = re.search(r"unexpected keyword argument '([^']+)'", msg)
    return m.group(1) if m else None


def extract_params_block(signature: str) -> tuple[str, str]:
    start = signature.find("(")
    if start == -1:
        return "", ""
    depth = 0
    params: List[str] = []
    end = None
    for idx, ch in enumerate(signature[start:], start):
        if ch == "(":
            depth += 1
            if depth == 1:
                continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                end = idx
                break
        if depth >= 1:
            params.append(ch)
    rest = signature[end + 1 :] if end is not None else ""
    return "".join(params), rest


def patch_function_signature(path: Path, func_name: str | None, missing: List[str], unexpected: str | None, allow_extra: bool) -> bool:
    if not func_name:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(func_name)}\s*\(", line):
            start = idx
            balance = line.count("(") - line.count(")")
            end = idx
            while balance > 0 and end + 1 < len(lines):
                end += 1
                balance += lines[end].count("(") - lines[end].count(")")
            sig_lines = lines[start : end + 1]
            indent_match = re.match(r"(\s*)def", sig_lines[0])
            indent = indent_match.group(1) if indent_match else ""
            compact_sig = " ".join(s.strip() for s in sig_lines)
            params_str, rest = extract_params_block(compact_sig)
            params = split_top_level_commas(params_str)
            if unexpected:
                names = [p.split(":")[0].split("=")[0].strip().lstrip("*") for p in params]
                if unexpected not in names:
                    params.append(f"{unexpected}=None")
            if missing:
                for arg in missing:
                    updated = False
                    for i, p in enumerate(params):
                        name = p.split(":")[0].split("=")[0].strip().lstrip("*")
                        if name == arg:
                            if "=" not in p:
                                params[i] = p + "=None"
                            updated = True
                            break
                    if not updated:
                        params.append(f"{arg}=None")
            if allow_extra:
                if not any(p.strip().startswith("*args") for p in params):
                    params.append("*args")
                if not any(p.strip().startswith("**kwargs") for p in params):
                    params.append("**kwargs")
            new_params = ", ".join(params)
            suffix = rest.strip()
            if suffix and not suffix.startswith(":"):
                suffix = " " + suffix
            elif not suffix:
                suffix = ":"
            new_sig = f"{indent}def {func_name}({new_params}){suffix}"
            lines[start : end + 1] = [new_sig]
            trailing = "\n" if text.endswith("\n") else ""
            path.write_text("\n".join(lines) + trailing, encoding="utf-8")
            print(f"[FIX] Patched signature for {func_name} in {rel_path(path)}")
            return True
    return False


def fix_model_traceback(output: str) -> bool:
    if "app/model" not in output.replace("\\", "/"):
        return False
    type_line = None
    for line in reversed(output.splitlines()):
        if "TypeError" in line:
            type_line = line.strip()
            break
    if not type_line:
        return False
    func_name = extract_function_name(type_line)
    missing = extract_missing_args(type_line)
    unexpected = extract_unexpected_kwarg(type_line)
    allow_extra = "positional arguments" in type_line and "were given" in type_line
    targets = [p for p in extract_trace_paths(output) if "app/model" in str(p).replace("\\", "/")]
    if not targets:
        return False
    for path in targets:
        if patch_function_signature(path, func_name, missing, unexpected, allow_extra):
            return True
    return False


def apply_fixes(output: str) -> bool:
    applied = False
    if "TradingClient.__init__() got an unexpected keyword argument 'base_url'" in output or ("unexpected keyword argument" in output and "base_url" in output):
        applied |= strip_base_url_args_in_repo()
    if "StockHistoricalDataClient.__init__() got an unexpected keyword argument 'base_url'" in output:
        applied |= strip_base_url_args_in_repo()
    if "cannot import name" in output or "No module named" in output or "ImportError" in output:
        applied |= fix_import_errors(output)
    if "MVC violation" in output:
        applied |= fix_mvc_violations(output)
    if "APCA" in output or "Alpaca" in output:
        applied |= ensure_env_loading(output)
    if "TypeError:" in output or ("Traceback" in output and "app/model" in output.replace("\\", "/")):
        applied |= fix_model_traceback(output)
    return applied


def main() -> None:
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n[AUTO] Iteration {i} running test_all.py")
        code, combined, stdout, stderr = run_tests()
        print(combined)
        if tests_passed(code, combined):
            print("[AUTO] Tests passed; stopping.")
            return
        applied = apply_fixes(combined)
        if not applied:
            print("[AUTO] No fix applied; stopping early.")
            return
    print("[AUTO] Reached max iterations without clean run.")


if __name__ == "__main__":
    main()

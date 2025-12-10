import sys
from pathlib import Path

root = Path("app/model")

for py in root.rglob("*.py"):
    text = py.read_text(encoding="utf8", errors="ignore").lower()
    if "import streamlit" in text or "from streamlit" in text:
        print(f"[ERROR] Streamlit found in model: {py}")
        sys.exit(1)

sys.exit(0)

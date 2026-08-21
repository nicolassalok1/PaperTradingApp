"""G9 skeptic probe: is the render guard really outside the CI gate, how long does
it take, and would the proposed marker change work under --strict-markers?

Also measures service.py / tab coverage under the exact CI selection and runs
the phase-1 service test sketches to confirm they are green and stable.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(sys.argv[1])
OUT = ROOT / "scripts" / "review_iv_dashboard"


def sh(*args, timeout=900):
    t0 = time.perf_counter()
    p = subprocess.run([PY, "-m", "pytest", *args, "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, p.stdout + p.stderr, round(time.perf_counter() - t0, 1)


rc, out, _ = sh("-m", "unit or smoke", "--co", "-q")
sel = [ln for ln in out.splitlines() if "iv_dashboard" in ln]
print(f"[CI selection] iv_dashboard items under -m 'unit or smoke': {len(sel)}; render guard included: {any('render' in s for s in sel)}")
print("   ", *sel, sep="\n    ")

rc, out, sec = sh("tests/integration/test_iv_dashboard_render.py", "-q")
print(f"[render guard] rc={rc} wall={sec}s :: {out.strip().splitlines()[-1]}")

rc, out, sec = sh("tests/integration/test_app_boot.py", "-q")
print(f"[app boot (integration, same pattern)] rc={rc} wall={sec}s :: {out.strip().splitlines()[-1]}")

# proposed fix: add the smoke marker -> copy the test next to its driver under a p4_ name, run explicitly
tmp = OUT / "p4_test_render_smoke_marker_copy.py"
src = (ROOT / "tests/integration/test_iv_dashboard_render.py").read_text(encoding="utf-8")
src = src.replace("pytestmark = pytest.mark.integration",
                  "pytestmark = [pytest.mark.integration, pytest.mark.smoke]")
src = src.replace('REPO_ROOT = Path(__file__).resolve().parents[2]', f'REPO_ROOT = Path(r"{ROOT}")')
src = src.replace('DRIVER = Path(__file__).resolve().parent / "_iv_dashboard_render_driver.py"',
                  f'DRIVER = Path(r"{ROOT}") / "tests/integration/_iv_dashboard_render_driver.py"')
tmp.write_text(src, encoding="utf-8")
rc, out, sec = sh(str(tmp), "-m", "unit or smoke", "-q")
print(f"[fix: smoke+integration marker, selected by CI expr] rc={rc} wall={sec}s :: {out.strip().splitlines()[-1]}")
tmp.unlink(missing_ok=True)

# coverage of the iv_dashboard package under the CI selection
rc, out, sec = sh("-m", "unit or smoke", "--cov=app.model.iv_dashboard", "--cov=app.vue.tabs.tab_iv_dashboard",
                  "--cov-report=term-missing", "-q", "--no-header")
for ln in out.splitlines():
    if re.search(r"iv_dashboard[\\/](service|analytics)\.py|tab_iv_dashboard\.py|TOTAL", ln):
        print("[cov CI sel]", ln.strip()[:140])
m = re.search(r"(\d+ passed.*)", out)
print("[cov CI sel]", m.group(1) if m else out[-300:])

# phase-1 service sketches: green? (run twice to catch date/ordering flakiness)
for i in range(2):
    rc, out, sec = sh(str(OUT / "p1_tests_service_sketches.py"), "-q")
    print(f"[p1 service sketches run {i+1}] rc={rc} wall={sec}s :: {out.strip().splitlines()[-1]}")

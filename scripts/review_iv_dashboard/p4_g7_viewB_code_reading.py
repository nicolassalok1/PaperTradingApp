"""p4 skeptic probe (G7_viewB, lens = CODE READING) — offline, deterministic.

Extracts, from the working copy and the *installed* packages, the exact code
paths behind three candidate findings:

  1. chart-title-overprinted-by-legend
     - view: `_base_layout` (margin.t=48, legend h / y=1.02 / yanchor=bottom,
       title position left to plotly's default).
     - plotly.js bundled in plotly 5.22: title with `y='auto'` is drawn at
       `gs.t/2` (middle of the *expanded* top margin, i.e. inside the legend band).
     - Streamlit 1.51 frontend bundle: `applyStreamlitTheme` forces
       `title.xanchor='left', x=0` and wraps the title in <b>..</b>; it does not
       touch margin.t / title.y / legend position  -> same left edge as the legend.
  2. hardcoded-dark-palette-light-theme
     - Streamlit 1.51 frontend bundle: with a `[theme]` section and no
       `[theme.light]`/`[theme.dark]`, `createCustomThemes` yields ONE theme named
       "Custom Theme"; `processThemeInput` calls `addThemes(N, {keepPresetThemes:false})`
       (Light/Dark/System removed) and the settings selectbox is
       `disabled: activeTheme.name === CUSTOM_THEME_NAME`.
  3. split-annotation-units-mismatch
     - view: L451 annotation `.3f` vs L456 `tickformat='.0%'`; legacy script used
       decimal on both axis and annotation (consistent), the port changed the axis only.

Usage: python p4_g7_viewB_code_reading.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEW = REPO / "app" / "vue" / "tabs" / "tab_iv_dashboard.py"
CONFIG = REPO / ".streamlit" / "config.toml"
LEGACY = Path("C:/Users/Nathalie Asus/Downloads/option_trading_dashboard.py")

sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def show_lines(path: Path, lo: int, hi: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i in range(lo, hi + 1):
        print(f"{i:4d}: {lines[i - 1]}")


def grep_ctx(text: str, pattern: str, before: int, after: int, limit: int = 1) -> list[str]:
    out = []
    for m in re.finditer(pattern, text):
        out.append(text[max(0, m.start() - before): m.end() + after])
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
section("1a. view: _base_layout + title calls")
show_lines(VIEW, 97, 105)
src = VIEW.read_text(encoding="utf-8")
for m in re.finditer(r"update_layout\(\s*title=.*", src):
    print("   title call @", src[: m.start()].count("\n") + 1, "->", m.group(0).strip()[:90])
names = re.findall(r'name=(f?"[^"]+")', src)
print("   legend entries (series chart max): 5 unconditional + 2 conditional (iv_history, current_iv)")
print("   all trace names:", names)

# --------------------------------------------------------------------------- #
section("1b. plotly.js (plotly package_data) — title y='auto' => gs.t/2")
import plotly  # noqa: E402

pjs = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
js = pjs.read_text(encoding="utf-8", errors="replace")
print("plotly", plotly.__version__, "bundle", pjs.name, f"{pjs.stat().st_size/1e6:.1f} MB")
for snip in grep_ctx(js, r'"auto"===[a-zA-Z]\.y\?[a-zA-Z]\.t/2', 260, 120):
    print("  ...", snip, "...")

# --------------------------------------------------------------------------- #
section("1c. Streamlit frontend bundle — applyStreamlitTheme layout overrides")
import streamlit  # noqa: E402

static_js = Path(streamlit.__file__).parent / "static" / "static" / "js"
print("streamlit", streamlit.__version__)
theme_bundle = None
for f in static_js.glob("*.js"):
    t = f.read_text(encoding="utf-8", errors="replace")
    if "PlotlyChart:CustomTheme" in t:
        theme_bundle = (f, t)
        break
assert theme_bundle, "plotly CustomTheme bundle not found"
f, t = theme_bundle
i = t.find("PlotlyChart:CustomTheme")
body = t[i: i + 2600]
m = re.search(r"title:\{[^}]*\{[^}]*\}[^}]*\{[^}]*\}[^}]*\}", body)
print(f"  bundle: {f.name}")
print("  title override :", m.group(0) if m else "??")
print("  -> title.xanchor/x forced:", re.findall(r'xanchor:"left",x:0', m.group(0)) if m else "??",
      "| title.y / margin.t touched:", bool(re.search(r"title:\{[^}]*\by:", body)), "/", bool(re.search(r"margin:\{[^}]*\bt:", body)))
m = re.search(r"legend:\{.*?\}\}", body)
print("  legend override:", m.group(0)[:200] if m else "??")
m = re.search(r"margin:\{[^}]*\}", body)
print("  margin override:", m.group(0) if m else "??")
m = re.search(r'function [A-Za-z]{2}\(an,Wi\)\{try\{[A-Za-z]{2}\(an\.layout\.template\.layout,Wi\)\}.*?<b>.*?\}\)\)\}', t)
print("  applied to     :", m.group(0)[:260] if m else "??")

# --------------------------------------------------------------------------- #
section("2. Streamlit frontend bundle — theme selector with a custom [theme]")
print("config.toml :", CONFIG.read_text(encoding="utf-8").splitlines()[:3])
cfg = CONFIG.read_text(encoding="utf-8")
print("  [theme.light] present:", "[theme.light]" in cfg, "| [theme.dark] present:", "[theme.dark]" in cfg)
app_bundle = None
for f in static_js.glob("*.js"):
    t = f.read_text(encoding="utf-8", errors="replace")
    if 'CUSTOM_THEME_NAME="Custom Theme"' in t:
        app_bundle = (f, t)
        break
assert app_bundle, "app bundle not found"
f, t = app_bundle
print(f"  bundle: {f.name}")
for label, pat, b, a in (
    ("createCustomThemes", r"createCustomThemes=t=>\{", 0, 520),
    ("processThemeInput", r"processThemeInput\(g\)\{", 0, 560),
    ("settings selectbox", r'"Choose app theme"', 0, 200),
):
    for snip in grep_ctx(t, pat, b, a):
        print(f"  [{label}]\n    {snip}")

# --------------------------------------------------------------------------- #
section("3. units: annotation / axis / title / log  vs legacy")
show_lines(VIEW, 445, 457)
show_lines(VIEW, 383, 392)
show_lines(VIEW, 480, 491)
if LEGACY.exists():
    leg = LEGACY.read_text(encoding="utf-8", errors="replace")
    print("  legacy PercentFormatter / tickformat used:", bool(re.search(r"PercentFormatter|tickformat", leg)))
    for pat in (r"Regime Split.*", r"set_title\(f'Forward.*", r"Intersection with y=x.*"):
        for s in re.findall(pat, leg):
            print("  legacy:", s.strip()[:110])
else:
    print("  legacy script not found at", LEGACY)

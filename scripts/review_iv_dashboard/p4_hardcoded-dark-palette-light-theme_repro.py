"""
p4_hardcoded-dark-palette-light-theme_repro.py — skeptic repro (offline, deterministic).

1. Re-measure WCAG contrast of the palette actually defined in the view (parsed from
   source, not copied) on #0E1117 (config.toml) and #FFFFFF.
2. Check the finding's premise "the Streamlit settings menu still lets the user pick
   Light": inspect the Streamlit 1.51 frontend bundle shipped in the venv for the
   theme-selector logic when a [theme] section is configured.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEW = REPO / "app" / "vue" / "tabs" / "tab_iv_dashboard.py"
CONFIG = REPO / ".streamlit" / "config.toml"


def _lum(hexcol: str) -> float:
    h = hexcol.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    l1, l2 = _lum(fg), _lum(bg)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


src = VIEW.read_text(encoding="utf-8")
palette = dict(re.findall(r'^(_COL_[A-Z]+)\s*=\s*"(#[0-9a-fA-F]{6})"', src, flags=re.M))
cfg = CONFIG.read_text(encoding="utf-8")
bg_cfg = re.search(r'backgroundColor\s*=\s*"(#[0-9a-fA-F]{6})"', cfg).group(1)
base_cfg = re.search(r'base\s*=\s*"(\w+)"', cfg).group(1)
has_light_section = "[theme.light]" in cfg
has_dark_section = "[theme.dark]" in cfg

print(f"config.toml: base={base_cfg} backgroundColor={bg_cfg} "
      f"[theme.light]={has_light_section} [theme.dark]={has_dark_section}")
print("\n1. WCAG contrast of the view palette (parsed from source)")
print(f"   {'name':13s} {'hex':8s} on {bg_cfg}   on #FFFFFF")
for name, col in palette.items():
    print(f"   {name:13s} {col}  {contrast(col, bg_cfg):5.2f}       {contrast(col, '#FFFFFF'):5.2f}")

# ---------------------------------------------------------------- 2. Streamlit frontend
import streamlit  # noqa: E402

static_js = Path(streamlit.__file__).parent / "static" / "static" / "js"
bundle = next(p for p in static_js.glob("index.*.js") if "Custom Theme" in p.read_text(encoding="utf-8"))
js = bundle.read_text(encoding="utf-8")
print(f"\n2. streamlit {streamlit.__version__} frontend bundle: {bundle.name}")


def snippet(anchor: str, before: int = 0, after: int = 400) -> str:
    i = js.find(anchor)
    assert i >= 0, anchor
    return js[max(0, i - before): i + after]


# (a) what themes get created from a [theme] section
s = snippet("createCustomThemes=t=>", after=520)
print("   createCustomThemes:", s[:520])
# (b) how they are installed (are presets kept?)
s = snippet("processThemeInput(g)", after=420)
print("\n   processThemeInput:", s)
# (c) the Settings dialog selectbox
s = snippet('children:"Choose app theme"', before=0, after=160)
print("\n   SettingsDialog:", s)

keep_presets_false = "addThemes(N,{keepPresetThemes:!1})" in js
disabled_when_custom = "disabled:ve.name===CUSTOM_THEME_NAME" in js
single_custom = (not has_light_section) and (not has_dark_section)
print("\nRESULT:")
print(f"   presets dropped when a config theme exists (keepPresetThemes:false): {keep_presets_false}")
print(f"   selectbox disabled when active theme is 'Custom Theme':            {disabled_when_custom}")
print(f"   this repo's config yields a single 'Custom Theme' (no light/dark sections): {single_custom}")
if keep_presets_false and disabled_when_custom and single_custom:
    print("   => In Streamlit 1.51 with this config.toml the Settings menu offers ONLY 'Custom Theme'"
          " (disabled selectbox). 'Light' is NOT selectable from the UI: the finding's premise is false.")

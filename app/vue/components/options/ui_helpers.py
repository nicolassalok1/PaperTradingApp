"""
UI helpers for the Options panels (Streamlit widgets, dashboard actions, visuals).
"""

from __future__ import annotations

import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.controller.options_ui_controller import (
    get_cached_iv_for as ui_get_cached_iv_for,
    get_json_dir,
)


def _choose_option_select(key_prefix: str, option_char: str | None = None):
    choice = st.radio(
        "Type d'option",
        ["Call", "Put"],
        index=0 if (option_char or "c").lower() == "c" else 1,
        key=f"{key_prefix}_choice",
        horizontal=True,
    )
    return choice, "c" if choice == "Call" else "p"


def _render_option_text(title: str, key: str | None = None) -> None:
    st.markdown(f"### {title}")


def render_method_explainer(title: str, body: str) -> None:
    st.markdown(f"**{title}**")
    st.write(body)


def _get_cached_iv_for(*args, **kwargs):
    try:
        return ui_get_cached_iv_for(*args, **kwargs)
    except Exception:
        return None


def _render_heatmaps_for_current_option(
    label: str,
    call_matrix: np.ndarray,
    put_matrix: np.ndarray,
    x_values,
    y_values,
    option_char: str,
):
    """Minimal heatmap renderer using matplotlib (no app.vue dependency)."""
    try:
        is_call = str(option_char or "c").lower().startswith("c")
        matrix = call_matrix if is_call else put_matrix
        fig, ax = plt.subplots()
        img = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            cmap="viridis",
        )
        ax.set_xlabel("Spot")
        ax.set_ylabel("Strike")
        ax.set_title(label)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.info("Heatmap non disponible.")


def render_add_to_dashboard_button(
    product_label: str,
    option_char: str,
    price_value: float,
    strike: float,
    maturity: float,
    key_prefix: str,
    spot: float | None = None,
    premium_paid: float | None = None,
    qty: float | None = None,
    misc: dict | None = None,
):
    """Append a priced structure to the dashboard JSON cache."""
    key = f"{key_prefix}_{product_label}_{option_char}_{strike}_{maturity}"
    if st.button(f"Ajouter {product_label} ({'Call' if option_char == 'c' else 'Put'})", key=key):
        dashboard_entry = {
            "product": product_label,
            "type": "Call" if option_char == "c" else "Put",
            "price": float(price_value),
            "strike": float(strike),
            "maturity": float(maturity),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "spot": float(spot) if spot is not None else None,
            "premium_paid": float(premium_paid) if premium_paid is not None else None,
            "qty": float(qty) if qty is not None else None,
            "misc": misc or {},
        }
        path = get_json_dir() / "dashboard_entries.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            entries = json.loads(path.read_text()) if path.exists() else []
        except Exception:
            entries = []
        entries.append(dashboard_entry)
        path.write_text(json.dumps(entries, indent=2))
        st.success(f"{product_label} ajoute au dashboard !")


def _common_defaults():
    """Fetch common option parameters with safe fallbacks."""
    try:
        spot = float(st.session_state.get("common_spot_value", 100.0))
        maturity = float(st.session_state.get("common_maturity_value", 1.0))
        rate = float(st.session_state.get("common_rate_value", 0.01))
        sigma = float(st.session_state.get("common_sigma_value", 0.2))
        div = float(st.session_state.get("d_common", 0.0))
    except Exception:
        spot, maturity, rate, sigma, div = 100.0, 1.0, 0.01, 0.2, 0.0
    return spot, maturity, rate, sigma, div


# Expose globals used by the panels
common_spot_value, common_maturity_value, common_rate_value, common_sigma_value, d_common = (
    _common_defaults()
)
option_char = st.session_state.get("option_char", "c") if hasattr(st, "session_state") else "c"


def _k(s: str) -> str:
    """Generate a Streamlit key prefix for options panels."""
    return f"opt_{s}"


__all__ = [
    "_choose_option_select",
    "_render_option_text",
    "render_method_explainer",
    "_get_cached_iv_for",
    "_render_heatmaps_for_current_option",
    "render_add_to_dashboard_button",
    "common_spot_value",
    "common_maturity_value",
    "common_rate_value",
    "common_sigma_value",
    "d_common",
    "option_char",
    "_k",
    "datetime",
]

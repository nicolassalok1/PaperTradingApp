"""
Shared UI helpers for options and dashboard components.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def k_key(base: str, option_label: str) -> str:
    """Generate a Streamlit key combining base + option label."""
    return f"{base}_{option_label.lower()}"


def render_heatmap(
    matrix, x_values, y_values, title, xlabel="Spot", ylabel="Strike", wrap_in_expander=True
):
    """Draw a 2D heatmap using matplotlib."""

    def _draw():
        fig, ax = plt.subplots()
        img = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            cmap="viridis",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)

    if wrap_in_expander:
        with st.expander(title, expanded=False):
            _draw()
    else:
        _draw()


def render_heatmaps_for_current_option(
    label, call_matrix, put_matrix, x_values, y_values, option_char
):
    """Render either call or put heatmap based on option_char."""
    is_call = str(option_char).lower() == "c"
    matrix = call_matrix if is_call else put_matrix
    title = f"{'Call' if is_call else 'Put'} - {label}"
    render_heatmap(matrix, x_values, y_values, title, wrap_in_expander=True)


def choose_option_select(key_prefix: str, option_char: str | None = None) -> tuple[str, str]:
    """
    Small wrapper to choose Call/Put with a stable Streamlit key prefix.
    Returns (label, char) where char is 'c' or 'p'.
    """
    default_idx = 0 if (option_char or "c").lower() == "c" else 1
    choice = st.radio(
        "Type d'option",
        ["Call", "Put"],
        index=default_idx,
        key=f"{key_prefix}_choice",
        horizontal=True,
    )
    return choice, "c" if choice == "Call" else "p"


def render_option_text(title: str, key: str | None = None) -> None:
    """Minimal text renderer; kept as a stub."""
    st.markdown(f"### {title}")


def render_unlock_sidebar_button(context_key: str, label: str) -> None:
    """Simple button to unlock sidebar interactions; kept for backward compatibility."""
    if st.button(label, key=f"unlock_sidebar_{context_key}"):
        st.session_state["options_tab_locked"] = False


def render_method_explainer(title: str, body: str) -> None:
    st.markdown(f"**{title}**")
    st.write(body)


def render_general_definition_explainer(title: str, body: str) -> None:
    st.markdown(f"**{title}**")
    st.write(body)


def render_section_explainer(title: str, body: str) -> None:
    st.markdown(f"**{title}**")
    st.write(body)


def render_inputs_explainer(title: str, body: str) -> None:
    st.markdown(f"**{title}**")
    st.write(body)


def plot_crr_tree(spots, values=None, title: str = "CRR Tree"):
    """
    Draw a CRR tree for UI educational display.
    Accepts numpy arrays (spots, values) shaped (n_steps+1, n_steps+1).
    """
    spots_arr = np.array(spots)
    values_arr = np.array(values) if values is not None else None
    n_levels = spots_arr.shape[0]
    fig_width = min(12, 4 + n_levels * 0.25)
    fig_height = min(10, 3 + n_levels * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    def node_coords(level: int, index: int) -> tuple[float, float]:
        x = index - level / 2
        y = n_levels - 1 - level
        return x, y

    for level in range(n_levels - 1):
        for index in range(level + 1):
            x_curr, y_curr = node_coords(level, index)
            x_down, y_down = node_coords(level + 1, index)
            x_up, y_up = node_coords(level + 1, index + 1)
            ax.plot([x_curr, x_down], [y_curr, y_down], color="lightgray", linewidth=0.8)
            ax.plot([x_curr, x_up], [y_curr, y_up], color="lightgray", linewidth=0.8)

    x_coords = []
    y_coords = []
    color_values = []
    spots_list = []
    option_list = []

    for level in range(n_levels):
        for index in range(level + 1):
            value = spots_arr[level, index]
            option_value = values_arr[level, index] if values_arr is not None else np.nan
            if np.isnan(value):
                continue
            x, y = node_coords(level, index)
            x_coords.append(x)
            y_coords.append(y)
            color_values.append(option_value if not np.isnan(option_value) else 0.0)
            spots_list.append(value)
            option_list.append(option_value)

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=color_values,
        cmap="viridis",
        s=120,
        edgecolors="black",
        linewidths=0.5,
    )
    display_labels = n_levels - 1 <= 10
    if display_labels:
        for x, y, spot_value, option_value in zip(x_coords, y_coords, spots_list, option_list):
            ax.text(x, y + 0.25, f"S={spot_value:.2f}", ha="center", va="bottom", fontsize=7)
            if not np.isnan(option_value):
                ax.text(x, y - 0.25, f"V={option_value:.2f}", ha="center", va="top", fontsize=7)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Option value")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


__all__ = [
    "k_key",
    "render_heatmap",
    "render_heatmaps_for_current_option",
    "choose_option_select",
    "render_option_text",
    "render_unlock_sidebar_button",
    "render_method_explainer",
    "render_general_definition_explainer",
    "render_section_explainer",
    "render_inputs_explainer",
    "plot_crr_tree",
]

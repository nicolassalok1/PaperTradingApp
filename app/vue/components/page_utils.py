from __future__ import annotations

import textwrap

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str | None = None,
    *,
    icon: str = "✨",
    badge: str | None = None,
) -> None:
    """Render a stylized page title card to keep headers consistent."""
    html = textwrap.dedent(
        f"""
        <div class="page-hero">
          <div class="page-hero__icon">{icon}</div>
          <div class="page-hero__titles">
            <div class="page-hero__title">{title}</div>
            {f'<div class="page-hero__subtitle">{subtitle}</div>' if subtitle else ''}
          </div>
          {f'<div class="page-hero__badge">{badge}</div>' if badge else ''}
        </div>
        """
    ).strip()

    if hasattr(st, "html"):
        st.html(html)  # type: ignore[attr-defined]
    else:  # pragma: no cover
        st.markdown(html, unsafe_allow_html=True)


__all__ = ["render_page_header"]


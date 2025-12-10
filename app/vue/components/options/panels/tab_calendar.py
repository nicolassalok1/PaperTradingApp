import streamlit as st
from app.vue.components.options.panels.calendar.tab_calendar import render_tab_calendar
from app.vue.components.options.panels.calendar.tab_diagonal import render_tab_diagonal
from app.vue.components.options.controller_bridge import *


def render_panel_calendar():
    st.subheader("Options Calendriers & Diagonals")

    calendar_labels = [
        "Calendar Spread",
        "Diagonal Spread",
    ]
    calendar_tabs = st.tabs(calendar_labels)

    for label, tab in zip(calendar_labels, calendar_tabs):
        with tab:
            if label == "Calendar Spread":
                render_tab_calendar()
            elif label == "Diagonal Spread":
                render_tab_diagonal()
            else:
                st.error("Type inconnu.")

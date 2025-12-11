from app.vue.components.options.vanilla.bermuda import render
from app.vue.components.options.controller_bridge import *


def render_tab_bermudan():
    """Render the active Bermudan (LSMC) option panel."""
    ctx = get_option_context()
    if not ensure_close_history(ctx):
        return
    render()

from app.vue.components.options.vanilla.american import render
from app.vue.components.options.controller_bridge import *


def render_tab_american():
    """Render the active American (CRR) option panel."""
    ctx = get_option_context()
    if not ensure_close_history(ctx):
        return
    render()

from app.vue.components.options.vanilla.mc_unified import render_mc_panel
from app.vue.components.options.controller_bridge import *


def render_tab_american():
    """Render the American option panel using unified MC engine."""
    ctx = get_option_context()
    if not ensure_close_history(ctx):
        return
    render_mc_panel(default_style="american")

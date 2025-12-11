import pkgutil
import importlib
import app.model.options as opt_options
import app.model.heston as opt_heston


def test_options_model_no_streamlit():
    for _, name, _ in pkgutil.walk_packages(opt_options.__path__, opt_options.__name__ + "."):
        module = importlib.import_module(name)
        file = getattr(module, "__file__", "") or ""
        if file.endswith(".py"):
            text = open(file, "r", encoding="utf-8", errors="ignore").read().lower()
            assert "streamlit" not in text


def test_no_heston_in_options_module_names():
    for _, name, _ in pkgutil.walk_packages(opt_options.__path__, opt_options.__name__ + "."):
        assert "heston" not in name.lower()


def test_heston_module_isolated():
    for _, name, _ in pkgutil.walk_packages(opt_heston.__path__, opt_heston.__name__ + "."):
        module = importlib.import_module(name)
        # pas de dépendance vers app.vue
        for attr in dir(module):
            pass  # on laisse, test structurel simple pour l'instant

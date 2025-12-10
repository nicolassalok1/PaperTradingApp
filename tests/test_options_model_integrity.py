import pkgutil
import importlib
import app.model.options as opt


def test_model_options_no_streamlit():
    for _, name, _ in pkgutil.walk_packages(opt.__path__, opt.__name__ + "."):
        module = importlib.import_module(name)
        source = module.__dict__.get("__file__", "")
        if source and source.endswith(".py"):
            text = open(source, "r", encoding="utf8").read()
            assert "streamlit" not in text.lower()

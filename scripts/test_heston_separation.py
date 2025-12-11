import app.model.heston
import app.model.options


def test_no_heston_in_options():
    import pkgutil

    for _, name, _ in pkgutil.walk_packages(
        app.model.options.__path__, app.model.options.__name__ + "."
    ):
        module = __import__(name, fromlist=["dummy"])
        assert "heston" not in name.lower()

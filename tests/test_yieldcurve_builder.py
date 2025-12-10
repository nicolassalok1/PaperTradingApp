from app.model.yieldcurve.builder import build_curve  # adapte


def test_yieldcurve_monotone_maturity():
    raw_data = [
        (0.5, 0.01),
        (1.0, 0.012),
        (2.0, 0.015),
    ]
    curve = build_curve(raw_data)
    r1 = curve(0.5)
    r2 = curve(2.0)
    assert r2 >= r1

from app.model.portfolio.valuation import compute_portfolio_value  # adapte au vrai nom


def test_portfolio_value_zero_positions():
    positions = []  # ou structure vide selon ta signature
    result = compute_portfolio_value(positions, prices={})
    assert result == 0 or abs(result) < 1e-8

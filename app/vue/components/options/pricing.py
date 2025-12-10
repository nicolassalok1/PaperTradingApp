from app.controller import options_controller as oc

black_scholes_price = oc.black_scholes_price
price_asset_or_nothing = oc.price_asset_or_nothing
price_basket_call = oc.price_basket_call
price_basket_put = oc.price_basket_put
price_butterfly = oc.price_butterfly
price_chooser = oc.price_chooser
price_call_spread = oc.price_call_spread
price_calendar_spread = oc.price_calendar_spread
price_condor = oc.price_condor
price_diagonal_spread = oc.price_diagonal_spread
price_digital = oc.price_digital
price_forward_start = oc.price_forward_start
price_iron_butterfly = oc.price_iron_butterfly
price_iron_condor = oc.price_iron_condor
price_put_spread = oc.price_put_spread
price_quanto = oc.price_quanto
price_rainbow = oc.price_rainbow
price_straddle = oc.price_straddle
price_strangle = oc.price_strangle
price_american_crr = oc.price_american_crr
price_bermuda_crr = oc.price_bermuda_crr
price_heston_european_call = oc.price_heston_european_call
price_asian_geo_mc = oc.price_asian_geo_mc
price_asian_mc = oc.price_asian_mc
price_barrier_digital = oc.price_barrier_digital
price_barrier_vanilla = oc.price_barrier_vanilla
price_cliquet = oc.price_cliquet
price_lookback_fixed_mc = oc.price_lookback_fixed_mc
price_lookback_mc = oc.price_lookback_mc

__all__ = [
    "black_scholes_price",
    "price_asset_or_nothing",
    "price_basket_call",
    "price_basket_put",
    "price_butterfly",
    "price_chooser",
    "price_call_spread",
    "price_calendar_spread",
    "price_condor",
    "price_diagonal_spread",
    "price_digital",
    "price_forward_start",
    "price_iron_butterfly",
    "price_iron_condor",
    "price_put_spread",
    "price_quanto",
    "price_rainbow",
    "price_straddle",
    "price_strangle",
    "price_american_crr",
    "price_bermuda_crr",
    "price_heston_european_call",
    "price_asian_geo_mc",
    "price_asian_mc",
    "price_barrier_digital",
    "price_barrier_vanilla",
    "price_cliquet",
    "price_lookback_fixed_mc",
    "price_lookback_mc",
]

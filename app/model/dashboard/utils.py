from typing import Dict, Iterable, Set


def collect_dashboard_tickers(
    *, portfolio: Dict[str, dict], forwards: Dict[str, dict], custom_options: Dict[str, dict]
) -> Set[str]:
    tickers: Set[str] = set()
    for sym in portfolio.keys():
        if sym:
            tickers.add(sym.upper())
    for fwd in forwards.values():
        sym = fwd.get("symbol")
        if sym:
            tickers.add(str(sym).upper())
    for opt in custom_options.values():
        und = opt.get("underlying") or opt.get("symbol")
        if und:
            tickers.add(str(und).upper())
    return tickers

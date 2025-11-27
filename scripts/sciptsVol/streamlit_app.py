import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

def _configure_page():
    """Set page config when running this module directly."""
    try:
        st.set_page_config(page_title="Volatility Tools", layout="wide")
    except Exception:
        # Streamlit raises if page config is set elsewhere; ignore in embedded use.
        pass


# ---------------------- Black-Scholes helpers ----------------------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return -norm.cdf(-d1)


def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calculate_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def calculate_theta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return theta


# ---------------------- Data helpers (Yahoo Finance) ----------------------
def fetch_history(symbol: str, start: datetime = None, end: datetime = None, period: str | None = "2y"):
    ticker = yf.Ticker(symbol)
    if period:
        hist = ticker.history(period=period, interval="1d")
    else:
        hist = ticker.history(start=start, end=end, interval="1d")
    hist = hist.sort_index()
    # Drop timezone info so we can compare with naive datetime objects without errors
    if hasattr(hist.index, "tz") and hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)
    return hist


def compute_iv_proxy(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.copy()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["implied_vol"] = df["log_returns"].rolling(window=window, min_periods=max(1, min(window, len(df)))).std() * np.sqrt(252)
    if df["implied_vol"].isna().all():
        iv_const = df["log_returns"].std() * np.sqrt(252)
        df["implied_vol"] = iv_const
    df["implied_vol"] = df["implied_vol"].fillna(method="ffill").fillna(method="bfill")
    return df


# ---------------------- Tab: Volatility Crush (manual) ----------------------
def render_vol_crush_tab(tab):
    with tab:
        st.title("📊 Volatility Crush Trade Analyzer")
        col1, col2 = st.columns(2)

        with col1:
            st.header("📈 Market Data & Parameters")
            ticker = st.text_input("Ticker Symbol", value="NVDA")
            spot_price = st.number_input("Spot Price ($)", min_value=0.01, value=100.00, step=0.01)
            strike_price = st.number_input("Strike Price ($)", min_value=0.01, value=100.00, step=0.01)
            iv_percent = st.number_input("Implied Volatility (%)", min_value=0.1, value=50.0, step=0.1)
            days_to_expiry = st.number_input("Days to Expiry", min_value=1, value=30, step=1)
            risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100

            st.markdown("---")
            if st.button("💰 Price Straddle", type="primary"):
                T = days_to_expiry / 365.0
                iv = iv_percent / 100.0
                call_price = black_scholes_call(spot_price, strike_price, T, risk_free_rate, iv)
                put_price = black_scholes_put(spot_price, strike_price, T, risk_free_rate, iv)
                straddle_price = call_price + put_price

                delta_call = calculate_delta(spot_price, strike_price, T, risk_free_rate, iv, "call")
                delta_put = calculate_delta(spot_price, strike_price, T, risk_free_rate, iv, "put")
                delta_straddle = delta_call + delta_put
                gamma = calculate_gamma(spot_price, strike_price, T, risk_free_rate, iv)
                vega = calculate_vega(spot_price, strike_price, T, risk_free_rate, iv) * 2
                theta = calculate_theta(spot_price, strike_price, T, risk_free_rate, iv, "call") + calculate_theta(
                    spot_price, strike_price, T, risk_free_rate, iv, "put"
                )

                st.session_state["current_data"] = {
                    "call_price": call_price,
                    "put_price": put_price,
                    "straddle_price": straddle_price,
                    "delta": delta_straddle,
                    "gamma": gamma,
                    "vega": vega,
                    "theta": theta,
                    "spot_price": spot_price,
                    "strike_price": strike_price,
                    "T": T,
                    "iv": iv,
                    "risk_free_rate": risk_free_rate,
                }

        if "current_data" in st.session_state:
            data = st.session_state["current_data"]
            with col1:
                st.subheader("📊 Current Straddle Price")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Call Price", f"${data['call_price']:.2f}")
                with col_b:
                    st.metric("Put Price", f"${data['put_price']:.2f}")
                st.metric("**Straddle Price**", f"${data['straddle_price']:.2f}")

                st.subheader("🔤 Current Greeks")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.metric("Delta", f"{data['delta']:.4f}")
                    st.metric("Vega", f"{data['vega']:.2f}")
                with col_d:
                    st.metric("Gamma", f"{data['gamma']:.4f}")
                    st.metric("Theta", f"{data['theta']:.2f}")

        with col2:
            st.header("🔮 Scenario Analysis")
            new_spot_price = st.number_input(
                "New Spot Price ($)",
                min_value=0.01,
                value=data["spot_price"] if "current_data" in st.session_state else 100.00,
                step=0.01,
                key="new_spot",
            )
            new_iv_percent = st.number_input(
                "New Implied Volatility (%)",
                min_value=0.1,
                value=(data["iv"] * 100) if "current_data" in st.session_state else 50.0,
                step=0.1,
                key="new_iv",
            )
            st.markdown("---")
            if st.button("🎯 Analyze Scenario", type="primary") and "current_data" in st.session_state:
                data = st.session_state["current_data"]
                new_iv = new_iv_percent / 100.0
                new_call_price = black_scholes_call(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv)
                new_put_price = black_scholes_put(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv)
                new_straddle_price = new_call_price + new_put_price

                pnl_long = new_straddle_price - data["straddle_price"]
                pnl_short = -pnl_long

                new_delta_call = calculate_delta(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv, "call")
                new_delta_put = calculate_delta(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv, "put")
                new_delta = new_delta_call + new_delta_put
                new_gamma = calculate_gamma(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv)
                new_vega = calculate_vega(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv) * 2
                new_theta = calculate_theta(new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv, "call") + calculate_theta(
                    new_spot_price, data["strike_price"], data["T"], data["risk_free_rate"], new_iv, "put"
                )

                st.session_state["scenario_data"] = {
                    "new_straddle_price": new_straddle_price,
                    "pnl_long": pnl_long,
                    "pnl_short": pnl_short,
                    "new_delta": new_delta,
                    "new_gamma": new_gamma,
                    "new_vega": new_vega,
                    "new_theta": new_theta,
                }

        if "scenario_data" in st.session_state:
            scenario = st.session_state["scenario_data"]
            with col2:
                st.subheader("💵 P/L Analysis")
                st.metric("New Straddle Price", f"${scenario['new_straddle_price']:.2f}")
                col_e, col_f = st.columns(2)
                with col_e:
                    st.metric("Long Straddle P/L", f"${scenario['pnl_long']:+.2f}", delta=f"{scenario['pnl_long']:+.2f}", delta_color="normal")
                with col_f:
                    st.metric("Short Straddle P/L", f"${scenario['pnl_short']:+.2f}", delta=f"{scenario['pnl_short']:+.2f}", delta_color="inverse")

                st.subheader("🔤 New Scenario Greeks")
                col_g, col_h = st.columns(2)
                with col_g:
                    st.metric("Delta", f"{scenario['new_delta']:.4f}")
                    st.metric("Vega", f"{scenario['new_vega']:.2f}")
                with col_h:
                    st.metric("Gamma", f"{scenario['new_gamma']:.4f}")
                    st.metric("Theta", f"{scenario['new_theta']:.2f}")

        st.markdown("---")
        st.info("ℹ️ Manual pricing using Black-Scholes. Enter parameters then analyze scenarios.")


# ---------------------- Tab: IV Dashboard ----------------------
def render_iv_dashboard_tab(tab):
    if "volatility_data" not in st.session_state:
        st.session_state.volatility_data = None
        st.session_state.current_implied_vol = None

    with tab:
        st.title("📈 Implied Volatility Trading Dashboard")
        col_inputs = st.columns(3)
        with col_inputs[0]:
            symbol = st.text_input("Symbol", value="SPY")
        with col_inputs[1]:
            years = st.number_input("History (years)", min_value=1, max_value=10, value=2, step=1)
        with col_inputs[2]:
            window = st.number_input("Rolling window (days)", min_value=10, max_value=120, value=30, step=5)

        if st.button("Fetch Data", type="primary"):
            try:
                hist = fetch_history(symbol, period=f"{years}y")
                if hist.empty:
                    st.error("No data returned.")
                else:
                    df = compute_iv_proxy(hist, window=int(window))
                    st.session_state.volatility_data = df
                    st.session_state.current_implied_vol = df["implied_vol"].iloc[-1] if not df.empty else None
                    st.success(f"Loaded {len(df)} points for {symbol}")
            except Exception as exc:
                st.error(f"Error fetching data: {exc}")

        df = st.session_state.volatility_data
        if df is not None and not df.empty:
            st.header("Volatility Analysis")
            if st.button("Analyze Implied Volatility"):
                forward_period = 30
                analysis_df = pd.DataFrame(
                    {
                        "current_vol": df["implied_vol"][:-forward_period],
                        "forward_30d_vol": df["implied_vol"].shift(-forward_period)[:-forward_period],
                    }
                ).dropna()
                analysis_df["vol_diff"] = analysis_df["forward_30d_vol"] - analysis_df["current_vol"]

                slope1, intercept1, r_value1, p_value1, std_error1 = stats.linregress(
                    analysis_df["current_vol"], analysis_df["forward_30d_vol"]
                )
                slope2, intercept2, r_value2, p_value2, std_error2 = stats.linregress(
                    analysis_df["current_vol"], analysis_df["vol_diff"]
                )

                intersection_x = intercept1 / (1 - slope1) if slope1 != 1 else analysis_df["current_vol"].median()
                high_vol_regime = analysis_df["current_vol"] > intersection_x
                low_vol_regime = analysis_df["current_vol"] <= intersection_x

                slope_high = intercept_high = r_high = p_high = None
                slope_low = intercept_low = r_low = p_low = None
                if high_vol_regime.sum() > 10:
                    slope_high, intercept_high, r_high, p_high, _ = stats.linregress(
                        analysis_df.loc[high_vol_regime, "current_vol"], analysis_df.loc[high_vol_regime, "vol_diff"]
                    )
                if low_vol_regime.sum() > 10:
                    slope_low, intercept_low, r_low, p_low, _ = stats.linregress(
                        analysis_df.loc[low_vol_regime, "current_vol"], analysis_df.loc[low_vol_regime, "vol_diff"]
                    )

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
                ax1.scatter(analysis_df["current_vol"], analysis_df["forward_30d_vol"], alpha=0.6, s=20)
                x_range = np.linspace(analysis_df["current_vol"].min(), analysis_df["current_vol"].max(), 100)
                y_pred1 = slope1 * x_range + intercept1
                ax1.plot(x_range, y_pred1, "r-", linewidth=2, label=f"Regression R² = {r_value1**2:.3f}")
                min_val = min(analysis_df["current_vol"].min(), analysis_df["forward_30d_vol"].min())
                max_val = max(analysis_df["current_vol"].max(), analysis_df["forward_30d_vol"].max())
                ax1.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.7, label="y=x (No Change)")
                ax1.set_xlabel("Current Implied Volatility (proxy)")
                ax1.set_ylabel("30-Day Forward Average Vol")
                ax1.set_title(f"Forward Vol vs Current Vol\ny = {slope1:.3f}x + {intercept1:.3f}, R² = {r_value1**2:.3f}")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.scatter(
                    analysis_df.loc[high_vol_regime, "current_vol"],
                    analysis_df.loc[high_vol_regime, "vol_diff"],
                    alpha=0.6,
                    s=20,
                    color="red",
                    label="High Vol Regime",
                )
                ax2.scatter(
                    analysis_df.loc[low_vol_regime, "current_vol"],
                    analysis_df.loc[low_vol_regime, "vol_diff"],
                    alpha=0.6,
                    s=20,
                    color="blue",
                    label="Low Vol Regime",
                )

                if slope_high is not None:
                    x_high = analysis_df.loc[high_vol_regime, "current_vol"]
                    x_range_high = np.linspace(x_high.min(), x_high.max(), 100)
                    y_pred_high = slope_high * x_range_high + intercept_high
                    ax2.plot(x_range_high, y_pred_high, "r-", linewidth=2, label=f"High Vol R² = {r_high**2:.3f}")

                if slope_low is not None:
                    x_low = analysis_df.loc[low_vol_regime, "current_vol"]
                    x_range_low = np.linspace(x_low.min(), x_low.max(), 100)
                    y_pred_low = slope_low * x_range_low + intercept_low
                    ax2.plot(x_range_low, y_pred_low, "b-", linewidth=2, label=f"Low Vol R² = {r_low**2:.3f}")

                ax2.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.7, label="No Change (y=0)")
                ax2.axvline(x=intersection_x, color="g", linestyle=":", linewidth=1, alpha=0.7, label=f"Regime Split (Vol={intersection_x:.3f})")
                ax2.set_xlabel("Current Implied Volatility (proxy)")
                ax2.set_ylabel("Vol Difference (Forward - Current)")
                ax2.set_title("Vol Difference vs Current Vol (Regime Analysis)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                ax3.plot(df.index, df["implied_vol"], label="Implied Volatility (proxy)", linewidth=1)
                vol_75th = df["implied_vol"].quantile(0.75)
                vol_25th = df["implied_vol"].quantile(0.25)
                ax3.axhline(y=vol_75th, color="red", linestyle="--", alpha=0.7, label="75th Percentile")
                ax3.axhline(y=vol_25th, color="green", linestyle="--", alpha=0.7, label="25th Percentile")
                ax3.axhline(y=df["implied_vol"].mean(), color="black", linestyle="-", alpha=0.7, label="Mean")
                if st.session_state.current_implied_vol is not None:
                    ax3.scatter(df.index[-1], st.session_state.current_implied_vol, color="red", s=100, zorder=5, label="Current")
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Implied Volatility (proxy)")
                ax3.set_title("Implied Volatility Time Series with Regime Bands")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis="x", rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Statistical Analysis")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Regression 1: Forward Vol on Current Vol**")
                    st.write(f"Slope: {slope1:.4f}")
                    st.write(f"Intercept: {intercept1:.4f}")
                    st.write(f"R²: {r_value1**2:.4f}")
                    st.write(f"P-value: {p_value1:.4f}")
                    st.write(f"Intersection with y=x: {intersection_x:.4f}")
                with col_b:
                    st.write("**Regression 2: Vol Difference on Current Vol**")
                    st.write(f"Slope: {slope2:.4f}")
                    st.write(f"Intercept: {intercept2:.4f}")
                    st.write(f"R²: {r_value2**2:.4f}")
                    st.write(f"P-value: {p_value2:.4f}")

                st.subheader("Trading Insights")
                if slope1 < 1:
                    st.info("📉 Forward volatility tends to mean-revert (slope < 1)")
                else:
                    st.info("📈 Forward volatility tends to trend (slope > 1)")
                if slope2 < 0:
                    st.info("🔄 High current volatility predicts lower future volatility (mean reversion)")
                else:
                    st.info("⚡ High current volatility predicts higher future volatility (momentum)")

            st.subheader("Current Data Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Implied Vol (proxy)", f"{st.session_state.current_implied_vol:.4f}" if st.session_state.current_implied_vol else "N/A")
            with c2:
                st.metric("Data Points", len(df))
            with c3:
                st.metric("Symbol", symbol)
            with st.expander("View Raw Data"):
                st.dataframe(df.tail(50))


# ---------------------- Tab: Earnings IV Crush ----------------------
def render_earnings_dashboard_tab(tab):
    if "earnings_analysis_results" not in st.session_state:
        st.session_state.earnings_analysis_results = None
        st.session_state.earnings_stock_data = None
        st.session_state.earnings_vix_data = None

    with tab:
        st.title("📊 Earnings Trading Dashboard - IV Crush")
        col_inputs = st.columns(4)
        with col_inputs[0]:
            ticker = st.text_input("Ticker", value="NVDA")
        with col_inputs[1]:
            earnings_date = st.date_input("Earnings Date", value=datetime.today().date())
        with col_inputs[2]:
            days_to_expiry = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365, step=1)
        with col_inputs[3]:
            window_days = st.number_input("Window (+/- days)", value=20, min_value=5, max_value=60, step=5)

        if st.button("Analyze IV Crush", type="primary"):
            try:
                start = datetime.combine(earnings_date, datetime.min.time()) - timedelta(days=window_days)
                end = datetime.combine(earnings_date, datetime.min.time()) + timedelta(days=window_days)
                stock_hist = fetch_history(ticker, start=start, end=end, period=None)
                if stock_hist.empty:
                    st.error("No stock data returned.")
                    return

                stock_hist = compute_iv_proxy(stock_hist, window=30)
                st.session_state.earnings_stock_data = stock_hist

                try:
                    vix_hist = fetch_history("^VIX", start=start, end=end, period=None)
                    st.session_state.earnings_vix_data = vix_hist
                except Exception:
                    st.session_state.earnings_vix_data = None

                dates = stock_hist.index
                earnings_dt = datetime.combine(earnings_date, datetime.min.time())
                pre_date = dates[dates <= earnings_dt].max() if len(dates[dates <= earnings_dt]) > 0 else dates.min()
                post_date = dates[dates > earnings_dt].min() if len(dates[dates > earnings_dt]) > 0 else dates.max()

                pre_stock_price = stock_hist.loc[pre_date, "Close"]
                post_open = stock_hist.loc[post_date, "Open"]
                post_close = stock_hist.loc[post_date, "Close"]
                post_stock_price = (post_open + post_close) / 2

                pre_iv = stock_hist.loc[pre_date, "implied_vol"] if "implied_vol" in stock_hist.columns else np.nan
                post_iv = stock_hist.loc[post_date, "implied_vol"] if "implied_vol" in stock_hist.columns else np.nan

                if (np.isnan(pre_iv) or np.isnan(post_iv)) and st.session_state.earnings_vix_data is not None:
                    vix_dates = st.session_state.earnings_vix_data.index
                    pre_mask = vix_dates[vix_dates <= earnings_dt]
                    post_mask = vix_dates[vix_dates > earnings_dt]
                    pre_vix = st.session_state.earnings_vix_data.loc[pre_mask.max(), "Close"] if len(pre_mask) else st.session_state.earnings_vix_data["Close"].iloc[0]
                    post_vix = st.session_state.earnings_vix_data.loc[post_mask.min(), "Close"] if len(post_mask) else st.session_state.earnings_vix_data["Close"].iloc[-1]
                    if np.isnan(pre_iv):
                        pre_iv = pre_vix / 100.0 * 1.5
                    if np.isnan(post_iv):
                        post_iv = post_vix / 100.0 * 1.2

                if np.isnan(pre_iv):
                    pre_iv = 0.40
                if np.isnan(post_iv):
                    post_iv = 0.25

                iv_crush_pct = (pre_iv - post_iv) / pre_iv * 100 if pre_iv else 0

                T = days_to_expiry / 365.0
                r = 0.05
                K = pre_stock_price

                pre_call = black_scholes_call(pre_stock_price, K, T, r, pre_iv)
                pre_put = black_scholes_put(pre_stock_price, K, T, r, pre_iv)
                post_call = black_scholes_call(post_stock_price, K, T, r, post_iv)
                post_put = black_scholes_put(post_stock_price, K, T, r, post_iv)

                pre_straddle = pre_call + pre_put
                post_straddle = post_call + post_put
                straddle_change = post_straddle - pre_straddle
                straddle_change_pct = straddle_change / pre_straddle * 100 if pre_straddle else 0

                pre_delta = calculate_delta(pre_stock_price, K, T, r, pre_iv, "call") + calculate_delta(pre_stock_price, K, T, r, pre_iv, "put")
                post_delta = calculate_delta(post_stock_price, K, T, r, post_iv, "call") + calculate_delta(post_stock_price, K, T, r, post_iv, "put")

                pre_vega = calculate_vega(pre_stock_price, K, T, r, pre_iv) * 2
                post_vega = calculate_vega(post_stock_price, K, T, r, post_iv) * 2

                st.session_state.earnings_analysis_results = {
                    "ticker": ticker,
                    "earnings_date": earnings_dt,
                    "pre_date": pre_date,
                    "post_date": post_date,
                    "pre_stock_price": pre_stock_price,
                    "post_stock_price": post_stock_price,
                    "pre_iv": pre_iv,
                    "post_iv": post_iv,
                    "iv_crush_pct": iv_crush_pct,
                    "pre_call": pre_call,
                    "post_call": post_call,
                    "pre_put": pre_put,
                    "post_put": post_put,
                    "pre_straddle": pre_straddle,
                    "post_straddle": post_straddle,
                    "straddle_change": straddle_change,
                    "straddle_change_pct": straddle_change_pct,
                    "pre_delta": pre_delta,
                    "post_delta": post_delta,
                    "pre_vega": pre_vega,
                    "post_vega": post_vega,
                }
                st.success("Analysis complete!")
            except Exception as exc:
                st.error(f"Error: {exc}")

        results = st.session_state.earnings_analysis_results
        if results:
            st.header(f"📈 Analysis Results for {results['ticker']}")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Stock Price (Pre)", f"${results['pre_stock_price']:.2f}")
            with col_b:
                st.metric(
                    "Stock Price (Post)",
                    f"${results['post_stock_price']:.2f}",
                    f"{((results['post_stock_price'] - results['pre_stock_price']) / results['pre_stock_price'] * 100):+.2f}%",
                )
            with col_c:
                st.metric("Pre-Earnings IV (proxy)", f"{results['pre_iv']:.1%}")
            with col_d:
                st.metric("IV Crush", f"-{results['iv_crush_pct']:.1f}%", delta_color="inverse")

            st.divider()
            st.subheader("ATM Options Pricing & Straddle")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Pre-Earnings**")
                st.write(f"Call: ${results['pre_call']:.2f}")
                st.write(f"Put: ${results['pre_put']:.2f}")
                st.write(f"**Straddle: ${results['pre_straddle']:.2f}**")
            with c2:
                st.write("**Post-Earnings**")
                st.write(f"Call: ${results['post_call']:.2f}")
                st.write(f"Put: ${results['post_put']:.2f}")
                st.write(f"**Straddle: ${results['post_straddle']:.2f}**")
            with c3:
                call_change = results["post_call"] - results["pre_call"]
                put_change = results["post_put"] - results["pre_put"]
                st.write("**Change**")
                st.write(f"Call: ${call_change:+.2f}")
                st.write(f"Put: ${put_change:+.2f}")
                st.write(f"**Straddle: ${results['straddle_change']:+.2f} ({results['straddle_change_pct']:+.1f}%)**")

            st.divider()
            st.subheader("P/L Analysis")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("LONG Straddle P/L", f"${results['straddle_change']:+.2f}", f"{results['straddle_change_pct']:+.1f}%")
            with c2:
                st.metric("SHORT Straddle P/L", f"${-results['straddle_change']:+.2f}", f"{-results['straddle_change_pct']:+.1f}%")

            st.divider()
            st.subheader("Greeks Analysis")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Pre-Earnings Delta", f"{results['pre_delta']:.3f}")
            with c2:
                st.metric("Post-Earnings Delta", f"{results['post_delta']:.3f}", f"{results['post_delta'] - results['pre_delta']:+.3f}")
            with c3:
                st.metric("Pre-Earnings Vega", f"{results['pre_vega']:.2f}")
            with c4:
                st.metric("Post-Earnings Vega", f"{results['post_vega']:.2f}", f"{results['post_vega'] - results['pre_vega']:+.2f}")

            st.divider()
            st.subheader("IV Crush Visualization")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            if st.session_state.earnings_stock_data is not None:
                earnings_dt = results["earnings_date"]
                window_stock = st.session_state.earnings_stock_data
                ax1.plot(window_stock.index, window_stock["Close"], "b-", linewidth=2, label="Stock Price")
                ax1.axvline(x=earnings_dt, color="red", linestyle="--", alpha=0.7, label="Earnings Date")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Stock Price ($)", color="blue")
                ax1.tick_params(axis="y", labelcolor="blue")
                ax1.set_title(f"{results['ticker']} Stock Price Around Earnings")
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc="upper left")

                if "implied_vol" in window_stock.columns:
                    ax1_twin = ax1.twinx()
                    ax1_twin.plot(window_stock.index, window_stock["implied_vol"] * 100, "g-", linewidth=2, label="IV (proxy)")
                    ax1_twin.set_ylabel("Implied Volatility (%)", color="green")
                    ax1_twin.tick_params(axis="y", labelcolor="green")
                    ax1_twin.legend(loc="upper right")

                ax1.tick_params(axis="x", rotation=45)

            option_types = ["Call", "Put", "Straddle"]
            pre_prices = [results["pre_call"], results["pre_put"], results["pre_straddle"]]
            post_prices = [results["post_call"], results["post_put"], results["post_straddle"]]
            x = np.arange(len(option_types))
            width = 0.35
            bars1 = ax2.bar(x - width / 2, pre_prices, width, label="Pre-Earnings (High IV)", color="lightblue", alpha=0.8)
            bars2 = ax2.bar(x + width / 2, post_prices, width, label="Post-Earnings (Low IV)", color="lightcoral", alpha=0.8)
            for bar1, bar2 in zip(bars1, bars2):
                ax2.text(bar1.get_x() + bar1.get_width() / 2.0, bar1.get_height() + 0.5, f"${bar1.get_height():.1f}", ha="center", va="bottom", fontsize=9)
                ax2.text(bar2.get_x() + bar2.get_width() / 2.0, bar2.get_height() + 0.5, f"${bar2.get_height():.1f}", ha="center", va="bottom", fontsize=9)
            ax2.set_xlabel("Option Strategy")
            ax2.set_ylabel("Option Price ($)")
            ax2.set_title("ATM Options & Straddle: IV Crush Impact")
            ax2.set_xticks(x)
            ax2.set_xticklabels(option_types)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

        else:
            st.info("Fetch data and run an analysis to see results.")


if __name__ == "__main__":
    _configure_page()
    tab_vol_crush, tab_iv_dashboard, tab_earnings = st.tabs(
        ["Volatility Crush Analyzer", "IV Dashboard", "Earnings IV Crush"]
    )
    render_vol_crush_tab(tab_vol_crush)
    render_iv_dashboard_tab(tab_iv_dashboard)
    render_earnings_dashboard_tab(tab_earnings)

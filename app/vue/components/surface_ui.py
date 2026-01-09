"""
Reusable UI helpers for calibration tabs.

Kept Streamlit-first (no business logic): preview, diagnostics, filtering, and
helpers to move IV grids across tabs (e.g., Calibration -> Options).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.model.calibration.implied_vol import bs_call_price


_SURF_K_ALIASES = {"k", "strike", "strike_price", "strikeprice"}
_SURF_T_ALIASES = {"t", "ttm", "tau", "time_to_maturity", "maturity"}
_SURF_S0_ALIASES = {"s0", "spot", "underlying", "underlyingprice"}
_SURF_IV_ALIASES = {"iv", "implied_vol", "impliedvol", "implied_volatility", "impliedvolatility", "sigma"}
_SURF_TYPE_ALIASES = {"type", "option_type", "cp", "right"}


def plot_iv_heatmap(z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str) -> None:
    z_arr = np.asarray(z, dtype=float)
    title_lower = str(title or "").lower()
    is_error = "erreur" in title_lower or "error" in title_lower

    colorscale = "Viridis"
    heatmap_kwargs = {}
    if is_error:
        finite = np.abs(z_arr[np.isfinite(z_arr)])
        vmax = float(finite.max()) if finite.size else 1.0
        colorscale = [
            [0.0, "#f7d4d4"],  # negative large -> light red
            [0.5, "#0b0b0b"],  # zero -> dark center
            [1.0, "#d2e8ff"],  # positive large -> light blue
        ]
        heatmap_kwargs.update({"zmin": -vmax, "zmax": vmax, "zmid": 0.0})

    fig = go.Figure(
        data=go.Heatmap(
            z=z_arr,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title="IV" if not is_error else "Erreur IV"),
            **heatmap_kwargs,
        )
    )
    fig.update_layout(title=title, xaxis_title="Moneyness", yaxis_title="Time to maturity (y)")
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True, "scrollZoom": False})


def price_grid_from_iv_grid(
    *,
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_grid: np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """
    Convert an IV grid on (t_grid, m_grid) to call prices via Black-Scholes.
    """
    S0_f = float(S0)
    r_f = float(r)
    q_f = float(q)
    iv = np.asarray(iv_grid, dtype=float)
    m = np.asarray(m_grid, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    out = np.full((len(t), len(m)), np.nan, dtype=float)
    for i_t, tau in enumerate(t):
        t_val = float(tau)
        for j_m, mm in enumerate(m):
            vol = float(iv[i_t, j_m]) if i_t < iv.shape[0] and j_m < iv.shape[1] else np.nan
            if not np.isfinite(vol) or vol <= 0 or t_val <= 0 or S0_f <= 0:
                continue
            K = float(mm * S0_f)
            out[i_t, j_m] = bs_call_price(S0_f, K, t_val, r_f, q_f, vol)
    return out


def render_price_surface_grid(
    *,
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    price_grid: np.ndarray,
    title: str,
    key: str,
    colorscale: str = "Viridis",
) -> None:
    """
    3D surface for prices on a fixed (t_grid, m_grid) grid.
    Axes: Strike K (x), TTM (years) (y), Price (z).
    """
    m = np.asarray(m_grid, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    px = np.asarray(price_grid, dtype=float)
    if px.shape != (len(t), len(m)):
        st.info("Grille de prix invalide pour l'affichage 3D.")
        return

    K = m * float(S0)
    if len(K) == 0 or len(t) == 0:
        st.info("Grille vide.")
        return

    fig = go.Figure(
        data=[
            go.Surface(
                x=K,
                y=t,
                z=px,
                colorscale=colorscale,
                colorbar=dict(title="Prix"),
                showscale=True,
                opacity=0.9,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="TTM (années)",
            zaxis_title="Prix",
            xaxis=dict(range=[float(K.min()), float(K.max())]),
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True}, key=key)


def grid_to_surface_df(
    *,
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_grid: np.ndarray,
    opt_type: str = "call",
) -> pd.DataFrame:
    m = np.asarray(m_grid, dtype=float).reshape(1, -1)
    t = np.asarray(t_grid, dtype=float).reshape(-1, 1)
    iv = np.asarray(iv_grid, dtype=float)
    if iv.ndim != 2 or iv.shape != (t.shape[0], m.shape[1]):
        return pd.DataFrame(columns=["K", "T", "S0", "iv", "type"])

    rows = []
    for i_t in range(iv.shape[0]):
        for j_m in range(iv.shape[1]):
            v = float(iv[i_t, j_m])
            if not np.isfinite(v) or v <= 0:
                continue
            T = float(t_grid[i_t])
            K = float(float(S0) * float(m_grid[j_m]))
            rows.append({"K": K, "T": T, "S0": float(S0), "iv": v, "type": str(opt_type)})

    return pd.DataFrame(rows, columns=["K", "T", "S0", "iv", "type"])


def discover_cached_surfaces(cache_dir: Path) -> list[Path]:
    def _is_surface_csv(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                header = (f.readline() or "").strip().lower()
        except Exception:
            return False
        if not header:
            return False
        cols = {c.strip() for c in header.split(",")}
        return {"k", "t", "s0", "iv"}.issubset(cols)

    if not cache_dir.exists():
        return []
    search_dirs = [
        cache_dir,
        cache_dir / "YahooOptionChains",
    ]
    files: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        files.extend([p for p in d.glob("*.csv") if p.is_file() and _is_surface_csv(p)])
    return sorted(files, key=lambda p: p.name.lower())


def median_s0(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "S0" not in df.columns:
        return None
    s0_vals = pd.to_numeric(df["S0"], errors="coerce")
    s0_pos = s0_vals[s0_vals > 0]
    if s0_pos.empty:
        return None
    try:
        return float(s0_pos.median())
    except Exception:
        return None


def _find_surface_col(df: pd.DataFrame, aliases: set[str]):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in cols:
            return cols[alias]
    return None


def canonicalize_surface_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["K", "T", "S0", "iv", "type"])

    k_col = _find_surface_col(df, _SURF_K_ALIASES)
    t_col = _find_surface_col(df, _SURF_T_ALIASES)
    s0_col = _find_surface_col(df, _SURF_S0_ALIASES)
    iv_col = _find_surface_col(df, _SURF_IV_ALIASES)
    typ_col = _find_surface_col(df, _SURF_TYPE_ALIASES)

    if not (k_col and t_col and s0_col and iv_col):
        return pd.DataFrame(columns=["K", "T", "S0", "iv", "type"])

    out = pd.DataFrame(
        {
            "K": pd.to_numeric(df[k_col], errors="coerce"),
            "T": pd.to_numeric(df[t_col], errors="coerce"),
            "S0": pd.to_numeric(df[s0_col], errors="coerce"),
            "iv": pd.to_numeric(df[iv_col], errors="coerce"),
        }
    )
    if typ_col is not None:
        typ = df[typ_col].astype(str).str.lower().str.strip()
        typ = typ.replace({"c": "call", "p": "put", "calll": "call"})
        out["type"] = typ
    else:
        out["type"] = "call"

    out = out.dropna(subset=["K", "T", "S0", "iv"])
    out = out[(out["K"] > 0) & (out["T"] > 0) & (out["S0"] > 0) & (out["iv"] > 0)]
    out = out.reset_index(drop=True)
    return out[["K", "T", "S0", "iv", "type"]]


def render_surface_filters(df_canon: pd.DataFrame, *, key_prefix: str) -> pd.DataFrame:
    if df_canon is None or df_canon.empty:
        st.info("Aucune surface pour filtrage.")
        return df_canon

    dfw = df_canon.copy()
    dfw["moneyness"] = dfw["K"] / dfw["S0"]
    dfw = dfw.dropna(subset=["moneyness"])
    if dfw.empty:
        st.warning("Impossible de calculer la moneyness (K/S0).")
        return df_canon

    m_min = float(dfw["moneyness"].min())
    m_max = float(dfw["moneyness"].max())
    t_min = float(dfw["T"].min())
    t_max = float(dfw["T"].max())
    iv_min = float(dfw["iv"].min())
    iv_max = float(dfw["iv"].max())

    if m_min == m_max:
        m_max = m_min + 1e-6
    if t_min == t_max:
        t_max = t_min + 1e-6
    if iv_min == iv_max:
        iv_max = iv_min + 1e-6

    st.caption(f"Avant filtre: {len(dfw):,} lignes")

    moneyness_key = f"{key_prefix}_filter_moneyness"
    ttm_key = f"{key_prefix}_filter_ttm"
    iv_key = f"{key_prefix}_filter_iv"

    calls_only_key = f"{key_prefix}_filter_calls_only"
    quantile_on_key = f"{key_prefix}_filter_iv_quantile_on"
    quantiles_key = f"{key_prefix}_filter_iv_quantiles"

    def _clamp_range(key: str, lo: float, hi: float) -> None:
        prev = st.session_state.get(key)
        if isinstance(prev, (list, tuple)) and len(prev) == 2:
            try:
                a = float(prev[0])
                b = float(prev[1])
            except Exception:
                return
            a = max(lo, min(hi, a))
            b = max(lo, min(hi, b))
            if a > b:
                a, b = b, a
            st.session_state[key] = (float(a), float(b))

    _clamp_range(moneyness_key, m_min, m_max)
    _clamp_range(ttm_key, t_min, t_max)
    _clamp_range(iv_key, iv_min, iv_max)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        calls_only = st.checkbox("CALL only", value=True, key=calls_only_key)
        max_rows = int(
            st.number_input(
                "Max rows (sample, 0=off)",
                min_value=0,
                max_value=200000,
                value=0,
                step=1000,
            )
        )
        quantile_on = st.checkbox("Filtrer IV par quantiles", value=False, key=quantile_on_key)
    with col_b:
        m_rng = st.slider(
            "Moneyness (K/S0)",
            min_value=float(m_min),
            max_value=float(m_max),
            value=st.session_state.get(moneyness_key, (float(m_min), float(m_max))),
            step=0.01,
            key=moneyness_key,
        )
        t_rng = st.slider(
            "TTM T (années)",
            min_value=float(t_min),
            max_value=float(t_max),
            value=st.session_state.get(ttm_key, (float(t_min), float(t_max))),
            step=0.01,
            key=ttm_key,
        )
        iv_rng = st.slider(
            "IV",
            min_value=float(iv_min),
            max_value=float(iv_max),
            value=st.session_state.get(iv_key, (float(iv_min), float(iv_max))),
            step=0.01,
            key=iv_key,
        )

    q_low, q_high = 0.01, 0.99
    if quantile_on:
        q_low, q_high = st.slider(
            "Quantiles IV (low/high)",
            min_value=0.0,
            max_value=1.0,
            value=(0.01, 0.99),
            step=0.01,
            key=quantiles_key,
        )

    df_f = dfw.copy()
    if calls_only and "type" in df_f.columns:
        df_f = df_f[df_f["type"].astype(str).str.lower().str.startswith("c")]
    df_f = df_f[
        (df_f["moneyness"] >= float(m_rng[0]))
        & (df_f["moneyness"] <= float(m_rng[1]))
        & (df_f["T"] >= float(t_rng[0]))
        & (df_f["T"] <= float(t_rng[1]))
        & (df_f["iv"] >= float(iv_rng[0]))
        & (df_f["iv"] <= float(iv_rng[1]))
    ]

    if quantile_on and not df_f.empty:
        try:
            lo = float(df_f["iv"].quantile(float(q_low)))
            hi = float(df_f["iv"].quantile(float(q_high)))
            df_f = df_f[(df_f["iv"] >= lo) & (df_f["iv"] <= hi)]
        except Exception:
            pass

    if max_rows > 0 and len(df_f) > max_rows:
        df_f = df_f.sample(max_rows, random_state=42).reset_index(drop=True)

    st.caption(f"Après filtre: {len(df_f):,} lignes")
    st.dataframe(df_f.head(30), hide_index=True, width="stretch")
    try:
        st.download_button(
            "Télécharger la surface filtrée (CSV)",
            data=df_f[["K", "T", "S0", "iv", "type"]].to_csv(index=False).encode("utf-8"),
            file_name="surface_filtered.csv",
            mime="text/csv",
            width="stretch",
        )
    except Exception:
        pass

    return df_f[["K", "T", "S0", "iv", "type"]]


def surface_diagnostics(df: pd.DataFrame, *, scatter_key: str) -> None:
    if df is None or df.empty:
        st.info("Aucune donnée à diagnostiquer.")
        return

    st.caption(f"{len(df):,} lignes | colonnes: {', '.join([str(c) for c in df.columns[:20]])}")

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    required = ["k", "t", "s0", "iv"]
    missing = [c.upper() for c in required if c not in cols_lower]
    if missing:
        st.warning(f"Colonnes manquantes pour calibration: {', '.join(missing)}")
        return

    k_col = cols_lower["k"]
    t_col = cols_lower["t"]
    s0_col = cols_lower["s0"]
    iv_col = cols_lower["iv"]
    typ_col = cols_lower.get("type")

    k = pd.to_numeric(df[k_col], errors="coerce")
    t = pd.to_numeric(df[t_col], errors="coerce")
    s0 = pd.to_numeric(df[s0_col], errors="coerce")
    iv = pd.to_numeric(df[iv_col], errors="coerce")

    dfw = pd.DataFrame({"K": k, "T": t, "S0": s0, "iv": iv})
    dfw = dfw.dropna(subset=["K", "T", "S0", "iv"])
    dfw = dfw[(dfw["K"] > 0) & (dfw["T"] > 0) & (dfw["S0"] > 0) & (dfw["iv"] > 0)]
    if dfw.empty:
        st.warning("Toutes les lignes ont été filtrées (valeurs non valides).")
        return

    dfw["moneyness"] = dfw["K"] / dfw["S0"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("K", f"[{dfw['K'].min():.2f} ; {dfw['K'].max():.2f}]")
    col_b.metric("T (années)", f"[{dfw['T'].min():.4f} ; {dfw['T'].max():.4f}]")
    col_c.metric("IV", f"[{dfw['iv'].min():.4f} ; {dfw['iv'].max():.4f}]")
    col_d.metric("Moneyness", f"[{dfw['moneyness'].min():.3f} ; {dfw['moneyness'].max():.3f}]")

    if typ_col is not None:
        try:
            counts = (
                df[typ_col]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace({"c": "call", "p": "put"})
                .value_counts()
            )
            if not counts.empty:
                st.caption("Répartition `type`: " + ", ".join([f"{k}={int(v)}" for k, v in counts.items()]))
        except Exception:
            pass

    show_scatter = st.checkbox(
        "Afficher le nuage de points (moneyness x T, couleur=IV)",
        value=False,
        key=scatter_key,
    )
    if show_scatter:
        try:
            dfp = dfw.copy()
            if len(dfp) > 4000:
                dfp = dfp.sample(4000, random_state=42)
            fig = go.Figure(
                data=go.Scattergl(
                    x=dfp["moneyness"],
                    y=dfp["T"],
                    mode="markers",
                    marker=dict(
                        color=dfp["iv"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="IV"),
                        size=5,
                        opacity=0.7,
                    ),
                    text=[
                        f"K={k:.2f}, T={tt:.3f}, IV={vv:.3f}"
                        for k, tt, vv in zip(dfp["K"], dfp["T"], dfp["iv"])
                    ],
                    hoverinfo="text",
                )
            )
            fig.update_layout(xaxis_title="Moneyness (K/S0)", yaxis_title="T (years)", height=360)
            st.plotly_chart(fig, width="stretch", config={"staticPlot": True, "scrollZoom": False})
        except Exception as exc:
            st.warning(f"Plot impossible: {exc}")


def render_market_surface_3d(
    df_canon: pd.DataFrame,
    *,
    key: str,
) -> None:
    """
    Render a 3D market IV surface (K, TTM, IV) directly from the dataset, with light smoothing.
    """
    if df_canon is None or df_canon.empty:
        st.info("Charge d'abord une surface pour afficher la nappe 3D.")
        return

    dfw = df_canon.copy()
    dfw = dfw.dropna(subset=["K", "T", "iv"])
    dfw = dfw[(dfw["K"] > 0) & (dfw["T"] > 0) & (dfw["iv"] > 0)]
    if dfw.empty:
        st.info("Aucune donnée exploitable pour tracer la nappe 3D.")
        return

    k_min = float(dfw["K"].min())
    k_max = float(dfw["K"].max())
    t_min = float(dfw["T"].min())
    t_max = float(dfw["T"].max())
    if k_max <= k_min:
        k_max = k_min + 1e-6
    if t_max <= t_min:
        t_max = t_min + 1e-6

    # Build a coarse grid (median IV per bin) to draw a smooth surface and overlay raw points.
    grid = None
    try:
        k_edges = np.linspace(k_min, k_max, 26)
        t_edges = np.linspace(t_min, t_max, 26)
        dfw["k_bin"] = pd.cut(dfw["K"], bins=k_edges, labels=False, include_lowest=True)
        dfw["t_bin"] = pd.cut(dfw["T"], bins=t_edges, labels=False, include_lowest=True)
        dfw = dfw.dropna(subset=["k_bin", "t_bin"])
        if not dfw.empty:
            dfw["k_center"] = dfw["k_bin"].astype(int).map(
                lambda idx: float(0.5 * (k_edges[idx] + k_edges[idx + 1]))
            )
            dfw["t_center"] = dfw["t_bin"].astype(int).map(
                lambda idx: float(0.5 * (t_edges[idx] + t_edges[idx + 1]))
            )
            grid = (
                dfw.groupby(["t_center", "k_center"])["iv"]
                .median()
                .reset_index()
                .pivot(index="t_center", columns="k_center", values="iv")
                .sort_index()
                .sort_index(axis=1)
            )
    except Exception:
        grid = None

    def _smooth(z: np.ndarray, passes: int = 2) -> np.ndarray:
        arr = np.array(z, dtype=float)
        for _ in range(max(1, passes)):
            padded = np.pad(arr, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
            windows = sliding_window_view(padded, (3, 3))
            arr = np.nanmean(windows, axis=(2, 3))
        return arr

    fig = go.Figure()
    if grid is not None and grid.shape[0] >= 2 and grid.shape[1] >= 2:
        z_raw = grid.to_numpy()
        z_smooth = _smooth(z_raw, passes=2)
        if not np.isfinite(z_smooth).any():
            z_smooth = z_raw
        fig.add_trace(
            go.Surface(
                x=np.array(grid.columns, dtype=float),
                y=np.array(grid.index, dtype=float),
                z=z_smooth,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                showscale=True,
                opacity=0.9,
                name="Surface",
            )
        )
    else:
        st.info("Pas assez de données pour construire une nappe lissée.")
        return

    fig.update_layout(
        title="Nappe IV marché (3D)",
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="TTM (années)",
            zaxis_title="IV",
            xaxis=dict(range=[k_min, k_max]),
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True}, key=key)


def render_surface_preview_dropdown(
    df: pd.DataFrame,
    *,
    label: str,
    key: str,
    diag_key_prefix: str,
) -> None:
    if df is None or df.empty:
        st.info("Surface vide.")
        return

    preview_n = 30

    def _render() -> None:
        st.dataframe(df.head(preview_n), hide_index=True, width="stretch")
        if len(df) > preview_n:
            st.caption(f"Aperçu: {preview_n} premières lignes affichées (sur {len(df):,}).")
        st.divider()
        surface_diagnostics(df, scatter_key=f"{diag_key_prefix}_diag_show_scatter")

    title = f"{label} ({len(df):,} lignes)"

    if hasattr(st, "popover"):
        try:
            with st.popover(title):
                _render()
            return
        except Exception:
            pass

    try:
        with st.expander(title, expanded=False):
            _render()
        return
    except Exception:
        pass

    if st.checkbox(title, value=False, key=key):
        _render()


__all__ = [
    "canonicalize_surface_df",
    "discover_cached_surfaces",
    "grid_to_surface_df",
    "median_s0",
    "plot_iv_heatmap",
    "price_grid_from_iv_grid",
    "render_price_surface_grid",
    "render_surface_filters",
    "render_surface_preview_dropdown",
    "render_market_surface_3d",
    "surface_diagnostics",
]

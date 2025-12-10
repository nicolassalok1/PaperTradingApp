"""Heavy options pricing utilities extracted from the ancienne `run_app_options` block.

These helpers are computation-only (no UI coupling) and are shared across
the Options UI components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats
from scipy.linalg import lu_factor, lu_solve
from scipy.stats import norm


class CrankNicolsonBS:
    """
    Solveur Crank-Nicolson pour la PDE de Black-Scholes en log(S).

    Typeflag:
        'Eu'  : option européenne
        'Am'  : option américaine (exercice possible à chaque date de grille)
        'Bmd' : option bermudéenne (exercice possible à certaines dates)
    cpflag:
        'c' : call
        'p' : put
    """

    def __init__(
        self,
        Typeflag: str,
        cpflag: str,
        S0: float,
        K: float,
        T: float,
        vol: float,
        r: float,
        d: float,
        n_spatial: int = 500,
        n_time: int = 600,
        exercise_step: int | None = None,
        n_exercise_dates: int | None = None,
        **_,
    ) -> None:
        self.Typeflag = Typeflag
        self.cpflag = cpflag
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.vol = float(vol)
        self.r = float(r)
        self.d = float(d)

        self.n_spatial = max(50, int(n_spatial))
        self.n_time = max(50, int(n_time))

        # Deux modes possibles pour la Bermudane :
        # - exercise_step       : exercice tous les 'exercise_step' pas
        # - n_exercise_dates    : nb de dates d'exercice (incluant T)
        # Si les deux sont donnés -> erreur, c'est ambigu.
        if exercise_step is not None and n_exercise_dates is not None:
            raise ValueError("Spécifie soit exercise_step, soit n_exercise_dates, pas les deux.")

        self.exercise_step = int(exercise_step) if exercise_step is not None else None
        self.n_exercise_dates = int(n_exercise_dates) if n_exercise_dates is not None else None

    # -------------------- utils --------------------

    def _resolve_params(
        self,
        Typeflag: str | None,
        cpflag: str | None,
        S0: float | None,
        K: float | None,
        T: float | None,
        vol: float | None,
        r: float | None,
        d: float | None,
    ):
        """Résout les paramètres effectifs sans casser les valeurs 0 éventuelles."""

        Typeflag = self.Typeflag if Typeflag is None else Typeflag
        cpflag = self.cpflag if cpflag is None else cpflag
        S0 = self.S0 if S0 is None else float(S0)
        K = self.K if K is None else float(K)
        T = self.T if T is None else float(T)
        vol = self.vol if vol is None else float(vol)
        r = self.r if r is None else float(r)
        d = self.d if d is None else float(d)
        return Typeflag, cpflag, S0, K, T, vol, r, d

    # -------------------- solveur CN --------------------

    def CN_option_info(
        self,
        Typeflag: str | None = None,
        cpflag: str | None = None,
        S0: float | None = None,
        K: float | None = None,
        T: float | None = None,
        vol: float | None = None,
        r: float | None = None,
        d: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Résout la PDE et retourne (Price, Delta, Gamma, Theta).
        """

        Typeflag, cpflag, S0, K, T, vol, r, d = self._resolve_params(
            Typeflag, cpflag, S0, K, T, vol, r, d
        )

        Typeflag = Typeflag.strip()
        cpflag = cpflag.strip()
        if Typeflag not in {"Eu", "Am", "Bmd"}:
            raise ValueError("Typeflag doit être 'Eu', 'Am' ou 'Bmd'.")
        if cpflag not in {"c", "p"}:
            raise ValueError("cpflag doit être 'c' ou 'p'.")

        # Cas trivial T=0
        if T <= 0.0 or self.n_time <= 0:
            payoff0 = max(S0 - K, 0.0) if cpflag == "c" else max(K - S0, 0.0)
            return float(payoff0), 0.0, 0.0, 0.0

        if Typeflag == "Bmd":
            M_lsmc = max(1, min(self.n_time, 50))
            N_paths = 50_000
            n_ex_dates = self.n_exercise_dates or 6
            seed_base = 12345

            def _lsmc_price(s0_val: float, t_val: float) -> float:
                from app.model.options import logic as opt_logic

                return opt_logic.price_bermudan_lsmc(
                    S0=s0_val,
                    K=K,
                    T=max(t_val, 1e-6),
                    r=r,
                    q=d,
                    sigma=vol,
                    cpflag=cpflag,
                    M=M_lsmc,
                    N_paths=N_paths,
                    degree=3,
                    n_ex_dates=n_ex_dates,
                    seed=seed_base,
                )

            price_bmd = _lsmc_price(S0, T)

            bump_s = max(1e-4, 0.01 * S0)
            price_up = _lsmc_price(S0 + bump_s, T)
            price_down = _lsmc_price(max(S0 - bump_s, 1e-6), T)
            delta = (price_up - price_down) / (2.0 * bump_s)
            gamma = (price_up - 2.0 * price_bmd + price_down) / (bump_s**2)

            theta = 0.0
            theta_h = min(max(1.0 / 365.0, 0.01 * T), max(T / 2.0, 1e-6))
            if T > theta_h:
                price_short = _lsmc_price(S0, T - theta_h)
                theta = (price_short - price_bmd) / theta_h

            return float(price_bmd), float(delta), float(gamma), float(theta)

        # ----- Grille en log(S) -----
        mu = r - d - 0.5 * vol * vol
        x_max = vol * np.sqrt(max(T, 1e-8)) * 5.0
        n_points = self.n_spatial
        dx = 2.0 * x_max / n_points

        X = np.linspace(-x_max, x_max, n_points + 1)
        max_log = np.log(np.finfo(float).max / max(S0, 1e-12))
        X_clipped = np.clip(X, -max_log, max_log)
        s_grid = S0 * np.exp(X_clipped)

        n_index = np.arange(0, n_points + 1)

        n_time = self.n_time
        dt = T / n_time

        a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
        b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
        c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

        main_diag_A = 1.0 - b - 2.0 * a
        upper_A = a + c
        lower_A = a - c

        main_diag_B = 1.0 + b + 2.0 * a
        upper_B = -a - c
        lower_B = -a + c

        A = np.zeros((n_points + 1, n_points + 1))
        B = np.zeros((n_points + 1, n_points + 1))

        np.fill_diagonal(A, main_diag_A)
        np.fill_diagonal(A[1:], lower_A[:-1])
        np.fill_diagonal(A[:, 1:], upper_A[:-1])
        A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

        np.fill_diagonal(B, main_diag_B)
        np.fill_diagonal(B[1:], lower_B[:-1])
        np.fill_diagonal(B[:, 1:], upper_B[:-1])
        B = np.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)

        lu_factor_A = lu_factor(A)

        # Payoff terminal
        if cpflag == "c":
            values = np.maximum(s_grid - K, 0.0)
        else:
            values = np.maximum(K - s_grid, 0.0)

        payoff = values.copy()
        values_prev_time = values.copy()

        S_max = s_grid[-1]
        S_min = s_grid[0]  # pas utilisé mais dispo si besoin

        if Typeflag == "Am":
            exercise_dates = None
        elif Typeflag == "Bmd":
            if self.exercise_step is not None:
                exercise_dates = np.arange(0, n_time + 1, self.exercise_step)
            elif self.n_exercise_dates is not None:
                exercise_dates = np.linspace(0, n_time, self.n_exercise_dates).astype(int)
            else:
                exercise_dates = np.arange(0, n_time + 1, max(1, n_time // 5))
        else:
            exercise_dates = None

        for t_idx in range(n_time):
            values = B.dot(values)
            values, _ = lu_solve(lu_factor_A, values, check_finite=False), None
            values = np.clip(values, 0.0, 1e12)

            if Typeflag in {"Am", "Bmd"}:
                intrinsic = payoff
                if Typeflag == "Bmd":
                    if exercise_dates is None or t_idx not in exercise_dates:
                        intrinsic = values
                values = np.maximum(values, intrinsic)

            s_max = S_max * np.exp(mu * dt + vol * np.sqrt(dt))
            s_min = S_min * np.exp(mu * dt - vol * np.sqrt(dt))
            values[-1] = values[-2] * s_max / s_grid[-2]
            values[0] = values[1] * s_min / s_grid[1]

            if t_idx == n_time - 1:
                values_prev_time = values.copy()

        middle_index = n_points // 2
        price = values[middle_index]

        s_plus = S0 * np.exp(dx)
        s_minus = S0 * np.exp(-dx)
        delta = (values[middle_index + 1] - values[middle_index - 1]) / (s_plus - s_minus)

        d_value_d_s_plus = (values[middle_index + 1] - values[middle_index]) / (s_plus - S0)
        d_value_d_s_minus = (values[middle_index] - values[middle_index - 1]) / (S0 - s_minus)
        gamma = (d_value_d_s_plus - d_value_d_s_minus) / ((s_plus - s_minus) / 2.0)

        theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

        return float(price), float(delta), float(gamma), float(theta)


def CN_Barrier_option(Typeflag, cpflag, S0, K, Hu, Hd, T, vol, r, d):
    """
    Pricing d'une option barrière par Crank–Nicolson.
    """
    mu = r - d - 0.5 * vol * vol
    x_max = vol * np.sqrt(T) * 5
    n_points = 500
    dx = 2 * x_max / n_points
    X = np.linspace(-x_max, x_max, n_points + 1)
    n_index = np.arange(0, n_points + 1)

    n_time = 600
    dt = T / n_time

    a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
    b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
    c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

    main_diag_A = 1 - b - 2 * a
    upper_A = a + c
    lower_A = a - c

    main_diag_B = 1 + b + 2 * a
    upper_B = -a - c
    lower_B = -a + c

    A = np.zeros((n_points + 1, n_points + 1))
    B = np.zeros((n_points + 1, n_points + 1))

    np.fill_diagonal(A, main_diag_A)
    np.fill_diagonal(A[1:], lower_A[:-1])
    np.fill_diagonal(A[:, 1:], upper_A[:-1])

    np.fill_diagonal(B, main_diag_B)
    np.fill_diagonal(B[1:], lower_B[:-1])
    np.fill_diagonal(B[:, 1:], upper_B[:-1])

    Ainv = np.linalg.inv(A)
    s_grid = S0 * np.exp(X)
    if cpflag == "c":
        values = np.clip(s_grid - K, 0, 1e10)
    elif cpflag == "p":
        values = np.clip(K - s_grid, 0, 1e10)
    else:
        raise ValueError("cpflag doit être 'c' ou 'p'.")

    typeflag = Typeflag.upper()
    if typeflag in {"UNO", "UO"}:
        values = np.where(s_grid < Hu, values, 0.0)
    elif typeflag == "DNO":
        values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
    elif typeflag in {"DO"}:
        values = np.where(s_grid > Hd, values, 0.0)
    else:
        raise ValueError("Typeflag doit être 'UNO', 'UO', 'DO' ou 'DNO'.")

    values_prev_time = values.copy()

    for time_index in range(n_time):
        if time_index == n_time - 1:
            values_prev_time = values.copy()

        values = B.dot(values)
        values = Ainv.dot(values)

        s_grid = S0 * np.exp(X)
        if typeflag in {"UNO", "UO"}:
            values = np.where(s_grid < Hu, values, 0.0)
        elif typeflag == "DNO":
            values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
        elif typeflag == "DO":
            values = np.where(s_grid > Hd, values, 0.0)

    middle_index = n_points // 2
    price = values[middle_index]

    s_plus = S0 * np.exp(dx)
    s_minus = S0 * np.exp(-dx)

    delta = (values[middle_index + 1] - values[middle_index - 1]) / (s_plus - s_minus)

    d_value_d_s_plus = (values[middle_index + 1] - values[middle_index]) / (s_plus - S0)
    d_value_d_s_minus = (values[middle_index] - values[middle_index - 1]) / (S0 - s_minus)
    gamma = (d_value_d_s_plus - d_value_d_s_minus) / ((s_plus - s_minus) / 2.0)

    theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

    return float(price), float(delta), float(gamma), float(theta)


class BasketOption:
    def __init__(self, weights, prices, volatility, corr, strike, maturity, rate):
        self.weights = weights
        self.vol = volatility
        self.strike = strike
        self.mat = maturity
        self.rate = rate
        self.corr = corr
        self.prices = prices

    def get_mc(self, m_paths: int = 10000):
        b_ts = stats.multivariate_normal(np.zeros(len(self.weights)), cov=self.corr).rvs(
            size=m_paths
        )
        s_ts = self.prices * np.exp((self.rate - 0.5 * self.vol**2) * self.mat + self.vol * b_ts)
        if len(self.weights) > 1:
            payoffs = (np.sum(self.weights * s_ts, axis=1) - self.strike).clip(0)
        else:
            payoffs = np.maximum(s_ts - self.strike, np.zeros(m_paths))
        return float(np.exp(-self.rate * self.mat) * np.mean(payoffs))

    def get_bs_price(self):
        """
        Approximate BS price by collapsing the basket to a weighted spot.
        """
        S_eff = (
            float(np.dot(self.weights, self.prices)) if np.ndim(self.prices) else float(self.prices)
        )
        if S_eff <= 0 or self.strike <= 0 or self.vol <= 0 or self.mat <= 0:
            return 0.0

        d1 = (np.log(S_eff / self.strike) + (self.rate + 0.5 * self.vol**2) * self.mat) / (
            self.vol * np.sqrt(self.mat)
        )
        d2 = d1 - self.vol * np.sqrt(self.mat)
        bs_price = stats.norm.cdf(d1) * S_eff - stats.norm.cdf(d2) * self.strike * np.exp(
            -self.rate * self.mat
        )
        return float(bs_price)


class DataGen:
    def __init__(self, n_assets: int, n_samples: int):
        if n_samples <= 0:
            raise ValueError("n_samples needs to be positive")
        if n_assets <= 0:
            raise ValueError("n_assets needs to be positive")

        self.n_assets = n_assets
        self.n_samples = n_samples

        if self.n_assets == 1:
            self.weights = np.array([[1.0]] * self.n_samples)
        else:
            random_weights = np.random.dirichlet(alpha=np.ones(self.n_assets), size=self.n_samples)
            self.weights = random_weights

    def generate(self, corr: np.ndarray, strike_price: float, base_price: float, method="bs"):
        if corr.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"corr must be a {self.n_assets}x{self.n_assets} matrix")

        if method not in {"bs", "mc"}:
            raise ValueError("method must be either 'bs' or 'mc'")

        prices = stats.lognorm(s=0.2, scale=base_price).rvs(size=(self.n_samples, self.n_assets))
        vols = np.asarray(stats.lognorm(s=0.1, scale=0.2).rvs(size=self.n_samples)).ravel()
        # One strike per sample (shared across assets)
        strikes = stats.lognorm(s=0.1, scale=strike_price).rvs(size=(self.n_samples, 1))
        mats = np.asarray(stats.uniform.rvs(loc=0.1, scale=2.0, size=self.n_samples)).ravel()
        rates = np.asarray(stats.norm.rvs(loc=0.05, scale=0.01, size=self.n_samples)).ravel()

        labels = []
        weights = self.weights
        for i in range(self.n_samples):
            basket = BasketOption(
                weights[i],
                prices[i],
                vols[i],
                corr,
                float(strikes[i]),
                mats[i],
                rates[i],
            )
            if method == "bs":
                labels.append(basket.get_bs_price())
            else:
                labels.append(basket.get_mc())

        s_over_k = np.mean(prices / strikes, axis=1).ravel()
        prices_avg = prices.mean(axis=1).ravel()
        strikes_flat = strikes.ravel()
        labels_arr = np.asarray(labels, dtype=float).ravel()

        data = pd.DataFrame(
            {
                "S/K": s_over_k,
                "Maturity": mats,
                "Volatility": vols,
                "Rate": rates,
                "Labels": labels_arr,
                "Prices": prices_avg,
                "Strikes": strikes_flat,
            }
        )
        for i in range(self.n_assets):
            data[f"Weight_{i}"] = weights[:, i]
        return data


def simulate_dataset_notebook(
    n_assets: int,
    n_samples: int,
    method: str,
    corr: np.ndarray,
    base_price: float,
    base_strike: float,
):
    generator = DataGen(n_assets=n_assets, n_samples=n_samples)
    return generator.generate(
        corr=corr, strike_price=base_strike, base_price=base_price, method=method
    )


def split_data_nn(data: pd.DataFrame, split_ratio: float = 0.7):
    feature_cols = ["S/K", "Maturity", "Volatility", "Rate"]
    target_col = "Labels"
    train = data.iloc[: int(split_ratio * len(data)), :]
    test = data.iloc[int(split_ratio * len(data)) :, :]
    x_train, y_train = train[feature_cols], train[target_col]
    x_test, y_test = test[feature_cols], test[target_col]
    return x_train, y_train, x_test, y_test


def build_model_nn(input_dim: int):
    import tensorflow as tf

    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(1, activation="relu")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error"],
    )
    return model


def price_basket_nn(
    model, S: float, K: float, maturity: float, volatility: float, rate: float
) -> float:
    S_over_K = S / K
    x = np.array([[S_over_K, maturity, volatility, rate]], dtype=float)
    return float(model.predict(x, verbose=0)[0, 0])


def plot_heatmap_nn(
    model,
    data: pd.DataFrame,
    spot_ref: float | None = None,
    strike_ref: float | None = None,
    maturity_fixed: float = 1.0,
):
    df = data.copy()
    if "Prices" not in df.columns and spot_ref is not None:
        df["Prices"] = spot_ref
    if "Strikes" not in df.columns and strike_ref is not None:
        df["Strikes"] = strike_ref
    if not {"Prices", "Strikes"}.issubset(df.columns):
        raise ValueError(
            "Colonnes Prices et Strikes requises pour reproduire la heatmap du notebook."
        )

    s_min, s_max = df["Prices"].quantile([0.01, 0.99])
    k_min, k_max = df["Strikes"].quantile([0.01, 0.99])
    n_S, n_K = 50, 50
    s_vals = np.linspace(s_min, s_max, n_S)
    k_vals = np.linspace(k_min, k_max, n_K)
    K_grid, S_grid = np.meshgrid(k_vals, s_vals)
    s_over_k_grid = S_grid / K_grid
    sigma_ref = float(df["Volatility"].median())
    rate_ref = float(df["Rate"].median())

    X = np.stack(
        [
            s_over_k_grid.ravel(),
            np.full(s_over_k_grid.size, maturity_fixed),
            np.full(s_over_k_grid.size, sigma_ref),
            np.full(s_over_k_grid.size, rate_ref),
        ],
        axis=1,
    )
    prices_grid = model.predict(X, verbose=0).reshape(n_S, n_K)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        prices_grid,
        origin="lower",
        extent=[k_vals.min(), k_vals.max(), s_vals.min(), s_vals.max()],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Spot S")
    ax.set_title("Heatmap du prix NN en fonction de S et K (T=1 an)")
    fig.colorbar(im, ax=ax, label="Prix NN")
    plt.tight_layout()
    return fig


def make_iv_surface_figure(k_grid, t_grid, iv_grid, title_suffix=""):
    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    iv_flat = iv_grid[~np.isnan(iv_grid)]
    if iv_flat.size == 0:
        raise ValueError("La grille iv_grid ne contient aucune valeur non-NaN.")
    iv_mean = iv_flat.mean()
    iv_grid_filled = np.where(np.isnan(iv_grid), iv_mean, iv_grid)
    surf = ax3d.plot_surface(
        k_grid,
        t_grid,
        iv_grid_filled,
        rstride=1,
        cstride=1,
        linewidth=0.2,
        antialiased=True,
        cmap="viridis",
    )
    ax3d.set_xlabel("Strike K")
    ax3d.set_ylabel("Maturité T (années)")
    ax3d.set_zlabel("Implied vol")
    ax3d.set_title(f"Surface 3D de volatilité implicite{title_suffix}")
    fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax3d, label="iv")

    ax2d = fig.add_subplot(1, 2, 2)
    im = ax2d.imshow(
        iv_grid_filled,
        extent=[k_grid.min(), k_grid.max(), t_grid.min(), t_grid.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax2d.set_xlabel("Strike K")
    ax2d.set_ylabel("Maturité T (années)")
    ax2d.set_title(f"Heatmap IV{title_suffix}")
    fig.colorbar(im, ax=ax2d, label="iv")
    plt.tight_layout()
    return fig


def btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps):
    from Asian.analytic import btm_asian, btm_asian_float

    if strike_type == "floating":
        return btm_asian_float(
            strike_type="floating",
            option_type=option_type,
            spot=spot,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
            steps=int(steps),
        )
    return btm_asian(
        strike_type=strike_type,
        option_type=option_type,
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        steps=int(steps),
    )


def hw_btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps, m_points):
    from Asian.hull_white_btm import asian_hull_white

    return asian_hull_white(
        strike_type=strike_type,
        option_type=option_type,
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        steps=int(steps),
        m_points=int(m_points),
    )


def bs_option_price(time, spot, strike, maturity, rate, sigma, option_kind):
    tau = maturity - time
    if tau <= 0:
        if option_kind == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_kind == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return float(price)


def asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type):
    if n_obs < 1:
        return 0.0

    dt = maturity / n_obs
    nu = rate - 0.5 * sigma**2
    sigma_g_sq = (sigma**2) * (n_obs + 1) * (2 * n_obs + 1) / (6 * n_obs**2)
    sigma_g = np.sqrt(sigma_g_sq)
    mu_g = (nu * (n_obs + 1) / (2 * n_obs) + 0.5 * sigma_g_sq) * maturity

    d1 = (np.log(spot / strike) + mu_g + 0.5 * sigma_g_sq * maturity) / (
        sigma_g * np.sqrt(maturity)
    )
    d2 = d1 - sigma_g * np.sqrt(maturity)
    df = np.exp(-rate * maturity)

    if option_type == "call":
        return float(df * (spot * np.exp(mu_g) * norm.cdf(d1) - strike * norm.cdf(d2)))
    return float(df * (strike * norm.cdf(-d2) - spot * np.exp(mu_g) * norm.cdf(-d1)))


def asian_mc_control_variate(
    strike_type,
    option_type,
    spot,
    strike,
    rate,
    sigma,
    maturity,
    n_paths=100_000,
    n_obs=52,
    antithetic=True,
    seed=None,
    use_control=True,
):
    rng = np.random.default_rng(seed)
    dt = maturity / n_obs
    drift = (rate - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    n_eff = n_paths // 2 if antithetic else n_paths
    shocks = rng.standard_normal(size=(n_eff, n_obs))
    shocks_all = np.vstack([shocks, -shocks]) if antithetic else shocks
    shocks_all = shocks_all[:n_paths]

    paths = np.empty((n_paths, n_obs + 1))
    paths[:, 0] = spot
    for i in range(n_obs):
        paths[:, i + 1] = paths[:, i] * np.exp(drift + diff * shocks_all[:, i])

    if strike_type == "floating":
        strike_vals = np.mean(paths[:, 1:], axis=1)
        strike_eff = strike_vals
    else:
        strike_eff = strike

    arith_avg = np.mean(paths[:, 1:], axis=1)
    geom_avg = np.exp(np.mean(np.log(paths[:, 1:] + 1e-12), axis=1))

    if option_type == "call":
        arith_payoff = np.maximum(arith_avg - strike_eff, 0.0)
        geom_payoff = np.maximum(geom_avg - strike_eff, 0.0)
    else:
        arith_payoff = np.maximum(strike_eff - arith_avg, 0.0)
        geom_payoff = np.maximum(strike_eff - geom_avg, 0.0)

    closed_geom = asian_geometric_closed_form(
        spot,
        strike if strike_type == "fixed" else strike,
        rate,
        sigma,
        maturity,
        n_obs,
        option_type,
    )

    if use_control:
        cov = np.cov(arith_payoff, geom_payoff)[0, 1]
        var_geom = np.var(geom_payoff)
        c = cov / var_geom if var_geom > 0 else 0.0
        control_estimator = arith_payoff - c * (geom_payoff - closed_geom)
        disc = np.exp(-rate * maturity)
        disc_payoff = disc * control_estimator
        price = np.mean(disc_payoff)
        stderr = np.std(disc_payoff, ddof=1) / np.sqrt(n_eff)
        return float(price), float(stderr), float(c)

    disc = np.exp(-rate * maturity)
    disc_payoff = disc * arith_payoff
    price = np.mean(disc_payoff)
    stderr = np.std(disc_payoff, ddof=1) / np.sqrt(n_eff)
    return float(price), float(stderr), 0.0


def compute_asian_price(
    strike_type: str,
    option_type: str,
    model: str,
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    steps: int,
    m_points: int | None,
):
    if model == "BTM naïf":
        return btm_asian(
            strike_type=strike_type,
            option_type=option_type,
            spot=spot,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
            steps=int(steps),
        )

    m_points_val = int(m_points) if m_points is not None else 10
    return hw_btm_asian(
        strike_type=strike_type,
        option_type=option_type,
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        steps=int(steps),
        m_points=m_points_val,
    )

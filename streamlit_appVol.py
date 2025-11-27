import sys
import io
import math
import random
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy import stats
from scipy.optimize import minimize
from wordcloud import WordCloud

BASE_DIR = Path(__file__).parent
# Data and helper modules now live under scripts/scriptsRL; datasets live in database/RLtab/.
SOURCES_DIR = BASE_DIR / "scripts" / "scriptsRL"
DATA_DIR = BASE_DIR / "database" / "RLtab"
# Ensure helper modules under scripts/scriptsRL are importable
if SOURCES_DIR.exists() and str(SOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCES_DIR))
RL4F_BASE = SOURCES_DIR / "rl4f"
RL4F_DIRS = [
    RL4F_BASE / "trading_dql",
    RL4F_BASE / "hedging_dql",
    RL4F_BASE / "allocation_3ac",
]
DATA_ROOTS = [path for path in [SOURCES_DIR, DATA_DIR] if path.exists()]
NEXT_DATA_DIRS = sorted(
    [path for root in DATA_ROOTS for path in root.rglob("data") if path.is_dir()],
    key=lambda p: str(p),
)
for path in RL4F_DIRS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finance import Finance
from dqlagent_pytorch import DQLAgent, device
from bsm73 import bsm_call_value
from assetallocation_pytorch import Investing, InvestingAgent


# ---------- Helpers ----------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def capture_logs(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


@st.cache_data(show_spinner=False)
def load_raw_data():
    url = "https://certificate.tpq.io/rl4finance.csv"
    return pd.read_csv(url, index_col=0, parse_dates=True).dropna()


@st.cache_data(show_spinner=False)
def load_next_csv(name, parse_dates=None):
    candidates = []
    for data_dir in NEXT_DATA_DIRS:
        candidate = data_dir / name
        if candidate.exists():
            candidates.append(candidate)
    for root in DATA_ROOTS:
        fallback = root / name
        if fallback.exists():
            candidates.append(fallback)
    if not candidates:
        raise FileNotFoundError(
            f"Fichier manquant : {name} (cherché dans {SOURCES_DIR})"
        )
    path = candidates[0]
    read_attempts = [
        {"sep": None, "engine": None, "encoding": None},
        {"sep": ";", "engine": None, "encoding": None},
        {"sep": None, "engine": "python", "encoding": None},
        {"sep": None, "engine": None, "encoding": "latin1"},
        {"sep": ";", "engine": None, "encoding": "latin1"},
        {"sep": None, "engine": "python", "encoding": "latin1"},
    ]
    last_error = None
    for cfg in read_attempts:
        try:
            return pd.read_csv(
                path,
                parse_dates=parse_dates,
                sep=cfg["sep"],
                engine=cfg["engine"],
                encoding=cfg["encoding"],
                on_bad_lines="skip",
            )
        except Exception as exc:
            last_error = exc
            continue
    raise last_error


def train_test_split_series(series, train_ratio=0.8):
    n_train = max(5, int(len(series) * train_ratio))
    train = series.iloc[:n_train]
    test = series.iloc[n_train:]
    return train, test


def polyfit_predict(series, deg=2):
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, deg=deg)
    poly = np.poly1d(coeffs)
    preds = poly(x)
    next_pred = float(poly(len(series)))
    return preds, next_pred


def inverse_variance_weights(cov):
    inv_var = 1 / np.clip(np.diag(cov), 1e-8, None)
    w = inv_var / inv_var.sum()
    return w


def pca_svd(matrix, n_components=3):
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    comps = Vt[:n_components]
    explained = (s ** 2) / (len(matrix) - 1)
    explained_ratio = explained / explained.sum()
    scores = np.dot(matrix, comps.T)
    return comps, scores, explained_ratio


def bsm_put_value(St, K, T, t, r, sigma):
    call = bsm_call_value(St, K, T, t, r, sigma)
    return call - St + K * math.exp(-r * (T - t))


def plot_rewards(rewards, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rewards, color="tab:blue", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(True, alpha=0.3)
    return fig


# ---------- Hedging classes (notebook 07) ----------
class ObservationSpace:
    def __init__(self, n):
        self.shape = (n,)


class ActionSpace:
    def __init__(self, n):
        self.n = n

    def seed(self, seed):
        random.seed(seed)

    def sample(self):
        return random.random()


def simulate_gbm(S0, T, r, sigma, steps):
    path = [S0]
    dt = T / steps
    for _ in range(steps):
        st = path[-1] * math.exp((r - sigma ** 2 / 2) * dt +
                                 sigma * math.sqrt(dt) * random.gauss(0, 1))
        path.append(st)
    return np.array(path)


def bsm_delta(St, K, T, t, r, sigma):
    d1 = ((math.log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)) /
          (sigma * math.sqrt(T - t)))
    return stats.norm.cdf(d1, 0, 1)


def option_replication(path, K, T, r, sigma):
    dt = T / (len(path) - 1)
    bond = [math.exp(r * i * dt) for i in range(len(path))]
    res = pd.DataFrame()
    for i in range(len(path) - 1):
        C = bsm_call_value(path[i], K, T, i * dt, r, sigma)
        if i == 0:
            s = bsm_delta(path[i], K, T, i * dt, r, sigma)
            b = (C - s * path[i]) / bond[i]
        else:
            V = s * path[i] + b * bond[i]
            s = bsm_delta(path[i], K, T, i * dt, r, sigma)
            b = (C - s * path[i]) / bond[i]
            df = pd.DataFrame({"St": path[i], "C": C, "V": V,
                               "s": s, "b": b}, index=[0])
            res = pd.concat((res, df), ignore_index=True)
    return res


class Hedging:
    def __init__(self, S0, K_, T, r_, sigma_, steps):
        self.initial_value = S0
        self.strike_ = K_
        self.maturity = T
        self.short_rate_ = r_
        self.volatility_ = sigma_
        self.steps = steps
        self.observation_space = ObservationSpace(5)
        self.osn = self.observation_space.shape[0]
        self.action_space = ActionSpace(1)
        self._simulate_data()
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _simulate_data(self):
        s = [self.initial_value]
        self.strike = random.choice(self.strike_)
        self.short_rate = random.choice(self.short_rate_)
        self.volatility = random.choice(self.volatility_)
        self.dt = self.maturity / self.steps
        for _ in range(1, self.steps + 1):
            st = s[-1] * math.exp(
                (self.short_rate - self.volatility ** 2 / 2) * self.dt +
                self.volatility * math.sqrt(self.dt) * random.gauss(0, 1))
            s.append(st)
        self.data = pd.DataFrame(s, columns=["index"])
        self.data["bond"] = np.exp(self.short_rate *
                                   np.arange(len(self.data)) * self.dt)

    def _get_state(self):
        St = self.data["index"].iloc[self.bar]
        Bt = self.data["bond"].iloc[self.bar]
        ttm = self.maturity - self.bar * self.dt
        if ttm > 0:
            Ct = bsm_call_value(St, self.strike, self.maturity,
                                self.bar * self.dt, self.short_rate,
                                self.volatility)
        else:
            Ct = max(St - self.strike, 0)
        return np.array([St, Bt, ttm, Ct, self.strike, self.short_rate,
                         self.stock, self.bond]), {}

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def reset(self):
        self.bar = 0
        self.bond = 0
        self.stock = 0
        self.treward = 0
        self.episode += 1
        self._simulate_data()
        self.state, _ = self._get_state()
        return self.state, _

    def step(self, action):
        if self.bar == 0:
            reward = 0
            self.bar += 1
            self.stock = float(action)
            self.bond = ((self.state[3] - self.stock * self.state[0]) /
                         self.state[1])
            self.new_state, _ = self._get_state()
        else:
            self.bar += 1
            self.new_state, _ = self._get_state()
            phi_value = (self.stock * self.new_state[0] +
                         self.bond * self.new_state[1])
            pl = phi_value - self.new_state[3]
            df = pd.DataFrame({"e": self.episode, "s": self.stock,
                               "b": self.bond, "phi": phi_value,
                               "C": self.new_state[3], "p&l[$]": pl,
                               "p&l[%]": pl / max(self.new_state[3], 1e-4) * 100,
                               "St": self.new_state[0],
                               "Bt": self.new_state[1],
                               "K": self.strike, "r": self.short_rate,
                               "sigma": self.volatility}, index=[0])
            self.portfolios = pd.concat((self.portfolios, df),
                                        ignore_index=True)
            reward = -(phi_value - self.new_state[3]) ** 2
            self.stock = float(action)
            self.bond = ((self.new_state[3] -
                          self.stock * self.new_state[0]) /
                         self.new_state[1])
        done = self.bar == len(self.data) - 1
        self.state = self.new_state
        return self.state, float(reward), done, False, {}


class HedgingAgent(DQLAgent):
    def opt_action(self, state):
        bnds = [(0, 1)]

        def f_obj(x):
            s = state.copy()
            s[0, 6] = x
            s[0, 7] = ((s[0, 3] - x * s[0, 0]) / s[0, 1])
            s_tensor = torch.FloatTensor(s).to(device)
            with torch.no_grad():
                q_val = self.model(s_tensor)
            return q_val.cpu().numpy()[0, 0]

        try:
            res = minimize(lambda x: -f_obj(x), 0.5,
                           bounds=bnds, method="Powell")
            action = res["x"][0]
        except Exception:
            action = self.env.stock
        return action

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return self.opt_action(state)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            target = torch.tensor([reward], dtype=torch.float32).to(device)
            if not done:
                ns = next_state.copy()
                opt_act = self.opt_action(ns)
                ns[0, 6] = opt_act
                ns[0, 7] = ((ns[0, 3] - opt_act * ns[0, 0]) / ns[0, 1])
                ns_tensor = torch.FloatTensor(ns).to(device)
                with torch.no_grad():
                    future_q = self.model(ns_tensor)[0, 0]
                target = target + self.gamma * future_q
            state_tensor = torch.FloatTensor(state).to(device)
            self.optimizer.zero_grad()
            current_q = self.model(state_tensor)[0, 0]
            loss = self.criterion(current_q, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def test(self, episodes, verbose=True):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for _ in range(1, len(self.env.data) + 1):
                action = self.opt_action(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                treward += reward
                if done:
                    if verbose:
                        templ = f"total penalty={treward:4.2f}"
                        print(templ)
                    break


# ---------- UI renderers ----------
def render_rl_trading():
    st.subheader("RL · Trading (DQL)")
    st.write("Aperçu des données utilisées par l’agent de trading.")
    df = load_raw_data()
    col = st.selectbox("Colonne à tracer", list(df.columns), key="rl_trading_col")
    horizon = st.slider(
        "Fenêtre (jours)",
        50,
        min(500, len(df)),
        min(200, len(df)),
        25,
        key="rl_trading_window",
    )
    series = df[col].tail(horizon)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(series.index, series.values, lw=1.2, color="tab:blue")
    ax.set_title(f"{col} — derniers {len(series)} points")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_rl_hedging():
    st.subheader("RL · Hedging (environnement)")
    st.write("Simulation rapide d’un épisode aléatoire de couverture delta.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S0 = st.number_input("S0", value=100.0, step=1.0, key="rlh_s0")
        K = st.number_input("Strike", value=100.0, step=1.0, key="rlh_k")
    with col_b:
        T = st.number_input("Maturité (années)", value=1.0, step=0.25, key="rlh_t")
        r = st.number_input("Taux r", value=0.01, format="%.4f", key="rlh_r")
    with col_c:
        sigma = st.number_input("Vol (sigma)", value=0.2, format="%.4f", key="rlh_sigma")
        steps = st.slider("Steps", 20, 200, 80, 10, key="rlh_steps")
    seed = st.number_input("Seed", value=123, step=1, key="rlh_seed")

    if st.button("Lancer un épisode", key="rlh_run"):
        set_global_seed(seed)
        env = Hedging(S0=S0, K_=[K], T=T, r_=[r], sigma_=[sigma], steps=steps)
        env.seed(seed)
        state, _ = env.reset()
        for _ in range(env.steps):
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            if done:
                break
        st.write(env.portfolios.tail())
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(env.data["index"].values, lw=1.1, label="Sous-jacent")
        if not env.portfolios.empty:
            ax2 = ax.twinx()
            ax2.plot(env.portfolios["C"].values, color="tab:red", lw=1.0, label="Call")
            ax2.plot(env.portfolios["phi"].values, color="tab:green", lw=1.0, linestyle="--", label="Portefeuille")
            ax2.legend(loc="upper right")
        ax.set_title("Trajectoire simulée (env RL hedging)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        st.pyplot(fig)


def render_rl_allocation():
    st.subheader("RL · Allocation (crypto 3 actifs)")
    st.write("Aperçu des pondérations inverse-variance sur le dataset crypto.")
    try:
        df_crypto = load_next_csv("portfolio_allocation/data/crypto_portfolio.csv", parse_dates=["Date"])
        df_crypto = df_crypto.set_index("Date")
        ret = df_crypto.pct_change().dropna()
        method = st.radio(
            "Pondération RL (crypto)",
            ["Égal pondéré", "Inverse variance"],
            key="rl_alloc_method",
        )
        cov = ret.cov()
        if method == "Inverse variance":
            w = inverse_variance_weights(cov.values)
        else:
            w = np.repeat(1 / ret.shape[1], ret.shape[1])
        weights = pd.Series(w, index=ret.columns)
        st.write("Pondérations", weights.round(4))
        strat = (ret * weights).sum(axis=1)
        perf = (1 + strat).cumprod()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(perf.index, perf.values, lw=1.3, color="tab:purple")
        ax.set_title("Valeur de portefeuille (base 1)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Lecture impossible ({exc})")


def render_pricing_derivatives():
    st.subheader("Pricing · Derivatives")
    st.write("Prix BSM (call/put) et courbe en fonction du strike.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S = st.number_input("Spot S", value=100.0, step=1.0)
        K = st.number_input("Strike K", value=100.0, step=1.0)
    with col_b:
        T = st.number_input("Maturité (années)", value=1.0, step=0.25, key="pricing_maturity")
        r = st.number_input("Taux r", value=0.01, format="%.4f")
    with col_c:
        sigma = st.number_input("Vol (sigma)", value=0.20, format="%.4f", key="pricing_sigma")
        step_strikes = st.slider("Variation autour de K", 5, 50, 20, 5)

    call = bsm_call_value(S, K, T, 0, r, sigma)
    put = bsm_put_value(S, K, T, 0, r, sigma)
    st.metric("Call BSM", f"{call:.4f}")
    st.metric("Put BSM", f"{put:.4f}")

    strikes = np.linspace(max(1, K - step_strikes), K + step_strikes, 25)
    call_curve = [bsm_call_value(S, k, T, 0, r, sigma) for k in strikes]
    put_curve = [bsm_put_value(S, k, T, 0, r, sigma) for k in strikes]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(strikes, call_curve, label="Call", lw=1.4)
    ax.plot(strikes, put_curve, label="Put", lw=1.4)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Prix")
    ax.set_title("Courbes de prix BSM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def render_hedging_derivatives():
    st.subheader("Hedging · Derivatives")
    st.write("Réplication delta avec GBM simulé (version rapide).")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S0_h = st.number_input("S0", value=100.0, step=1.0, key="dh_s0")
        K_h = st.number_input("Strike", value=100.0, step=1.0, key="dh_k")
    with col_b:
        T_h = st.number_input("Maturité", value=1.0, step=0.25, key="dh_t")
        r_h = st.number_input("Taux", value=0.01, format="%.4f", key="dh_r")
    with col_c:
        sigma_h = st.number_input("Vol", value=0.2, format="%.4f", key="dh_sigma")
        steps_h = st.slider("Steps", 30, 300, 120, 10, key="dh_steps")
    seed_dh = st.number_input("Seed", value=123, step=1, key="dh_seed")

    if st.button("Simuler réplication", key="dh_run"):
        set_global_seed(seed_dh)
        path = simulate_gbm(S0_h, T_h, r_h, sigma_h, steps_h)
        rep = option_replication(path, K_h, T_h, r_h, sigma_h)
        st.write(rep.head())
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(path, lw=1.1, label="Sous-jacent")
        ax2 = ax.twinx()
        ax2.plot(rep["C"].values, color="tab:red", label="Call", lw=1.1)
        ax2.plot(rep["V"].values, color="tab:green", label="Portefeuille", lw=1.1, linestyle="--")
        ax.set_title("Réplication delta")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        st.pyplot(fig)


def render_signals_stock_prediction():
    st.subheader("Signals · Stock Return Prediction")
    st.write("Régression simple (polyfit) sur la série `data.csv`.")

    df = load_next_csv("stock_return_prediction/data/data.csv", parse_dates=["Date"])
    df = df.set_index("Date")
    feature = st.selectbox("Colonne cible", ["Close", "Adj Close", "Open", "High", "Low"])
    degree = st.slider("Degré du polynôme", 1, 3, 2)
    train_ratio = st.slider("Taille train (%)", 60, 95, 80, 1) / 100

    series = df[feature].dropna()
    train, test = train_test_split_series(series, train_ratio)
    preds_train, next_pred = polyfit_predict(train, deg=degree)
    x_all = np.arange(len(series))
    poly = np.poly1d(np.polyfit(np.arange(len(train)), train.values, deg=degree))
    preds_all = poly(x_all)
    test_pred = preds_all[len(train):len(series)]
    mae = float(np.mean(np.abs(test.values - test_pred))) if len(test) else float("nan")

    st.metric("MAE (test)", f"{mae:.2f}")
    st.metric("Prévision prochain point", f"{next_pred:.2f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(series.index, series.values, label="Réel", lw=1.5)
    ax.plot(series.index[:len(train)], preds_train, label="Fit train", lw=1.2)
    if len(test_pred):
        ax.plot(series.index[len(train):], test_pred, label="Préd test", lw=1.2, linestyle="--")
    ax.legend()
    ax.set_title(f"Prédiction sur {feature}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_signals_btc_pca():
    st.subheader("Signals · BTC PCA (réduction de dimension)")
    st.write("PCA rapide sur OHLCV pour visualiser les composantes principales.")

    df_btc_pca = load_next_csv("BitstampData.csv")
    df_btc_pca = df_btc_pca.dropna().copy()
    df_btc_pca["Date"] = pd.to_datetime(df_btc_pca["Timestamp"], unit="s")
    df_btc_pca = df_btc_pca.set_index("Date").sort_index()
    feats = ["Open", "High", "Low", "Close", "Volume_(BTC)"]
    X = df_btc_pca[feats].values
    comps, scores, explained = pca_svd(X, n_components=3)
    st.write("Variance expliquée", [f"{x:.2%}" for x in explained])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(scores[:, 0], scores[:, 1], s=6, alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Projection PCA (PC1 vs PC2)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_signals_eigen_portfolio():
    st.subheader("Signals · Eigen Portfolio (Dow 30)")
    st.write("Première composante propre du covariance Dow_adjcloses.")

    datasets = {
        "Dataset principal": "Dow_adjcloses.csv",
        "Dataset Next": "portfolio_eigen/data/Dow_adjcloses.csv",
    }
    choice = st.selectbox("Jeu de données", list(datasets.keys()), key="signals_eigen_dataset")
    df_dow = load_next_csv(datasets[choice], parse_dates=["Date"]).set_index("Date")
    ret = df_dow.pct_change().dropna()
    cov = ret.cov()
    vals, vecs = np.linalg.eigh(cov.values)
    idx = vals.argsort()[::-1]
    top_vec = vecs[:, idx[0]]
    weights = top_vec / np.sum(np.abs(top_vec))
    w_series = pd.Series(weights, index=ret.columns)
    st.write("Pondérations (normalisées)", w_series.round(4).sort_values(ascending=False))
    strat = (ret * w_series).sum(axis=1)
    perf = (1 + strat).cumprod()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(perf.index, perf.values, lw=1.2, color="tab:orange")
    ax.set_title("Eigen-portfolio (PC1)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_signals_yield_curve_construction():
    st.subheader("Signals · Yield Curve Construction")
    st.write("PCA sur courbe de swap (`DownloadedData.csv`).")

    datasets = {
        "Dataset principal": "DownloadedData.csv",
        "Dataset Next": "yield_curve_construction/data/DownloadedData.csv",
    }
    choice = st.selectbox("Jeu de données", list(datasets.keys()), key="yc_build_dataset")
    yc = load_next_csv(datasets[choice], parse_dates=["DATE"]).set_index("DATE")
    comps, scores, explained = pca_svd(yc.values, n_components=3)
    st.write("Variance expliquée", [f"{x:.2%}" for x in explained])
    k = st.slider("Nombre de composantes pour reconstruction", 1, 3, 2, key="yc_build_components")
    recon = scores[:, :k] @ comps[:k]
    recon += yc.values.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(yc.columns, yc.iloc[-1].values, label="Dernière courbe", lw=1.4)
    ax.plot(yc.columns, recon[-1], label=f"Reconstruction {k} PC", lw=1.2, linestyle="--")
    ax.set_title("Courbe de swap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def render_signals_yield_curve_prediction():
    st.subheader("Signals · Yield Curve Prediction")
    st.write("Projection d’un pas en avant par lissage exponentiel.")

    datasets = {
        "Dataset principal": "DownloadedData.csv",
        "Dataset Next": "yield_curve_prediction/data/DownloadedData.csv",
    }
    choice = st.selectbox("Jeu de données", list(datasets.keys()), key="yc_predict_dataset")
    yc = load_next_csv(datasets[choice], parse_dates=["DATE"]).set_index("DATE")
    alpha = st.slider("Facteur de lissage", 0.01, 0.99, 0.2, 0.01, key="yc_predict_alpha")
    forecast = yc.ewm(alpha=alpha).mean().iloc[-1]
    st.write("Prévision prochaine courbe")
    st.dataframe(forecast.to_frame("Prévision").T, width="stretch")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(yc.columns, yc.iloc[-1].values, label="Dernière observation", lw=1.2)
    ax.plot(yc.columns, forecast.values, label="Prévision", lw=1.2, linestyle="--")
    ax.set_title("Prévision courte échéance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def render_strategies_btc_ma():
    st.subheader("Strategies · Bitcoin MA Crossover")
    st.write("Stratégie simple de croisement de moyennes mobiles sur `BitstampData.csv`.")

    df_btc = load_next_csv("bitcoin_trading_strategy/data/BitstampData.csv")
    df_btc = df_btc.dropna()
    df_btc["Date"] = pd.to_datetime(df_btc["Timestamp"], unit="s")
    df_btc = df_btc.set_index("Date").sort_index()
    short = st.slider("SMA courte", 3, 50, 10, 1)
    long = st.slider("SMA longue", 10, 200, 50, 5)
    df_btc["ret"] = df_btc["Close"].pct_change()
    df_btc["sma_s"] = df_btc["Close"].rolling(short).mean()
    df_btc["sma_l"] = df_btc["Close"].rolling(long).mean()
    df_btc["signal"] = np.sign(df_btc["sma_s"] - df_btc["sma_l"])
    df_btc["strat_ret"] = df_btc["signal"].shift(1) * df_btc["ret"]
    perf = (1 + df_btc[["ret", "strat_ret"]].dropna()).cumprod()
    st.metric("Perf stratégie (x)", f"{perf['strat_ret'].iloc[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(perf.index, perf["ret"], label="Buy & Hold", lw=1.0)
    ax.plot(perf.index, perf["strat_ret"], label="Stratégie SMA", lw=1.2)
    ax.legend()
    ax.set_title("Cumul des rendements")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_strategies_crypto_allocation():
    st.subheader("Strategies · Crypto Allocation")
    st.write("Pondérations inverse-variance ou égalité.")

    datasets = {
        "Dataset principal": "crypto_portfolio.csv",
        "Dataset Next": "portfolio_allocation/data/crypto_portfolio.csv",
    }
    choice = st.selectbox("Jeu de données", list(datasets.keys()), key="strategies_alloc_dataset")
    df_crypto = load_next_csv(datasets[choice], parse_dates=["Date"])
    df_crypto = df_crypto.set_index("Date")
    ret = df_crypto.pct_change().dropna()
    method = st.radio("Pondération stratégie (crypto)", ["Égal pondéré", "Inverse variance"], key="strategies_alloc_method")
    cov = ret.cov()
    if method == "Inverse variance":
        w = inverse_variance_weights(cov.values)
    else:
        w = np.repeat(1 / ret.shape[1], ret.shape[1])
    weights = pd.Series(w, index=ret.columns)
    st.write("Pondérations", weights.round(4))
    strat = (ret * weights).sum(axis=1)
    perf = (1 + strat).cumprod()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(perf.index, perf.values, lw=1.3, color="tab:purple")
    ax.set_title("Valeur de portefeuille (base 1)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_strategies_rl_sp500():
    st.subheader("Strategies · RL SP500 (aperçu data)")
    st.write("Jeu de données S&P500 utilisé dans le notebook RL.")

    df_sp = load_next_csv("reinforcement_trading_strategy/data/SP500.csv", parse_dates=["Date"])
    df_sp = df_sp.set_index("Date")
    horizon = st.slider("Fenêtre (jours)", 50, min(500, len(df_sp)), min(200, len(df_sp)), 50)
    sample_sp = df_sp.tail(horizon)
    st.dataframe(sample_sp.head(), width="stretch")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sample_sp.index, sample_sp["Close"], lw=1.2, color="tab:blue")
    ax.set_title(f"S&P500 — Close (derniers {horizon} jours)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def render_strategies_nlp_sentiment():
    st.subheader("Strategies · NLP Sentiment")
    st.write("TP guidé : avance étape par étape (choix dataset → filtres → stats → graphiques → aperçu).")

    datasets = {
        "Step4 · Sentiments": "strategies_nlp_trading/data/Step4_DataWithSentimentsResults.csv",
        "Step3 · News + Return": "strategies_nlp_trading/data/Step3_NewsAndReturnData.csv",
        "Step2.2 · Return": "strategies_nlp_trading/data/Step2.2_ReturnData.csv",
        "Lexicon": "strategies_nlp_trading/data/LexiconData.csv",
        "Labelled News": "strategies_nlp_trading/data/LabelledNewsData.csv",
        "Correlation": "strategies_nlp_trading/data/correlation.csv",
    }

    # Bloc 1 : choix du dataset
    st.markdown("### 1. Choisir le dataset")
    choice = st.radio("Fichier NLP", list(datasets.keys()), key="nlp_ds_choice")
    path = datasets[choice]

    try:
        df_nlp = load_next_csv(path)
    except Exception as exc:
        st.error(f"Lecture impossible ({exc})")
        return

    st.caption(f"Shape initiale : {df_nlp.shape}")

    # Bloc 2 : filtres (date + tickers)
    st.markdown("### 2. Filtrer par dates et tickers")
    date_cols = [c for c in df_nlp.columns if "date" in c.lower()]
    ticker_cols = [c for c in df_nlp.columns if c.lower() in {"ticker", "symbol"}]
    sentiment_cols = [c for c in df_nlp.columns if "sentiment" in c.lower()]
    return_cols = [c for c in df_nlp.columns if "return" in c.lower()]

    if date_cols:
        date_col = date_cols[0]
        df_nlp[date_col] = pd.to_datetime(df_nlp[date_col], errors="coerce")
        df_nlp = df_nlp.dropna(subset=[date_col])
        min_d, max_d = df_nlp[date_col].min().date(), df_nlp[date_col].max().date()
        start, end = st.date_input("Période", (min_d, max_d), min_value=min_d, max_value=max_d, key="nlp_date")
        if start and end:
            df_nlp = df_nlp[(df_nlp[date_col] >= pd.to_datetime(start)) & (df_nlp[date_col] <= pd.to_datetime(end))]

    if ticker_cols:
        tcol = ticker_cols[0]
        uniq = sorted(df_nlp[tcol].dropna().unique().tolist())[:100]
        selected = st.multiselect("Tickers/Symboles", uniq, default=uniq[:5], key="nlp_tickers")
        if selected:
            df_nlp = df_nlp[df_nlp[tcol].isin(selected)]

    st.caption(f"Lignes après filtres : {len(df_nlp)}")

    # Bloc 3 : stats descriptives
    st.markdown("### 3. Lire les statistiques clés")
    if sentiment_cols:
        scol = sentiment_cols[0]
        if pd.api.types.is_numeric_dtype(df_nlp[scol]):
            st.write(f"Moyenne sentiment : {df_nlp[scol].mean():.3f} | Médiane : {df_nlp[scol].median():.3f}")
            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.hist(df_nlp[scol].dropna(), bins=30, color="tab:blue", alpha=0.7)
            ax.set_title(f"Distribution de {scol}")
            st.pyplot(fig)
        else:
            counts = df_nlp[scol].value_counts().head(20)
            st.bar_chart(counts)

    # Bloc 4 : graphiques (timeline sentiment + corrélation éventuelle)
    st.markdown("### 4. Visualiser le signal (timeline, corrélation)")
    if date_cols and sentiment_cols and pd.api.types.is_numeric_dtype(df_nlp[sentiment_cols[0]]):
        scol = sentiment_cols[0]
        date_col = date_cols[0]
        daily = df_nlp.groupby(df_nlp[date_col].dt.date)[scol].mean()
        st.line_chart(daily, height=200)

    if sentiment_cols and return_cols:
        scol = sentiment_cols[0]
        rcol = return_cols[0]
        num_df = df_nlp[[scol, rcol]].dropna()
        if len(num_df) > 5:
            corr = num_df.corr().iloc[0, 1]
            st.metric("Corr(sentiment, return)", f"{corr:.3f}")

    # Bloc 5 : aperçu
    st.markdown("### 5. Inspecter un extrait des données filtrées")
    st.dataframe(df_nlp.head(200), width="stretch")


def render_extra_nlp_overview():
    st.subheader("NLP · Overview")
    st.write(
        "Mini labo NLP intégré : wordcloud synthétique, vecteurs doc-term 3D et démo cosinus. "
        "Tout est exécuté dans l’app (pas de lecture du notebook)."
    )
    st.write(
        "- Wordcloud : génère des tweets fictifs sur des tickers/sentiments.\n"
        "- Doc Vectors : matrice doc-term simplifiée + scatter 3D.\n"
        "- Cosine Sim : visualisation interactive de cos(θ) entre deux vecteurs."
    )
    st.info("Utilisez les sous-onglets pour tester ces démos.")


def render_extra_nlp_wordcloud():
    st.subheader("NLP · Wordcloud & Tokens")
    st.write("Génère des tweets fictifs (tickers + sentiments) puis un wordcloud.")

    tickers = ["$AAPL", "$TSLA", "$GOOG", "$AMZN", "$MSFT", "$NVDA", "$META", "$BTC", "$ETH", "$NFLX"]
    sentiments = ["bullish", "bearish", "range", "breakout", "momentum", "mean-revert", "macro", "earnings", "AI", "chips"]
    emojis = ["🚀", "📉", "🤖", "📈", "💤", "⚠️", "🧠", "💰"]
    n_tweets = st.slider("Nombre de tweets synthétiques", 20, 200, 80, 10, key="nlp_wc_count")
    random.seed(42)
    tweets = []
    for _ in range(n_tweets):
        t = random.choice(tickers)
        s = random.choice(sentiments)
        e = random.choice(emojis)
        tweets.append(f"{t} looks {s} {e}")
    text = " ".join(tweets)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    st.write("Aperçu des tokens :", ", ".join(tweets[:5]) + " ...")


def render_extra_nlp_vectors():
    st.subheader("NLP · Doc-Term Vectors")
    st.write("Visualisation 3D de 5 documents simples (doc-term matrix) et de leurs similarités géométriques.")

    raw_vectors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0.5, 0.5, 0.5],
        ]
    )
    labels = ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"]
    df = pd.DataFrame(raw_vectors, columns=["x", "y", "z"], index=labels)
    st.dataframe(df, width="stretch")

    try:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=6, color="orange"),
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            height=420,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.error(f"Plotly non disponible ({exc})")


def render_extra_nlp_cosine():
    st.subheader("NLP · Cosine Similarity")
    st.write("Choisissez l’angle entre deux vecteurs pour voir la similarité (cos θ).")

    angle = st.slider("Angle (degrés)", 0, 180, 45, 5, key="nlp_cos_angle")
    theta = math.radians(angle)
    cosine = math.cos(theta)
    st.metric("cos(θ)", f"{cosine:.3f}")

    v1 = np.array([1, 0])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    v2 = rot @ v1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1, color="tab:blue", label="v1")
    ax.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1, color="tab:red", label="v2")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="RL & Finance Lab", layout="wide")
    st.title("RL & Quant Finance Lab")
    st.caption("Onglets restructurés : Agents RL / Pricing & Hedging / Signals / Strategies / NLP Sentiment / Notebooks.")

    tabs_main = st.tabs(
        ["Agents RL", "Pricing & Hedging", "Signals", "Strategies", "NLP Sentiment", "Notebooks"]
    )

    with tabs_main[0]:
        rl_tabs = st.tabs(["Trading DQL", "Hedging DQL", "Allocation"])
        with rl_tabs[0]:
            render_rl_trading()
        with rl_tabs[1]:
            render_rl_hedging()
        with rl_tabs[2]:
            render_rl_allocation()

    with tabs_main[1]:
        ph_tabs = st.tabs(["Pricing", "Delta Hedging"])
        with ph_tabs[0]:
            render_pricing_derivatives()
        with ph_tabs[1]:
            render_hedging_derivatives()

    with tabs_main[2]:
        sig_tabs = st.tabs(
            [
                "Stock Prediction",
                "BTC PCA",
                "Eigen Portfolio",
                "Yield Curve Build",
                "Yield Curve Predict",
            ]
        )
        with sig_tabs[0]:
            render_signals_stock_prediction()
        with sig_tabs[1]:
            render_signals_btc_pca()
        with sig_tabs[2]:
            render_signals_eigen_portfolio()
        with sig_tabs[3]:
            render_signals_yield_curve_construction()
        with sig_tabs[4]:
            render_signals_yield_curve_prediction()

    with tabs_main[3]:
        strat_tabs = st.tabs(["BTC MA", "Crypto Allocation", "RL SP500"])
        with strat_tabs[0]:
            render_strategies_btc_ma()
        with strat_tabs[1]:
            render_strategies_crypto_allocation()
        with strat_tabs[2]:
            render_strategies_rl_sp500()

    with tabs_main[4]:
        render_strategies_nlp_sentiment()

    with tabs_main[5]:
        nb_tabs = st.tabs(["NLP Overview", "Wordcloud", "Doc Vectors", "Cosine Sim"])
        with nb_tabs[0]:
            render_extra_nlp_overview()
        with nb_tabs[1]:
            render_extra_nlp_wordcloud()
        with nb_tabs[2]:
            render_extra_nlp_vectors()
        with nb_tabs[3]:
            render_extra_nlp_cosine()


if __name__ == "__main__":
    main()

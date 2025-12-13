from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class KalmanResult:
    x_filt: np.ndarray
    P_filt: np.ndarray
    x_pred: np.ndarray
    P_pred: np.ndarray
    loglik: float


def kalman_filter(
    *,
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> KalmanResult:
    """
    Standard linear Gaussian Kalman filter.
    y: shape (T, m)
    x: shape (n,)
    """
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)
    P = np.asarray(P0, dtype=float)

    T = int(y.shape[0])
    n = int(x.shape[0])
    x_pred = np.zeros((T, n), dtype=float)
    P_pred = np.zeros((T, n, n), dtype=float)
    x_filt = np.zeros((T, n), dtype=float)
    P_filt = np.zeros((T, n, n), dtype=float)

    loglik = 0.0
    I = np.eye(n, dtype=float)

    for t in range(T):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        x_pred[t, :] = x
        P_pred[t, :, :] = P

        yt = y[t, :]
        mask = np.isfinite(yt)
        if not np.any(mask):
            x_filt[t, :] = x
            P_filt[t, :, :] = P
            continue

        Ht = H[mask, :]
        Rt = R[np.ix_(mask, mask)]
        yt_obs = yt[mask]

        # innovation
        v = yt_obs - Ht @ x
        S = Ht @ P @ Ht.T + Rt
        S = 0.5 * (S + S.T)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = P @ Ht.T @ Sinv

        x = x + K @ v
        P = (I - K @ Ht) @ P @ (I - K @ Ht).T + K @ Rt @ K.T  # Joseph form

        x_filt[t, :] = x
        P_filt[t, :, :] = P

        # log-likelihood (Gaussian)
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                raise np.linalg.LinAlgError
            quad = float(v.T @ Sinv @ v)
            loglik += -0.5 * (len(yt_obs) * np.log(2.0 * np.pi) + logdet + quad)
        except Exception:
            pass

    return KalmanResult(x_filt=x_filt, P_filt=P_filt, x_pred=x_pred, P_pred=P_pred, loglik=float(loglik))


def rts_smoother(
    *,
    F: np.ndarray,
    filt: KalmanResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel smoother for linear systems.
    Returns (x_smooth, P_smooth).
    """
    F = np.asarray(F, dtype=float)
    x_f = np.asarray(filt.x_filt, dtype=float)
    P_f = np.asarray(filt.P_filt, dtype=float)
    x_p = np.asarray(filt.x_pred, dtype=float)
    P_p = np.asarray(filt.P_pred, dtype=float)

    T = int(x_f.shape[0])
    n = int(x_f.shape[1])
    x_s = np.zeros((T, n), dtype=float)
    P_s = np.zeros((T, n, n), dtype=float)

    x_s[-1, :] = x_f[-1, :]
    P_s[-1, :, :] = P_f[-1, :, :]

    for t in range(T - 2, -1, -1):
        try:
            Pinv = np.linalg.inv(P_p[t + 1, :, :])
        except np.linalg.LinAlgError:
            Pinv = np.linalg.pinv(P_p[t + 1, :, :])
        C = P_f[t, :, :] @ F.T @ Pinv
        x_s[t, :] = x_f[t, :] + C @ (x_s[t + 1, :] - x_p[t + 1, :])
        P_s[t, :, :] = P_f[t, :, :] + C @ (P_s[t + 1, :, :] - P_p[t + 1, :, :]) @ C.T
        P_s[t, :, :] = 0.5 * (P_s[t, :, :] + P_s[t, :, :].T)

    return x_s, P_s


def smooth_parameters_random_walk(
    *,
    y: np.ndarray,
    q: float = 1e-4,
    r: float = 1e-2,
) -> Dict[str, np.ndarray]:
    """
    Convenience smoother for parameter sequences using a random-walk state model.
      x_{t+1} = x_t + w, w~N(0, q I)
      y_t = x_t + v, v~N(0, r I)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    T, n = y.shape
    F = np.eye(n, dtype=float)
    H = np.eye(n, dtype=float)
    Q = float(q) * np.eye(n, dtype=float)
    R = float(r) * np.eye(n, dtype=float)
    x0 = np.where(np.isfinite(y[0, :]), y[0, :], 0.0)
    P0 = np.eye(n, dtype=float)

    filt = kalman_filter(y=y, F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
    x_s, P_s = rts_smoother(F=F, filt=filt)
    return {"x_filt": filt.x_filt, "x_smooth": x_s, "loglik": np.array([filt.loglik], dtype=float)}


__all__ = ["KalmanResult", "kalman_filter", "rts_smoother", "smooth_parameters_random_walk"]


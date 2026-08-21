"""
Exact simulation of standard fractional Gaussian noise (fGn) and fractional
Brownian motion (fBm) via the Davies-Harte circulant-embedding method.

CONVENTION - WHICH FRACTIONAL PROCESS THIS MODULE PRODUCES (recorded decision)
=============================================================================
Standard (Mandelbrot-Van Ness) fBm ``B^H`` and the Riemann-Liouville Volterra
driver used by rough Bergomi

    W~_t = sqrt(2H) * int_0^t (t - s)^{H - 1/2} dB_s

are DIFFERENT Gaussian processes.  They share the marginal variance

    Var(B^H_t) = Var(W~_t) = t^{2H}

but they do NOT share the covariance:

    Cov(B^H_s, B^H_t) = 0.5 * (s^{2H} + t^{2H} - |t - s|^{2H})
        -> stationary increments, hence the fGn autocovariance below;

    Cov(W~_s, W~_t)   = 2H * int_0^{min(s,t)} (s-u)^{H-1/2} (t-u)^{H-1/2} du
        -> NOT the expression above for s != t; W~ has non-stationary
           increments (its kernel is truncated at 0, not at -infinity).

rBergomi - and the simulator already shipped in this package,
``app/model/volatility_models/rbergomi/simulator.py`` - is driven by ``W~``,
never by ``B^H``.  Therefore:

  * Davies-Harte (this module) powers the **fBm utility, the statistical
    validation of the fractional machinery, and the roughness diagnostics**
    (e.g. estimating H from a realised-variance series).  It is exact, cheap
    and stationary, which is precisely what such an estimator needs.
  * The **pricing path** (built in a later phase, NOT by this module) uses an
    exact joint Gaussian construction of ``(W~, W)`` so that the spot/vol
    correlation ``rho`` is well defined path by path.

Do NOT feed the output of this module into an rBergomi pricer as if it were
``W~``: the two processes agree in law only through their marginal variance.

ALGORITHM
=========
Target: ``n`` samples of *standard* fGn ``X_0, ..., X_{n-1}`` (unit time step),
a stationary centred Gaussian sequence with autocovariance

    gamma(j) = 0.5 * (|j+1|^{2H} + |j-1|^{2H} - 2 |j|^{2H}),   gamma(0) = 1.

1. Circulant embedding.  A circulant ``C`` of size ``m`` with first row ``c``
   satisfies ``C[i, j] = c[(j - i) mod m]``.  Its leading ``n x n`` block equals
   the fGn Toeplitz covariance iff ``c[k] = gamma(k)`` for ``k <= n-1`` *and*
   ``c[m-k] = gamma(k)`` for ``1 <= k <= n-1``.  Both constraints are met by the
   reflected extension

       c[k] = gamma(min(k, m - k)),

   which needs ``m >= 2 * (n - 1)``.  We take ``m = 2 * (n - 1)`` (the minimal
   embedding, ``padding="minimal"``) or the next power of two above it
   (``padding="power_of_two"``, the default): the radix-2 FFT is markedly
   faster than the mixed-radix / Bluestein path numpy must use when ``2*(n-1)``
   carries a large prime factor, and the reflection keeps the first ``n`` lags
   EXACT in either case - padding only extends the autocovariance to lags up to
   ``m/2``, it never truncates nor approximates the lags we actually need.

2. Eigenvalues.  ``lambda = FFT(c)``; ``c`` is symmetric so ``lambda`` is real
   up to round-off and ``lambda[k] = lambda[m-k]``.  ``C`` is a valid covariance
   matrix iff ``lambda >= 0``.  This is validated (see TOLERANCE below).

3. Spectral synthesis.  Let ``w = exp(2*pi*i/m)`` and draw a Hermitian complex
   sequence ``Z`` with ``Z[m-k] = conj(Z[k])``, ``E|Z_k|^2 = 1`` and
   ``E[Z_k conj(Z_l)] = delta_{kl}``.  Then

       Y_j = (1/sqrt(m)) * sum_k sqrt(lambda_k) Z_k w^{jk}

   is real (the summand is Hermitian) and

       E[Y_j Y_j'] = (1/m) sum_k lambda_k w^{(j-j')k} = c[(j-j') mod m]

   by inverse DFT of ``lambda`` - i.e. exactly ``gamma(|j-j'|)`` whenever
   ``|j-j'| <= n-1``.  So ``X = Y[0:n]`` is exact fGn, not an approximation.

   Building ``Z`` from i.i.d. standard normals ``V``:
       k = 0 and k = m/2 are self-conjugate -> ``Z_k = V`` must be REAL;
       0 < k < m/2                          -> ``Z_k = (V' + i V'') / sqrt(2)``
                                               so that ``E|Z_k|^2 = 1``.
   Folding the ``1/sqrt(m)`` in, the coefficient actually used is

       W_k = sqrt(lambda_k / m)       * V             (k = 0, k = m/2)
       W_k = sqrt(lambda_k / (2 * m)) * (V' + i V'')  (0 < k < m/2)

   which is where the ``sqrt(lambda_k / (2m))`` scaling comes from: the ``2``
   is the variance split between the real and the imaginary part of a
   unit-modulus complex Gaussian, the ``m`` is the DFT normalisation.  Exactly
   ``m`` standard normals are consumed per path: 1 + 1 + 2*(m/2 - 1).

   ``Y = m * irfft(W[0:m/2+1], n=m)`` evaluates ``sum_k W_k w^{+jk}`` directly
   from the Hermitian half-spectrum: no mirrored half to materialise, real
   output by construction, half the memory of a full complex FFT.

4. fBm.  fGn is the increment process of standard fBm at unit step.  On a grid
   ``t_k = k*dt`` with ``dt = T/n``, self-similarity ``B^H_{ct} =d= c^H B^H_t``
   gives increments ``dt^H * X_i``, hence

       B_k = dt^H * sum_{i<k} X_i,   B_0 = 0,   Var(B_k) = (k*dt)^{2H},

   and in particular ``Var(B_n) = T^{2H}``.

TOLERANCE AND FAILURE POLICY
============================
``min(lambda)`` is compared against ``-eig_rtol * max(|lambda|)`` (a relative
tolerance, default 1e-10).  Eigenvalues inside that band are round-off: they
are clamped to 0 and the clamp is REPORTED in
``FgnDiagnostics.n_clamped_eigenvalues`` - never silently.  A genuinely
negative eigenvalue raises :class:`CirculantEmbeddingError`; with
``method="auto"`` the sampler then falls back to an exact Cholesky
factorisation of the ``n x n`` fGn covariance and records the rejection in
``FgnDiagnostics.fallback_reason``.  For fGn with H in (0, 1) the reflected
embedding is non-negative definite (Davies & Harte 1987; Craigmile 2003), so
the fallback is a guard rather than a routine path - but it is reachable, and
can be forced with ``method="cholesky"``.

REPRODUCIBILITY
===============
Pass ``rng=`` (a ``numpy.random.Generator``) or ``seed=`` (an int), never both.
The same seed with the same arguments gives bit-identical output.  Changing
``n``, ``padding`` or ``method`` changes how many normals are drawn and in
which order, hence changes the sample - by design, not by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FgnMethod = Literal["auto", "davies_harte", "cholesky"]
FgnPadding = Literal["minimal", "power_of_two"]

#: Default relative tolerance of the circulant eigenvalue non-negativity test.
DEFAULT_EIG_RTOL: float = 1e-10


class FractionalNoiseError(RuntimeError):
    """Base class for fGn / fBm simulation failures."""


class CirculantEmbeddingError(FractionalNoiseError):
    """The circulant embedding is not non-negative definite (invalid spectrum)."""


class CholeskyFallbackError(FractionalNoiseError):
    """The exact Cholesky fallback failed (covariance not positive definite)."""


@dataclass(frozen=True)
class FgnDiagnostics:
    """
    Diagnostics for one fGn draw.

    Attributes
    ----------
    method:
        Sampler that actually produced the numbers: ``"davies_harte"`` or
        ``"cholesky"``.
    n, H, n_paths:
        Echo of the request (samples per path, Hurst index, path count).
    circulant_size:
        ``m``, the circulant embedding size that was built. ``0`` when the
        caller forced ``method="cholesky"`` and no embedding was attempted.
    min_eigenvalue, max_eigenvalue:
        Extremes of the circulant spectrum BEFORE clamping. ``nan`` when the
        Cholesky sampler produced the output.
    eigenvalue_tolerance:
        The absolute threshold used, ``eig_rtol * max(|lambda|)``. Eigenvalues
        in ``[-tolerance, 0)`` are round-off and were clamped to 0.
    n_clamped_eigenvalues:
        How many eigenvalues were clamped. ``0`` on a clean spectrum; a
        non-zero value is a warning sign, not an error.
    fallback_reason:
        ``None`` unless the Davies-Harte spectrum was rejected and the Cholesky
        fallback took over, in which case it carries the rejection message.
    """

    method: Literal["davies_harte", "cholesky"]
    n: int
    H: float
    n_paths: int
    circulant_size: int
    min_eigenvalue: float
    max_eigenvalue: float
    eigenvalue_tolerance: float
    n_clamped_eigenvalues: int
    fallback_reason: str | None


# ---------------------------------------------------------------------------
# Autocovariance
# ---------------------------------------------------------------------------
def _autocovariance(n_lags: int, H: float) -> np.ndarray:
    """
    fGn autocovariance ``gamma(j)``, j = 0 .. n_lags-1 (unit step, unit variance).

    Positional on purpose: this is the single injection point shared by both
    samplers, so a test can monkeypatch it to feed a non-embeddable sequence.

    The naive form ``0.5*((j+1)^{2H} + (j-1)^{2H} - 2 j^{2H})`` cancels
    catastrophically at large ``j`` (three O(j^{2H}) terms summing to
    O(j^{2H-2})).  We factor ``j^{2H}`` out and evaluate the bracket with
    ``expm1`` / ``log1p``:

        gamma(j) = 0.5 * j^{2H} * [expm1(2H*log1p(1/j)) + expm1(2H*log1p(-1/j))]

    which turns an O(eps * j^2 / (H|2H-1|)) relative error into
    O(eps * j / |2H-1|).  At H = 1/2 fGn IS white noise and that case is
    returned exactly (the general branch would leak ~1e-17 into the tail lags).
    """
    if n_lags <= 0:
        raise ValueError("n_lags must be >= 1.")
    gamma = np.zeros(int(n_lags), dtype=float)
    gamma[0] = 1.0
    if n_lags == 1 or H == 0.5:
        return gamma
    two_h = 2.0 * float(H)
    j = np.arange(1, n_lags, dtype=float)
    x = 1.0 / j
    with np.errstate(divide="ignore"):
        # log1p(-1) = -inf at j = 1 -> expm1(-inf) = -1, which is exactly the
        # |j-1|^{2H} = 0 term. The limit is exact; only the warning is spurious.
        bracket = np.expm1(two_h * np.log1p(x)) + np.expm1(two_h * np.log1p(-x))
    gamma[1:] = 0.5 * (j**two_h) * bracket
    return gamma


def fgn_autocovariance(*, n_lags: int, H: float) -> np.ndarray:
    """
    Autocovariance of *standard* fGn at lags 0 .. ``n_lags`` - 1.

    ``gamma(j) = 0.5 * (|j+1|^{2H} + |j-1|^{2H} - 2|j|^{2H})``, ``gamma(0) = 1``.
    """
    _validate_hurst(H)
    if int(n_lags) <= 0:
        raise ValueError(f"n_lags must be >= 1; got {n_lags!r}.")
    return _autocovariance(int(n_lags), float(H))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_hurst(H: float) -> None:
    H = float(H)
    if not np.isfinite(H) or not (0.0 < H < 1.0):
        raise ValueError(f"H must lie strictly inside (0, 1); got {H!r}.")


def _validate_common(*, n: int, H: float, n_paths: int, eig_rtol: float) -> None:
    _validate_hurst(H)
    if int(n) < 1:
        raise ValueError(f"n must be >= 1; got {n!r}.")
    if int(n_paths) < 1:
        raise ValueError(f"n_paths must be >= 1; got {n_paths!r}.")
    if not np.isfinite(eig_rtol) or float(eig_rtol) < 0.0:
        raise ValueError(
            f"eig_rtol must be a finite, non-negative float; got {eig_rtol!r}."
        )


def _resolve_rng(
    rng: np.random.Generator | None, seed: int | None
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Provide either `rng` or `seed`, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _circulant_size(n: int, padding: FgnPadding) -> int:
    """Embedding size ``m``: minimal ``2*(n-1)`` (>= 2), or the next power of two."""
    minimal = max(2 * (int(n) - 1), 2)
    if padding == "minimal":
        return minimal
    if padding == "power_of_two":
        return int(1 << int(np.ceil(np.log2(minimal))))
    raise ValueError(f"padding must be 'minimal' or 'power_of_two'; got {padding!r}.")


# ---------------------------------------------------------------------------
# Davies-Harte core
# ---------------------------------------------------------------------------
def _circulant_eigenvalues(
    *, n: int, H: float, m: int, eig_rtol: float
) -> tuple[np.ndarray, float, float, float, int]:
    """
    Spectrum of the reflected circulant embedding.

    Returns ``(lambda_clamped, min_eig, max_eig, tolerance, n_clamped)`` and
    raises :class:`CirculantEmbeddingError` when an eigenvalue is negative by
    more than the relative tolerance.
    """
    gamma = _autocovariance(m // 2 + 1, H)
    k = np.arange(m)
    first_row = gamma[np.minimum(k, m - k)]
    # `first_row` is symmetric, so its DFT is real up to round-off.
    eig = np.fft.fft(first_row).real
    max_abs = float(np.max(np.abs(eig)))
    tolerance = float(eig_rtol) * max_abs
    min_eig = float(np.min(eig))
    if min_eig < -tolerance:
        raise CirculantEmbeddingError(
            "Davies-Harte circulant embedding is not non-negative definite for "
            f"H={H!r}, n={n}, m={m}: min eigenvalue {min_eig:.6e} < "
            f"-{tolerance:.6e} (relative tolerance {eig_rtol:.3e})."
        )
    n_clamped = int(np.count_nonzero(eig < 0.0))
    return np.maximum(eig, 0.0), min_eig, float(np.max(eig)), tolerance, n_clamped


def _davies_harte_sample(
    *, eig: np.ndarray, m: int, n: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Spectral synthesis; see step 3 of the module docstring for the derivation."""
    half = m // 2
    # Exactly `m` standard normals per path: 1 (DC) + 1 (Nyquist) + 2*(half-1).
    v = rng.standard_normal((n_paths, m))
    coeff = np.zeros((n_paths, half + 1), dtype=np.complex128)
    coeff[:, 0] = np.sqrt(eig[0] / m) * v[:, 0]
    coeff[:, half] = np.sqrt(eig[half] / m) * v[:, half]
    if half > 1:
        scale = np.sqrt(eig[1:half] / (2.0 * m))
        coeff[:, 1:half] = scale * (v[:, 1:half] + 1j * v[:, half + 1 : m])
    # m * irfft(.) == sum_k coeff_k * exp(+2*pi*i*j*k/m), real by Hermitian symmetry.
    series = m * np.fft.irfft(coeff, n=m, axis=1)
    return np.ascontiguousarray(series[:, :n])


def _cholesky_sample(
    *, n: int, H: float, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Exact fallback: Cholesky factorisation of the ``n x n`` fGn Toeplitz covariance.

    ``X = Z @ L.T`` with ``L L.T = Gamma`` gives ``Cov(X) = Gamma`` exactly.
    Cost is O(n^3) time / O(n^2) memory - acceptable as a guard, useless as a
    default.
    """
    gamma = _autocovariance(n, H)
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    cov = gamma[idx]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise CholeskyFallbackError(
            f"fGn covariance matrix is not positive definite for H={H!r}, n={n}: {exc}"
        ) from exc
    z = rng.standard_normal((n_paths, n))
    return z @ chol.T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fgn_with_diagnostics(
    *,
    n: int,
    H: float,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    method: FgnMethod = "auto",
    padding: FgnPadding = "power_of_two",
    eig_rtol: float = DEFAULT_EIG_RTOL,
) -> tuple[np.ndarray, FgnDiagnostics]:
    """
    Draw standard fGn and report how it was produced.

    Parameters
    ----------
    n:
        Samples per path (unit time step, unit variance).
    H:
        Hurst index, strictly inside (0, 1).
    n_paths:
        Number of independent paths; the sampler is fully vectorised over them.
    rng, seed:
        At most one of them; neither means a fresh, non-reproducible generator.
    method:
        ``"auto"``          - Davies-Harte, falling back to Cholesky if the
                              spectrum is rejected (the default);
        ``"davies_harte"``  - Davies-Harte only, raise
                              :class:`CirculantEmbeddingError` on rejection;
        ``"cholesky"``      - force the exact O(n^3) fallback.
    padding:
        Circulant size: ``"minimal"`` (2*(n-1)) or ``"power_of_two"``.
    eig_rtol:
        Relative tolerance of the eigenvalue non-negativity test.

    Returns
    -------
    (fgn, diagnostics):
        ``fgn`` has shape ``(n_paths, n)``; ``diagnostics`` is an
        :class:`FgnDiagnostics` (sampler used, spectrum extremes, clamp count,
        fallback reason).
    """
    _validate_common(n=n, H=H, n_paths=n_paths, eig_rtol=eig_rtol)
    if method not in ("auto", "davies_harte", "cholesky"):
        raise ValueError(
            f"method must be 'auto', 'davies_harte' or 'cholesky'; got {method!r}."
        )
    n = int(n)
    n_paths = int(n_paths)
    H = float(H)
    generator = _resolve_rng(rng, seed)

    if method == "cholesky":
        sample = _cholesky_sample(n=n, H=H, n_paths=n_paths, rng=generator)
        return sample, FgnDiagnostics(
            method="cholesky",
            n=n,
            H=H,
            n_paths=n_paths,
            circulant_size=0,
            min_eigenvalue=float("nan"),
            max_eigenvalue=float("nan"),
            eigenvalue_tolerance=float("nan"),
            n_clamped_eigenvalues=0,
            fallback_reason=None,
        )

    m = _circulant_size(n, padding)
    try:
        eig, min_eig, max_eig, tolerance, n_clamped = _circulant_eigenvalues(
            n=n, H=H, m=m, eig_rtol=eig_rtol
        )
    except CirculantEmbeddingError as exc:
        if method == "davies_harte":
            raise
        # No randomness has been consumed yet (the spectrum is data-independent),
        # so the fallback draw is unaffected by the rejected attempt and the seed
        # still determines the output entirely.
        sample = _cholesky_sample(n=n, H=H, n_paths=n_paths, rng=generator)
        return sample, FgnDiagnostics(
            method="cholesky",
            n=n,
            H=H,
            n_paths=n_paths,
            circulant_size=m,
            min_eigenvalue=float("nan"),
            max_eigenvalue=float("nan"),
            eigenvalue_tolerance=float("nan"),
            n_clamped_eigenvalues=0,
            fallback_reason=str(exc),
        )

    sample = _davies_harte_sample(eig=eig, m=m, n=n, n_paths=n_paths, rng=generator)
    return sample, FgnDiagnostics(
        method="davies_harte",
        n=n,
        H=H,
        n_paths=n_paths,
        circulant_size=m,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        eigenvalue_tolerance=tolerance,
        n_clamped_eigenvalues=n_clamped,
        fallback_reason=None,
    )


def fgn_davies_harte(
    *,
    n: int,
    H: float,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    method: FgnMethod = "auto",
    padding: FgnPadding = "power_of_two",
    eig_rtol: float = DEFAULT_EIG_RTOL,
) -> np.ndarray:
    """
    Standard fractional Gaussian noise, shape ``(n_paths, n)``.

    Unit time step, ``Var(X_i) = 1``, ``Cov(X_i, X_{i+j}) = gamma(j)``.  See
    :func:`fgn_with_diagnostics` for the arguments and for the metadata return
    (sampler used, spectrum, clamped eigenvalues, fallback reason).
    """
    sample, _ = fgn_with_diagnostics(
        n=n,
        H=H,
        n_paths=n_paths,
        rng=rng,
        seed=seed,
        method=method,
        padding=padding,
        eig_rtol=eig_rtol,
    )
    return sample


def _fbm_from_fgn(fgn: np.ndarray, *, T: float, H: float) -> np.ndarray:
    """Cumulate unit-step fGn into fBm on [0, T]; increments scale as ``dt^H``."""
    n_paths, n = fgn.shape
    dt = float(T) / float(n)
    out = np.zeros((n_paths, n + 1), dtype=float)
    out[:, 1:] = np.cumsum((dt ** float(H)) * fgn, axis=1)
    return out


def fbm_with_diagnostics(
    *,
    n: int,
    H: float,
    T: float = 1.0,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    method: FgnMethod = "auto",
    padding: FgnPadding = "power_of_two",
    eig_rtol: float = DEFAULT_EIG_RTOL,
) -> tuple[np.ndarray, FgnDiagnostics]:
    """
    Draw standard fBm on ``[0, T]`` and report how its driving fGn was produced.

    Returns ``(fbm, diagnostics)`` with ``fbm`` of shape ``(n_paths, n + 1)``,
    ``fbm[:, 0] == 0`` and ``Var(fbm[:, k]) = (k * T / n)^{2H}``.
    """
    if not np.isfinite(T) or float(T) <= 0.0:
        raise ValueError(f"T must be a finite, strictly positive float; got {T!r}.")
    sample, diagnostics = fgn_with_diagnostics(
        n=n,
        H=H,
        n_paths=n_paths,
        rng=rng,
        seed=seed,
        method=method,
        padding=padding,
        eig_rtol=eig_rtol,
    )
    return _fbm_from_fgn(sample, T=float(T), H=float(H)), diagnostics


def fbm_davies_harte(
    *,
    n: int,
    H: float,
    T: float = 1.0,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    method: FgnMethod = "auto",
    padding: FgnPadding = "power_of_two",
    eig_rtol: float = DEFAULT_EIG_RTOL,
) -> np.ndarray:
    """
    Standard (Mandelbrot-Van Ness) fBm on ``[0, T]``, shape ``(n_paths, n + 1)``.

    Column ``k`` holds ``B^H`` at ``t_k = k * T / n``; column 0 is exactly 0 and
    ``Var(B^H_T) = T^{2H}``.  This is NOT the rBergomi Volterra driver ``W~`` -
    see the module docstring.
    """
    path, _ = fbm_with_diagnostics(
        n=n,
        H=H,
        T=T,
        n_paths=n_paths,
        rng=rng,
        seed=seed,
        method=method,
        padding=padding,
        eig_rtol=eig_rtol,
    )
    return path


__all__ = [
    "DEFAULT_EIG_RTOL",
    "CholeskyFallbackError",
    "CirculantEmbeddingError",
    "FgnDiagnostics",
    "FgnMethod",
    "FgnPadding",
    "FractionalNoiseError",
    "fbm_davies_harte",
    "fbm_with_diagnostics",
    "fgn_autocovariance",
    "fgn_davies_harte",
    "fgn_with_diagnostics",
]

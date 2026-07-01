"""
Provable Dual Attack on Kyber — complexity estimator for three schemes
(PS24 / QX25 / LaMS), parallelized search for optimal parameters.

  * PS24 is NOT searched. It directly uses the parameters provided by the authors.
  * QX25 and LaMS are optimized via a parallel search.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path

import numpy as np

LATTICE_ESTIMATOR_DIR = Path(__file__).resolve().parent / "lattice-estimator"
sys.path.insert(0, str(LATTICE_ESTIMATOR_DIR))

import warnings
from estimator.reduction import cost as estimator_reduction_cost, RC
from estimator.util import log2 as estimator_log2
from estimator.reduction import delta as deltaf
warnings.filterwarnings("ignore")


_DEFAULT_WORKERS = min(96, mp.cpu_count() or 1)


# ──────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────

def _scale_sigma(sigma_e):
    """Scale the raw sigma_e input by sqrt(2*pi), Correct prior mistakes."""
    return sigma_e * math.sqrt(2 * math.pi)


def _sieve(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i in range(2, limit + 1) if s[i]]


_PRIMES = _sieve(200)


def _crt_primes(q):
    """Smallest prime set whose cumulative product first exceeds q.
    For q = 3329 this returns {2, 3, 5, 7, 11, 13}."""
    result, prod = [], 1
    for p in _PRIMES:
        if p >= q:
            break
        prod *= p
        result.append(p)
        if prod > q:
            break
    return result


@lru_cache(maxsize=None)
def hermite_factor(beta):
    """H_beta from lattice-estimator's deltaf(beta), cast to float."""
    return float(deltaf(beta))


BKZ_COST_MODEL_NAME = "MATZOV"
BKZ_COST_COMPONENT = "rop"


@lru_cache(maxsize=None)
def _get_bkz_cost_model(name):
    """Return a (cached) lattice-estimator reduction cost model.
    MATZOV uses RC.MATZOV.__class__(nn='list_decoding-classical')."""
    if name == "MATZOV":
        return RC.MATZOV.__class__(nn="list_decoding-classical")
    models = {
        "BDGL16": RC.BDGL16,
        "ABLR21": RC.ABLR21,
        "ADPS16": RC.ADPS16,
        "GJ21": RC.GJ21,
        "Kyber": RC.Kyber,
    }
    if name not in models:
        raise ValueError(f"Unknown BKZ cost model: {name}")
    return models[name]


@lru_cache(maxsize=None)
def _log2_bkz_cached(beta, d, model_name, component):
    """Per-process lru_cache of log2 BKZ cost."""
    if beta < 2 or d < 2 or beta > d:
        return float("inf")
    model = _get_bkz_cost_model(model_name)
    c = estimator_reduction_cost(model, beta, d)
    return float(estimator_log2(c[component]))


def log2_bkz(beta, d, q=None, model_name=BKZ_COST_MODEL_NAME):
    return _log2_bkz_cached(int(beta), int(d), model_name, BKZ_COST_COMPONENT)


def log2_inv_delta_exact(b1, qs, Hb, m):
    """
    log2(1/Delta) = sum_{i=1}^{m} log2( rho_{s_i}(Z) ),

        s_i   = C_arc * H_beta^{-2(i-1)},   C_arc = b1 / (qs * H_beta)
    """
    if m < 1:
        return 0.0

    C_arc = b1 / (qs * Hb)
    if not (C_arc > 0.0) or Hb <= 1.0:
        return 0.0

    logHb2 = 2.0 * math.log(Hb)
    idx = np.arange(m, dtype=np.float64)
    s_vals = C_arc * np.exp(-idx * logHb2)

    rho = np.empty_like(s_vals)
    small = s_vals <= 1.0
    large = ~small
    rho[small] = 1.0 + 2.0 * np.exp(-math.pi / (s_vals[small] * s_vals[small]))
    rho[large] = s_vals[large] * (1.0 + 2.0 * np.exp(-math.pi * s_vals[large] * s_vals[large]))

    return float(np.log2(rho).sum())


def _la(a, b):
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a == float("inf") and b == float("inf"):
        return float("inf")
    if a == float("inf"):
        return a
    if b == float("inf"):
        return b
    hi, lo = (a, b) if a >= b else (b, a)
    if hi - lo > 60:
        return hi
    return hi + math.log2(1 + 2 ** (lo - hi))


def _total(*parts):
    c = float("-inf")
    for p in parts:
        c = _la(c, p)
    return c


def _dom(r):
    d = {
        r["log2_T_bkz"]: "BKZ",
        r["log2_T_mcmc"]: "MCMC",
        r["log2_T_guess"]: "GUESS",
    }
    return d[max(d)]


def log2_sum_prime_powers(primes, nguess):
    acc = float("-inf")
    for p in primes:
        acc = _la(acc, nguess * math.log2(p))
    return acc


# ──────────────────────────────────────────────────────────────────────────
# Literature parameter tables
# ──────────────────────────────────────────────────────────────────────────

# Fields: (n, sigma_e_raw, ref_log2_cost, m, nguess, beta)
PS24_TABLE = {
    "Kyber512":  (512,  math.sqrt(1.5), 238, 1023, 19, 746),
    "Kyber768":  (768,  1.0,            347, 1519, 29, 1129),
    "Kyber1024": (1024, 1.0,            478, 2025, 39, 1602),
}

# PS24 literature s values (Method [16] in the table).
PS24_FIXED_S = {"Kyber512": 0.090, "Kyber768": 0.110, "Kyber1024": 0.110}

_Q = 3329


# ──────────────────────────────────────────────────────────────────────────
# Build the s search interval (shared by all three methods)
# ──────────────────────────────────────────────────────────────────────────

def _build_s_search(s_geo, s_samp, ndual, m, q, n_points=10):
    """
    Build the s candidate list:
        s_min = max(2 * q^{ndual/m - 1}, s_geo)
        s_max = max(s_samp,              s_geo)
        candidates = [s_min + (s_max - s_min) * i / n_points  for i in range(n_points)]
    """
    s_mins1 = 2.0 * (q ** (ndual / m - 1))
    s_min = max(s_mins1, s_geo)
    s_max = max(s_samp, s_geo)
    if s_max < s_min:
        s_max = s_min
    interval = s_max - s_min
    values_of_s = [s_min + interval * i / n_points for i in range(n_points)]
    return s_min, s_max, values_of_s


# ──────────────────────────────────────────────────────────────────────────
# [PS24] evaluation
# ──────────────────────────────────────────────────────────────────────────

def evaluate_ps24(n, q, sigma_e, m, nguess, beta, fixed_s=None):
    """
    [PS24] evaluation:
        sigma   = sigma_e_raw * sqrt(2*pi)
        ||b1||  = H_beta^m * q^{(n-nguess)/m}
        s_geo   = sqrt(e) / (q^{1-n/m} - 2 sqrt(e) sigma)
        s_samp  = ||b1|| / (2q)
        delta   = (1/100) exp(-m s^2 sigma^2 / 2)
        N       = n * ln(q) / delta^2
        T_BKZ   = cost_bkz(model, beta, m)['rop']
        T_MCMC  = N * (1/Delta)
        T_guess = q^nguess
        T_total = T_BKZ + T_MCMC + T_guess
    """
    sigma_e_raw = sigma_e
    sigma_e = _scale_sigma(sigma_e)

    ndual = n - nguess
    if ndual <= 0 or nguess < 0 or beta < 50 or beta > m:
        return {"cost": float("inf")}

    Hb = hermite_factor(beta)
    if Hb <= 1.0:
        return {"cost": float("inf")}

    log2b1 = m * math.log2(Hb) + ndual / m * math.log2(q)
    b1 = 2 ** log2b1

    denom = q ** (1 - n / m) - 2 * math.sqrt(math.e) * sigma_e
    s_geo = math.sqrt(math.e) / denom if denom > 0 else float("inf")
    s_samp = b1 / (2 * q)

    if fixed_s is not None:
        values_of_s = [fixed_s]
        used_fixed_s = True
        s_min = s_max = fixed_s
    else:
        used_fixed_s = False
        if denom <= 0:
            return {"cost": float("inf")}
        s_min, s_max, values_of_s = _build_s_search(s_geo, s_samp, ndual, m, q)

    log2_T_bkz = log2_bkz(beta, d=m, q=q)

    best = {"cost": float("inf")}
    for s in values_of_s:
        qs = q * s
        log2d = math.log2(0.01) - m * s ** 2 * sigma_e ** 2 / (2 * math.log(2))
        if log2d < -3000:
            continue
        log2N = math.log2(n * math.log(q)) + 2 * abs(log2d)
        alpha = (qs / b1) ** 2
        log2iD = log2_inv_delta_exact(b1, qs, Hb, m)
        log2_T_mcmc = log2N + log2iD
        log2_T_guess = nguess * math.log2(q)
        cost = _total(log2_T_bkz, log2_T_mcmc, log2_T_guess)

        if cost < best["cost"]:
            best = {
                "cost": cost, "m": m, "nguess": nguess, "ndual": ndual, "beta": beta,
                "s": s, "s_geo": s_geo, "s_samp": s_samp,
                "fixed_s": fixed_s, "used_fixed_s": used_fixed_s,
                "sigma_e_raw": sigma_e_raw, "sigma_e": sigma_e,
                "log2_b1": log2b1, "log2_delta": log2d, "log2_N": log2N,
                "log2_inv_Delta": log2iD, "alpha": alpha,
                "log2_T_bkz": log2_T_bkz, "log2_T_mcmc": log2_T_mcmc,
                "log2_T_guess": log2_T_guess,
            }
    return best


# ──────────────────────────────────────────────────────────────────────────
# [QX25] evaluation
# ──────────────────────────────────────────────────────────────────────────

def evaluate_qx25(n, q, sigma_e, m, nguess, beta, primes_crt=None, fixed_s=None):
    """
    [QX25] evaluation. For q = 3329 the CRT primes are {2,3,5,7,11,13}, pk = 13.
        constraints: m > 500, nguess * ln(q) / m <= 0.8
        R       = q^{1-n/m} / (pk^{nguess/m} * sqrt(e))
        Q       = R - sigma
        s_geo   = 2Q / (Q^2 - sigma^2)   (Q > sigma)
        s_samp  = ||b1|| / (2q)
        delta   = (1/10) exp(-m s^2 sigma^2 / 2)
        N       = (m + log2(sum_j p_j^nguess)) / delta^2
        T_guess = pk^nguess
    """
    sigma_e_raw = sigma_e
    sigma_e = _scale_sigma(sigma_e)

    ndual = n - nguess
    if ndual <= 0 or nguess < 0 or beta < 50 or beta > m:
        return {"cost": float("inf")}
    if m <= 500 or nguess * math.log(q) / m > 0.8:
        return {"cost": float("inf")}

    if primes_crt is None:
        primes_crt = _crt_primes(q)
    pk = primes_crt[-1]

    Hb = hermite_factor(beta)
    if Hb <= 1.0:
        return {"cost": float("inf")}

    log2b1 = m * math.log2(Hb) + ndual / m * math.log2(q)
    b1 = 2 ** log2b1

    R = q ** (1 - n / m) / (pk ** (nguess / m) * math.sqrt(math.e))
    Q = R - sigma_e
    s_geo = 2 * Q / (Q ** 2 - sigma_e ** 2) if Q > sigma_e else float("inf")
    s_samp = b1 / (2 * q)

    log2_sum_pj = log2_sum_prime_powers(primes_crt, nguess)
    log2_T_bkz = log2_bkz(beta, d=m, q=q)

    if fixed_s is not None:
        values_of_s = [fixed_s]
        used_fixed_s = True
        s_min = s_max = fixed_s
    else:
        used_fixed_s = False
        if Q <= sigma_e:
            return {"cost": float("inf")}
        s_min, s_max, values_of_s = _build_s_search(s_geo, s_samp, ndual, m, q)

    best = {"cost": float("inf")}
    for s in values_of_s:
        qs = q * s
        log2d = math.log2(0.1) - m * s ** 2 * sigma_e ** 2 / (2 * math.log(2))
        if log2d < -3000:
            continue
        log2N = math.log2(m + log2_sum_pj) + 2 * abs(log2d)
        alpha = (qs / b1) ** 2
        log2iD = log2_inv_delta_exact(b1, qs, Hb, m)
        log2_T_mcmc = log2N + log2iD
        log2_T_guess = nguess * math.log2(pk)
        cost = _total(log2_T_bkz, log2_T_mcmc, log2_T_guess)

        if cost < best["cost"]:
            best = {
                "cost": cost, "m": m, "nguess": nguess, "ndual": ndual, "beta": beta,
                "s": s, "s_geo": s_geo, "s_samp": s_samp,
                "fixed_s": fixed_s, "used_fixed_s": used_fixed_s,
                "sigma_e_raw": sigma_e_raw, "sigma_e": sigma_e,
                "log2_b1": log2b1, "log2_delta": log2d, "log2_N": log2N,
                "log2_inv_Delta": log2iD, "alpha": alpha,
                "log2_T_bkz": log2_T_bkz, "log2_T_mcmc": log2_T_mcmc,
                "log2_T_guess": log2_T_guess,
                "pk": pk, "k_primes": len(primes_crt), "log2_sum_pj": log2_sum_pj,
            }
    return best


# ──────────────────────────────────────────────────────────────────────────
# [LaMS] evaluation (p-adic with p fixed to 2)
# ──────────────────────────────────────────────────────────────────────────

def evaluate_lams(n, q, sigma_e, m, nguess, beta, fixed_s=None):
    """
    [LaMS] evaluation (p = 2):
        l       = ceil(log2(q))
        constraints: m > 500, nguess * ln(q) / m <= 0.8
        R       = q^{1-n/m} / (2^{nguess/m} * sqrt(e))
        Q       = R - sigma
        s_geo   = 2Q / (Q^2 - sigma^2)
        s_samp  = ||b1|| / (2q)
        delta   = (1/10) exp(-m s^2 sigma^2 / 2)
        N       = (m + log2(l * 2^nguess)) / delta^2
        T_guess = l * 2^nguess
    """
    sigma_e_raw = sigma_e
    sigma_e = _scale_sigma(sigma_e)

    p = 2
    l = math.ceil(math.log(q) / math.log(p))

    ndual = n - nguess
    if ndual <= 0 or nguess < 0 or beta < 50 or beta > m:
        return {"cost": float("inf")}
    if m <= 500 or nguess * math.log(q) / m > 0.8:
        return {"cost": float("inf")}

    Hb = hermite_factor(beta)
    if Hb <= 1.0:
        return {"cost": float("inf")}

    log2b1 = m * math.log2(Hb) + ndual / m * math.log2(q)
    b1 = 2 ** log2b1

    R = q ** (1 - n / m) / (2 ** (nguess / m) * math.sqrt(math.e))
    Q = R - sigma_e
    s_geo = 2 * Q / (Q ** 2 - sigma_e ** 2) if Q > sigma_e else float("inf")
    s_samp = b1 / (2 * q)

    log2_lpng = math.log2(l) + nguess   # log2(l * 2^nguess) = T_guess (log2)
    log2_T_bkz = log2_bkz(beta, d=m, q=q)

    if fixed_s is not None:
        values_of_s = [fixed_s]
        used_fixed_s = True
        s_min = s_max = fixed_s
    else:
        used_fixed_s = False
        if Q <= sigma_e:
            return {"cost": float("inf")}
        s_min, s_max, values_of_s = _build_s_search(s_geo, s_samp, ndual, m, q)

    best = {"cost": float("inf")}
    for s in values_of_s:
        qs = q * s
        log2d = math.log2(0.1) - m * s ** 2 * sigma_e ** 2 / (2 * math.log(2))
        if log2d < -3000:
            continue
        log2N = math.log2(m + log2_lpng) + 2 * abs(log2d)
        alpha = (qs / b1) ** 2
        log2iD = log2_inv_delta_exact(b1, qs, Hb, m)
        log2_T_mcmc = log2N + log2iD
        log2_T_guess = log2_lpng
        cost = _total(log2_T_bkz, log2_T_mcmc, log2_T_guess)

        if cost < best["cost"]:
            best = {
                "cost": cost, "m": m, "nguess": nguess, "ndual": ndual, "beta": beta,
                "s": s, "s_geo": s_geo, "s_samp": s_samp,
                "fixed_s": fixed_s, "used_fixed_s": used_fixed_s,
                "sigma_e_raw": sigma_e_raw, "sigma_e": sigma_e,
                "log2_b1": log2b1, "log2_delta": log2d, "log2_N": log2N,
                "log2_inv_Delta": log2iD, "alpha": alpha,
                "log2_T_bkz": log2_T_bkz, "log2_T_mcmc": log2_T_mcmc,
                "log2_T_guess": log2_T_guess,
                "p": p, "l": l, "log2_lpng": log2_lpng,
            }
    return best


# ──────────────────────────────────────────────────────────────────────────
# Parallel workers: each task fixes (m, beta) and sweeps nguess
# ──────────────────────────────────────────────────────────────────────────

def _eval_block_qx25(args):
    n, q, sigma_e, m, beta, ng_values, primes_crt = args
    best = {"cost": float("inf")}
    for ng in ng_values:
        r = evaluate_qx25(n, q, sigma_e, m, ng, beta, primes_crt)
        if r["cost"] < best["cost"]:
            best = r
    return best


def _eval_block_lams(args):
    n, q, sigma_e, m, beta, ng_values = args
    best = {"cost": float("inf")}
    for ng in ng_values:
        r = evaluate_lams(n, q, sigma_e, m, ng, beta)
        if r["cost"] < best["cost"]:
            best = r
    return best


def _merge_best(best, candidate):
    if candidate.get("cost", float("inf")) < best.get("cost", float("inf")):
        return candidate
    return best


class _PoolCtx:
    def __init__(self, pool, n_workers):
        self._external = pool is not None
        self.pool = pool if pool is not None else mp.Pool(n_workers)

    def __enter__(self):
        return self.pool

    def __exit__(self, *exc):
        if not self._external:
            self.pool.close()
            self.pool.join()
        return False


# ──────────────────────────────────────────────────────────────────────────
# Parallel optimizer: QX25
# ──────────────────────────────────────────────────────────────────────────

def optimize_qx25(
    n, q, sigma_e,
    nguess_max=250,
    n_workers=None,
    pool=None,
    coarse_step_m=4, coarse_step_beta=4,
    fine_radius_m=40, fine_radius_beta=40,
    fine_step_m=1, fine_step_beta=1,
):
    if n_workers is None:
        n_workers = _DEFAULT_WORKERS

    m_lo, m_hi = int(1.4 * n), int(2.1 * n)
    beta_hi_cap = 1800

    pc = _crt_primes(q)
    ln_q = math.log(q)

    m_coarse = set(range(m_lo, m_hi + 1, coarse_step_m))
    beta_coarse = set(range(50, min(m_hi, beta_hi_cap) + 1, coarse_step_beta))

    m_coarse = sorted(m_coarse)
    beta_coarse = sorted(beta_coarse)

    def _ng_values_for_m(m):
        if m <= 500:
            return []
        ng_max = min(nguess_max, int(0.8 * m / ln_q))
        return list(range(0, ng_max + 1))

    tasks = []
    for m in m_coarse:
        ngs = _ng_values_for_m(m)
        if not ngs:
            continue
        for beta in beta_coarse:
            if beta <= m:
                tasks.append((n, q, sigma_e, m, beta, ngs, pc))

    best = {"cost": float("inf")}

    with _PoolCtx(pool, n_workers) as P:
        for r in P.imap_unordered(_eval_block_qx25, tasks, chunksize=8):
            best = _merge_best(best, r)

        if best["cost"] == float("inf"):
            return best

        mc, bc = best["m"], best["beta"]

        m_fine = set(range(
            max(m_lo, mc - fine_radius_m),
            min(m_hi, mc + fine_radius_m) + 1, fine_step_m))
        beta_fine = set(range(
            max(50, bc - fine_radius_beta),
            min(m_hi, beta_hi_cap, bc + fine_radius_beta) + 1, fine_step_beta))

        m_fine = sorted(m_fine)
        beta_fine = sorted(beta_fine)

        tasks_fine = []
        for m in m_fine:
            ngs = _ng_values_for_m(m)
            if not ngs:
                continue
            for beta in beta_fine:
                if beta <= m:
                    tasks_fine.append((n, q, sigma_e, m, beta, ngs, pc))

        for r in P.imap_unordered(_eval_block_qx25, tasks_fine, chunksize=4):
            best = _merge_best(best, r)

    best.update(evaluate_qx25(n, q, sigma_e, best["m"], best["nguess"], best["beta"], pc))
    return best


# ──────────────────────────────────────────────────────────────────────────
# Parallel optimizer: LaMS (no literature table)
# ──────────────────────────────────────────────────────────────────────────

def optimize_lams(
    n, q, sigma_e,
    nguess_max=300,
    n_workers=None,
    pool=None,
    coarse_step_m=4, coarse_step_beta=4,
    fine_radius_m=40, fine_radius_beta=40,
    fine_step_m=1, fine_step_beta=1,
):
    if n_workers is None:
        n_workers = _DEFAULT_WORKERS

    m_lo, m_hi = int(1.4 * n), int(2.1 * n)
    beta_hi_cap = 1800
    ln_q = math.log(q)

    m_coarse = sorted(set(range(m_lo, m_hi + 1, coarse_step_m)))
    beta_coarse = sorted(set(range(50, min(m_hi, beta_hi_cap) + 1, coarse_step_beta)))

    def _ng_values_for_m(m):
        if m <= 500:
            return []
        ng_max = min(nguess_max, int(0.8 * m / ln_q))
        return list(range(0, ng_max + 1))

    tasks = []
    for m in m_coarse:
        ngs = _ng_values_for_m(m)
        if not ngs:
            continue
        for beta in beta_coarse:
            if beta <= m:
                tasks.append((n, q, sigma_e, m, beta, ngs))

    best = {"cost": float("inf")}

    with _PoolCtx(pool, n_workers) as P:
        for r in P.imap_unordered(_eval_block_lams, tasks, chunksize=8):
            best = _merge_best(best, r)

        if best["cost"] == float("inf"):
            return best

        mc, bc = best["m"], best["beta"]

        m_fine = sorted(set(range(
            max(m_lo, mc - fine_radius_m),
            min(m_hi, mc + fine_radius_m) + 1, fine_step_m)))
        beta_fine = sorted(set(range(
            max(50, bc - fine_radius_beta),
            min(m_hi, beta_hi_cap, bc + fine_radius_beta) + 1, fine_step_beta)))

        tasks_fine = []
        for m in m_fine:
            ngs = _ng_values_for_m(m)
            if not ngs:
                continue
            for beta in beta_fine:
                if beta <= m:
                    tasks_fine.append((n, q, sigma_e, m, beta, ngs))

        for r in P.imap_unordered(_eval_block_lams, tasks_fine, chunksize=4):
            best = _merge_best(best, r)

    best.update(evaluate_lams(n, q, sigma_e, best["m"], best["nguess"], best["beta"]))
    return best


def _guess_formula_label(r):
    if "pk" in r:
        return "pk^ng"
    if "p" in r and "l" in r:
        return "l*2^ng"
    return "q^ng"


def _aux_label(r):
    if "pk" in r:
        return f"pk={r['pk']}"
    if "p" in r and "l" in r:
        return f"p={r['p']} l={r['l']}"
    return ""


def _print_optimum(method, r):
    print(f"\n  ---- [{method}] result ----")
    if r is None or r.get("cost", float("inf")) == float("inf"):
        print("      no feasible parameters")
        return
    aux = _aux_label(r) or "-"
    print(f"      m        = {r['m']}")
    print(f"      nguess   = {r['nguess']}")
    print(f"      beta     = {r['beta']}")
    print(f"      s        = {r['s']:.6f}")
    print(f"      log2(N)  = {r['log2_N']:.3f}")
    if method != "PS24":
        print(f"      T_BKZ    = 2^{r['log2_T_bkz']:.3f}")
        print(f"      T_MCMC   = 2^{r['log2_T_mcmc']:.3f}   [N * (1/Delta)]")
        print(f"      T_guess  = 2^{r['log2_T_guess']:.3f}   [{_guess_formula_label(r)}]")
    print(f"      T_total  = 2^{r['cost']:.3f} ")


def _print_summary(results, schemes):
    print(f"\n\n{'=' * 78}")
    print("Optimization summary (best found per method)")
    print(f"{'=' * 78}")

    print(f"{'Scheme':<12} {'Method':<7} {'Estimate':>9}  "
          f"{'m':>5} {'ng':>4} {'beta':>5} {'s':>7}")
    print("-" * 78)

    for name in schemes:
        for mth in ["PS24", "QX25", "LaMS"]:
            r = results[name][mth]

            if r["cost"] == float("inf"):
                print(f"{name:<12} {mth:<7} {'inf':>9}")
                continue

            print(f"{name:<12} {mth:<7} {r['cost']:>9.1f}  "
                  f"{r['m']:>5} {r['nguess']:>4} {r['beta']:>5} {r['s']:>7.3f}")


def _print_comparison(results, schemes):
    print(f"\n\n{'=' * 76}")
    print("Final complexity comparison of the three schemes")
    print(f"{'=' * 76}")
    print(f"{'Scheme':<12}  {'PS24':>8} {'QX25':>8} {'LaMS':>8}  "
          f"{'PS24-QX25':>10} {'PS24-LaMS':>10} {'QX25-LaMS':>10}")
    print("-" * 76)

    for name in schemes:
        c = {m: results[name][m]["cost"] for m in ["PS24", "QX25", "LaMS"]}
        if any(v == float("inf") for v in c.values()):
            print(f"{name:<12}  infeasible result present, skipped")
            continue
        print(f"{name:<12}  {c['PS24']:>8.1f} {c['QX25']:>8.1f} {c['LaMS']:>8.1f}  "
              f"{c['PS24'] - c['QX25']:>10.1f} "
              f"{c['PS24'] - c['LaMS']:>10.1f} "
              f"{c['QX25'] - c['LaMS']:>10.1f}")


def main(n_workers=None):
    if n_workers is None:
        n_workers = _DEFAULT_WORKERS

    q = _Q
    schemes = {
        "Kyber512":  (512,  math.sqrt(1.5)),
        "Kyber768":  (768,  1.0),
        "Kyber1024": (1024, 1.0),
    }

    results = {}

    with mp.Pool(n_workers) as pool:
        for name, (n, se) in schemes.items():
            print(f"\n  [{name}]  n={n}, q={q}, sigma_raw={se:.6f}, "
                  f"sigma_scaled={_scale_sigma(se):.6f}")
            results[name] = {}

            _, _, ref16, m16, ng16, b16 = PS24_TABLE[name]
            s16 = PS24_FIXED_S[name]
            r = evaluate_ps24(n, q, se, m16, ng16, b16, fixed_s=s16)
            r["cost"] = ref16            
            results[name]["PS24"] = r
            _print_optimum("PS24", r)

            # ---- QX25: parallel search ----
            r = optimize_qx25(n, q, se, nguess_max=250, n_workers=n_workers, pool=pool)
            results[name]["QX25"] = r
            _print_optimum("QX25", r)

            # ---- LaMS: parallel search ----
            r = optimize_lams(n, q, se, nguess_max=300, n_workers=n_workers, pool=pool)
            results[name]["LaMS"] = r
            _print_optimum("LaMS", r)

    _print_summary(results, schemes)
    _print_comparison(results, schemes)


if __name__ == "__main__":
    main(n_workers=_DEFAULT_WORKERS)

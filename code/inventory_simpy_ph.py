"""SimPy simulation for a lost-sales (s, S) inventory model with PH demand/lead times.

The main entry-point for one setting is:
    simulate_single_setting(...)
which returns:
    - input_vector: shape (100, 44)
    - inventory_distribution: shape (100, 31)
    - avg_orders_so_far: shape (100,)
    - avg_lost_sales_so_far: shape (100,)
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import simpy

try:
    from scipy.linalg import expm as _scipy_expm
except ImportError:
    _scipy_expm = None

FIXED_S = 4
FIXED_CAP_S = 8
PH_PERCENTILE_PROBS = np.array(
    [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
    dtype=float,
)
N_PH_MOMENTS = 10
N_PH_PERCENTILES = int(PH_PERCENTILE_PROBS.size)
S_INPUT_COL = 20
CAP_S_INPUT_COL = 21
INPUT_FEATURE_COUNT = 2 * N_PH_MOMENTS + 2 + 2 * N_PH_PERCENTILES


@dataclass(frozen=True)
class PHDistribution:
    """Phase-type distribution with transient generator T and initial probs alpha."""

    alpha: np.ndarray  # shape (n,)
    T: np.ndarray  # shape (n, n)
    moments: np.ndarray  # first 10 raw moments
    rates: np.ndarray  # -diag(T)
    jump_cdfs: np.ndarray  # row-wise CDF across n transient states + absorb state

    def sample(self, rng: np.random.Generator) -> float:
        """Sample one PH variate via CTMC transitions until absorption."""
        n = self.alpha.shape[0]
        state = int(rng.choice(n, p=self.alpha))
        elapsed = 0.0

        while True:
            elapsed += rng.exponential(1.0 / self.rates[state])
            nxt = int(np.searchsorted(self.jump_cdfs[state], rng.random(), side="right"))
            if nxt >= n:
                return elapsed
            state = nxt


@dataclass(frozen=True)
class DynamicDemandPlan:
    """Demand-change plan for one replication."""

    change_points: np.ndarray  # shape (k,), values in {1,...,horizon}
    means: np.ndarray  # shape (k+1,), segment means including initial segment
    phs: Tuple[PHDistribution, ...]  # length k+1


@dataclass(frozen=True)
class InputControlRanges:
    """Optional ranges for controlled random setting generation."""

    inter_avg_scv: Optional[Tuple[float, float]] = None
    lead_scv: Optional[Tuple[float, float]] = None
    mean_ratio: Optional[Tuple[float, float]] = None
    s: Optional[Tuple[int, int]] = None
    S: Optional[Tuple[int, int]] = None
    max_tries: int = 1000


@dataclass(frozen=True)
class InputPartition:
    """One CSV-defined coarse input partition."""

    test_set: int
    D: str
    L: str
    rho: str
    S: str
    s: str
    num_files: int


def _compute_ph_moments(alpha: np.ndarray, T: np.ndarray, k_max: int = 10) -> np.ndarray:
    """Compute first k_max raw moments of PH(alpha, T)."""
    n = T.shape[0]
    one = np.ones(n, dtype=float)
    A = -T
    v = one.copy()
    out = np.zeros(k_max, dtype=float)

    for k in range(1, k_max + 1):
        v = np.linalg.solve(A, v)  # v = (-T)^(-k) * 1
        out[k - 1] = factorial(k) * float(alpha @ v)
    return out


def _build_jump_cdfs(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-state jump CDFs and rates for fast PH sampling."""
    n = T.shape[0]
    rates = -np.diag(T).copy()
    jump_cdfs = np.zeros((n, n + 1), dtype=float)

    for i in range(n):
        probs = np.zeros(n + 1, dtype=float)
        for j in range(n):
            if i != j:
                probs[j] = max(T[i, j], 0.0) / rates[i]
        probs[n] = max(0.0, 1.0 - probs[:n].sum())  # absorption
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        jump_cdfs[i] = cdf

    return rates, jump_cdfs


def generate_random_ph(
    size: int,
    target_mean: float,
    rng: np.random.Generator,
) -> PHDistribution:
    """Generate a random PH distribution of given size and target mean.

    The generator uses a general transient CTMC structure (not restricted to
    pure Erlang/hyperexponential forms), with random topology and rate scales.
    """
    if size <= 0:
        raise ValueError("size must be positive.")
    if target_mean <= 0:
        raise ValueError("target_mean must be positive.")

    n = size
    style = int(rng.integers(0, 4))

    if style == 0:
        rate_sigma = 0.20
        absorb_a, absorb_b = 2.5, 8.0
        alpha_conc = 10.0
    elif style == 1:
        rate_sigma = 1.30
        absorb_a, absorb_b = 1.2, 2.0
        alpha_conc = 0.45
    elif style == 2:
        rate_sigma = 0.70
        absorb_a, absorb_b = 1.6, 3.5
        alpha_conc = 1.2
    else:
        rate_sigma = 0.45
        absorb_a, absorb_b = 4.0, 3.0
        alpha_conc = 2.5

    alpha = rng.dirichlet(np.full(n, alpha_conc, dtype=float))
    exit_rates = np.exp(rng.normal(loc=0.0, scale=rate_sigma, size=n))
    T = np.zeros((n, n), dtype=float)

    for i in range(n):
        lam = exit_rates[i]
        absorb_p = float(rng.beta(absorb_a, absorb_b))
        absorb_p = float(np.clip(absorb_p, 0.03, 0.97))
        trans_p = 1.0 - absorb_p

        others = [j for j in range(n) if j != i]
        if others and trans_p > 0:
            mask = rng.random(len(others)) < rng.uniform(0.35, 1.0)
            if not np.any(mask):
                mask[int(rng.integers(0, len(others)))] = True
            active = np.array(others, dtype=int)[mask]
            weights = rng.dirichlet(np.ones(active.size, dtype=float))
            T[i, active] = lam * trans_p * weights

        T[i, i] = -lam

    # Rescale to target mean exactly.
    current_mean = _compute_ph_moments(alpha, T, k_max=1)[0]
    scale = current_mean / target_mean
    T *= scale

    moments = _compute_ph_moments(alpha, T, k_max=10)
    rates, jump_cdfs = _build_jump_cdfs(T)
    return PHDistribution(alpha=alpha, T=T, moments=moments, rates=rates, jump_cdfs=jump_cdfs)


def _ph_from_alpha_T(alpha: np.ndarray, T: np.ndarray, target_mean: float) -> PHDistribution:
    """Build a PHDistribution from (alpha, T) after scaling to target mean."""
    if target_mean <= 0:
        raise ValueError("target_mean must be positive.")
    current_mean = _compute_ph_moments(alpha, T, k_max=1)[0]
    scale = current_mean / target_mean
    T_scaled = T * scale
    moments = _compute_ph_moments(alpha, T_scaled, k_max=10)
    rates, jump_cdfs = _build_jump_cdfs(T_scaled)
    return PHDistribution(alpha=alpha, T=T_scaled, moments=moments, rates=rates, jump_cdfs=jump_cdfs)


def _gen_erlang_like_ph(size: int, target_mean: float) -> PHDistribution:
    alpha = np.zeros(size, dtype=float)
    alpha[0] = 1.0
    T = np.zeros((size, size), dtype=float)
    lam = 1.0
    for i in range(size):
        T[i, i] = -lam
        if i < size - 1:
            T[i, i + 1] = lam
    return _ph_from_alpha_T(alpha=alpha, T=T, target_mean=target_mean)


def _gen_hyperexp_heavy_ph(size: int, target_mean: float, rng: np.random.Generator) -> PHDistribution:
    alpha = rng.dirichlet(np.full(size, 0.12, dtype=float))
    rates = np.exp(rng.uniform(np.log(0.01), np.log(120.0), size=size))

    slow_idx = int(np.argmin(rates))
    p_slow = float(rng.uniform(0.01, 0.18))
    alpha = (1.0 - p_slow) * alpha
    alpha[slow_idx] += p_slow
    alpha /= alpha.sum()

    T = -np.diag(rates)
    return _ph_from_alpha_T(alpha=alpha, T=T, target_mean=target_mean)


def _gen_hyperexp_ultra_ph(size: int, target_mean: float, rng: np.random.Generator) -> PHDistribution:
    alpha = rng.dirichlet(np.full(size, 0.08, dtype=float))
    rates = np.exp(rng.uniform(np.log(0.002), np.log(250.0), size=size))

    slow_idx = int(np.argmin(rates))
    rates[slow_idx] *= float(rng.uniform(0.002, 0.04))

    p_slow = float(rng.uniform(0.002, 0.12))
    alpha = (1.0 - p_slow) * alpha
    alpha[slow_idx] += p_slow
    alpha /= alpha.sum()

    T = -np.diag(rates)
    return _ph_from_alpha_T(alpha=alpha, T=T, target_mean=target_mean)


def _gen_hypererlang_ph(size: int, target_mean: float, rng: np.random.Generator) -> PHDistribution:
    """Generate a Hyper-Erlang PH as a mixture of Erlang branch blocks."""
    if size <= 1:
        return _gen_erlang_like_ph(size=size, target_mean=target_mean)

    max_blocks = min(6, max(2, size // 2 + 1))
    n_blocks = int(rng.integers(2, max_blocks + 1))

    cuts = np.sort(rng.choice(np.arange(1, size), size=n_blocks - 1, replace=False))
    block_lengths = np.diff(np.concatenate(([0], cuts, [size]))).astype(int)
    rng.shuffle(block_lengths)

    alpha = np.zeros(size, dtype=float)
    branch_probs = rng.dirichlet(np.full(n_blocks, 0.45, dtype=float))
    rates = np.exp(rng.uniform(np.log(0.02), np.log(120.0), size=n_blocks))

    slow_idx = int(np.argmin(rates))
    rates[slow_idx] *= float(rng.uniform(0.01, 0.20))
    p_slow = float(rng.uniform(0.01, 0.20))
    branch_probs = (1.0 - p_slow) * branch_probs
    branch_probs[slow_idx] += p_slow
    branch_probs /= branch_probs.sum()

    T = np.zeros((size, size), dtype=float)
    offset = 0
    for block_idx, block_len in enumerate(block_lengths):
        rate = float(rates[block_idx])
        alpha[offset] = branch_probs[block_idx]
        for phase in range(block_len):
            idx = offset + phase
            T[idx, idx] = -rate
            if phase < block_len - 1:
                T[idx, idx + 1] = rate
        offset += block_len

    return _ph_from_alpha_T(alpha=alpha, T=T, target_mean=target_mean)


def _gen_coxian_like_ph(size: int, target_mean: float, rng: np.random.Generator) -> PHDistribution:
    alpha = np.zeros(size, dtype=float)
    alpha[0] = 1.0
    rates = np.exp(rng.uniform(np.log(0.02), np.log(80.0), size=size))

    mode = str(rng.choice(["balanced", "tail-switch"], p=[0.55, 0.45]))
    if size <= 1:
        tail_start = 1
    else:
        low = max(1, size // 4)
        if low >= size:
            low = size - 1
        tail_start = int(rng.integers(low, size))

    T = np.zeros((size, size), dtype=float)
    for i in range(size):
        lam = rates[i]
        if mode == "tail-switch" and i >= tail_start:
            lam *= float(rng.uniform(2e-4, 5e-2))
        if i < size - 1:
            if mode == "balanced":
                cont = float(rng.uniform(0.70, 0.998))
            else:
                if i < tail_start - 1:
                    cont = float(rng.uniform(0.35, 0.92))
                elif i == tail_start - 1:
                    cont = float(rng.uniform(0.003, 0.09))
                else:
                    cont = float(rng.uniform(0.985, 0.9999))
            T[i, i + 1] = lam * cont
        T[i, i] = -lam

    return _ph_from_alpha_T(alpha=alpha, T=T, target_mean=target_mean)


def _gen_coxian_extreme_ph(size: int, target_mean: float, rng: np.random.Generator) -> PHDistribution:
    alpha = np.zeros(size, dtype=float)
    alpha[0] = 1.0
    rates = np.exp(rng.uniform(np.log(0.05), np.log(120.0), size=size))

    if size <= 1:
        tail_start = 1
    else:
        low = max(1, size // 3)
        if low >= size:
            low = size - 1
        tail_start = int(rng.integers(low, size))

    T = np.zeros((size, size), dtype=float)
    for i in range(size):
        lam = rates[i]
        if i >= tail_start:
            lam *= float(rng.uniform(5e-5, 1e-2))

        if i < size - 1:
            if i < tail_start - 1:
                cont = float(rng.uniform(0.45, 0.97))
            elif i == tail_start - 1:
                cont = float(rng.uniform(0.001, 0.04))
            else:
                cont = float(rng.uniform(0.992, 0.99995))
            T[i, i + 1] = lam * cont
        T[i, i] = -lam

    return _ph_from_alpha_T(alpha=alpha, T=T, target_mean=target_mean)


def _scv_from_moments(moments: np.ndarray) -> float:
    """Compute SCV from first two raw moments."""
    m1 = float(moments[0])
    m2 = float(moments[1])
    if m1 <= 0:
        return float("inf")
    var = max(0.0, m2 - m1 * m1)
    return var / (m1 * m1)


def _ph_cdf(ph: PHDistribution, t: float) -> float:
    """Evaluate the PH CDF at t."""
    if t <= 0:
        return 0.0
    one = np.ones(ph.alpha.shape[0], dtype=float)

    if _scipy_expm is not None:
        survival = float(ph.alpha @ _scipy_expm(ph.T * float(t)) @ one)
        if not np.isfinite(survival):
            survival = _ph_survival_uniformization(ph, t)
    else:
        survival = _ph_survival_uniformization(ph, t)

    return float(np.clip(1.0 - survival, 0.0, 1.0))


def _ph_survival_uniformization(
    ph: PHDistribution,
    t: float,
    tol: float = 1e-13,
    max_terms: int = 20000,
) -> float:
    """Evaluate PH survival by uniformizing the transient subgenerator."""
    nu = float(np.max(-np.diag(ph.T)))
    if nu <= 0:
        return 1.0

    x = nu * float(t)
    if x > 700.0:
        return 0.0

    P = np.eye(ph.T.shape[0]) + ph.T / nu
    pk = ph.alpha.copy()
    w = np.exp(-x)
    transient = w * pk

    k_cap = int(min(max_terms, max(50, x + 14.0 * np.sqrt(max(x, 1e-12)) + 40.0)))
    for k in range(1, k_cap + 1):
        pk = pk @ P
        w *= x / k
        transient += w * pk
        if (k > x) and (w < tol):
            break

    return float(np.clip(transient.sum(), 0.0, 1.0))


def ph_percentiles(
    ph: PHDistribution,
    probs: np.ndarray = PH_PERCENTILE_PROBS,
    iterations: int = 60,
) -> np.ndarray:
    """Return PH quantiles for probabilities in probs."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1 or np.any(probs <= 0.0) or np.any(probs >= 1.0):
        raise ValueError("Percentile probabilities must be a 1D array inside (0, 1).")
    if np.any(np.diff(probs) < 0):
        raise ValueError("Percentile probabilities must be sorted in ascending order.")

    mean = max(float(ph.moments[0]), 1e-300)
    out = np.zeros(probs.size, dtype=float)
    low = 0.0

    for idx, prob in enumerate(probs):
        high = max(low, mean / max(1.0 - float(prob), 1e-12))
        while _ph_cdf(ph, high) < prob:
            high *= 2.0
            if high > mean * 1e12:
                raise RuntimeError(f"Could not bracket PH percentile {prob:.6g}.")

        lo = low
        hi = high
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            if _ph_cdf(ph, mid) >= prob:
                hi = mid
            else:
                lo = mid
        out[idx] = hi
        low = hi

    return out


def _validate_float_range(name: str, value_range: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if value_range is None:
        return None
    low, high = float(value_range[0]), float(value_range[1])
    if low < 0 or high < low:
        raise ValueError(f"{name} must satisfy 0 <= min <= max.")
    return low, high


def _validate_int_range(
    name: str,
    value_range: Optional[Tuple[int, int]],
    min_allowed: int,
    max_allowed: int,
) -> Optional[Tuple[int, int]]:
    if value_range is None:
        return None
    low, high = int(value_range[0]), int(value_range[1])
    if low < min_allowed or high > max_allowed or high < low:
        raise ValueError(f"{name} must satisfy {min_allowed} <= min <= max <= {max_allowed}.")
    return low, high


def _in_float_range(value: float, value_range: Optional[Tuple[float, float]]) -> bool:
    if value_range is None:
        return True
    low, high = value_range
    return low <= float(value) <= high


def _sample_policy_in_ranges(
    rng: np.random.Generator,
    s_range: Optional[Tuple[int, int]],
    S_range: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    s_low, s_high = s_range if s_range is not None else (FIXED_S, FIXED_S)
    S_low, S_high = S_range if S_range is not None else (FIXED_CAP_S, FIXED_CAP_S)
    candidates = [
        (s, S)
        for S in range(int(S_low), int(S_high) + 1)
        for s in range(int(s_low), int(s_high) + 1)
        if 0 <= s < S <= 30
    ]
    if not candidates:
        raise ValueError("No feasible (s, S) policy exists for the requested ranges with 0 <= s < S <= 30.")
    return candidates[int(rng.integers(0, len(candidates)))]


def _dynamic_segment_lengths(horizon: int, change_points: Sequence[int]) -> np.ndarray:
    counts = np.zeros(len(change_points) + 1, dtype=float)
    cps = np.array(change_points, dtype=int)
    seg_idx = 0
    for t in range(1, horizon + 1):
        while seg_idx < cps.size and t >= cps[seg_idx]:
            seg_idx += 1
        counts[seg_idx] += 1.0
    return counts


def dynamic_plan_average_mean(plan: DynamicDemandPlan, horizon: int) -> float:
    weights = _dynamic_segment_lengths(horizon, plan.change_points)
    return float(np.average(np.array(plan.means, dtype=float), weights=weights))


def dynamic_plan_average_scv(plan: DynamicDemandPlan, horizon: int) -> float:
    weights = _dynamic_segment_lengths(horizon, plan.change_points)
    scvs = np.array([_scv_from_moments(ph.moments) for ph in plan.phs], dtype=float)
    return float(np.average(scvs, weights=weights))


def _generate_ph_with_size_sampling(
    max_size: int,
    target_mean: float,
    rng: np.random.Generator,
    scv_range: Optional[Tuple[float, float]],
    max_tries: int,
) -> PHDistribution:
    scv_range = _validate_float_range("scv_range", scv_range)
    min_scv, max_scv = scv_range if scv_range is not None else (0.0, 20.0)
    last_error: Optional[Exception] = None
    for _ in range(max_tries):
        try:
            size = _sample_ph_size(max_size, rng)
            return designated_ph_generator(
                size=size,
                target_mean=target_mean,
                rng=rng,
                min_scv=min_scv,
                max_scv=max_scv,
                max_tries=max(200, max_tries),
            )
        except RuntimeError as exc:
            last_error = exc
    detail = f" Last generator error: {last_error}" if last_error is not None else ""
    raise RuntimeError(f"Could not generate PH with sampled size in SCV range {scv_range}.{detail}")


def _load_input_partitions(csv_path: Path, max_num_files: int) -> list[InputPartition]:
    if max_num_files < 0:
        raise ValueError("max_num_files must be non-negative.")
    partitions: list[InputPartition] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"Test Set", "D", "L", "rho", "S", "s", "num_files"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Partition CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            num_files = int(row["num_files"])
            if num_files < max_num_files:
                partitions.append(
                    InputPartition(
                        test_set=int(row["Test Set"]),
                        D=str(row["D"]).strip(),
                        L=str(row["L"]).strip(),
                        rho=str(row["rho"]).strip(),
                        S=str(row["S"]).strip(),
                        s=str(row["s"]).strip().lower(),
                        num_files=num_files,
                    )
                )
    if not partitions:
        raise ValueError(f"No partition rows have num_files < {max_num_files}.")
    return partitions


def _range_from_threshold_label(
    label: str,
    threshold: float,
    low_value: float,
    high_value: float,
) -> Tuple[float, float]:
    normalized = str(label).replace(" ", "")
    eps = 1e-9
    if normalized == f"<={threshold:g}":
        return low_value, threshold
    if normalized == f">{threshold:g}":
        return threshold + eps, high_value
    raise ValueError(f"Unsupported threshold label: {label!r}")


def _S_range_from_partition(label: str) -> Tuple[int, int]:
    normalized = str(label).replace(" ", "")
    if normalized == "<=15":
        return 1, 15
    if normalized == ">15":
        return 16, 30
    raise ValueError(f"Unsupported S partition label: {label!r}")


def _sample_policy_from_partition(partition: InputPartition, rng: np.random.Generator) -> Tuple[int, int]:
    S_low, S_high = _S_range_from_partition(partition.S)
    candidates: list[tuple[int, int]] = []
    for S in range(S_low, S_high + 1):
        midpoint = S / 2.0
        if partition.s == "small":
            candidates.extend((s, S) for s in range(0, S) if s <= midpoint)
        elif partition.s == "large":
            candidates.extend((s, S) for s in range(0, S) if s > midpoint)
        else:
            raise ValueError(f"Unsupported s partition label: {partition.s!r}")
    if not candidates:
        raise ValueError(f"No feasible s/S policy for partition test set {partition.test_set}.")
    return candidates[int(rng.integers(0, len(candidates)))]


def _control_ranges_from_partition(partition: InputPartition, max_tries: int) -> InputControlRanges:
    return InputControlRanges(
        inter_avg_scv=_range_from_threshold_label(partition.D, threshold=5.0, low_value=0.0, high_value=20.0),
        lead_scv=_range_from_threshold_label(partition.L, threshold=5.0, low_value=0.0, high_value=20.0),
        mean_ratio=_range_from_threshold_label(partition.rho, threshold=1.0, low_value=0.1, high_value=10.0),
        max_tries=max_tries,
    )


def generate_random_ph_wide(
    size: int,
    target_mean: float,
    rng: np.random.Generator,
    max_tries: int = 200,
    min_scv: float = 0.0,
    max_scv: float = 20.0,
) -> PHDistribution:
    """Wide PH generator aligned with ph_summary_table.py family mix."""
    if size <= 0:
        raise ValueError("size must be positive.")
    if target_mean <= 0:
        raise ValueError("target_mean must be positive.")
    if min_scv < 0 or max_scv < min_scv:
        raise ValueError("Need 0 <= min_scv <= max_scv.")

    families = ("base", "erlang", "hypererlang", "hyperexp", "hyperexp_ultra", "coxian", "coxian_extreme")
    probs = np.array([0.18, 0.12, 0.16, 0.16, 0.14, 0.12, 0.12], dtype=float)

    for _ in range(max_tries):
        fam = str(rng.choice(families, p=probs))
        try:
            if fam == "base":
                ph = generate_random_ph(size=size, target_mean=target_mean, rng=rng)
            elif fam == "erlang":
                ph = _gen_erlang_like_ph(size=size, target_mean=target_mean)
            elif fam == "hyperexp":
                ph = _gen_hyperexp_heavy_ph(size=size, target_mean=target_mean, rng=rng)
            elif fam == "hyperexp_ultra":
                ph = _gen_hyperexp_ultra_ph(size=size, target_mean=target_mean, rng=rng)
            elif fam == "hypererlang":
                ph = _gen_hypererlang_ph(size=size, target_mean=target_mean, rng=rng)
            elif fam == "coxian":
                ph = _gen_coxian_like_ph(size=size, target_mean=target_mean, rng=rng)
            else:
                ph = _gen_coxian_extreme_ph(size=size, target_mean=target_mean, rng=rng)

            moments = ph.moments
            scv = _scv_from_moments(moments)
            if (
                np.all(np.isfinite(moments))
                and np.all(moments > 0)
                and np.max(moments) < 1e300
                and np.isfinite(scv)
                and scv >= min_scv
                and scv <= max_scv
            ):
                return ph
        except np.linalg.LinAlgError:
            continue
        except FloatingPointError:
            continue

    fallback = generate_random_ph(size=size, target_mean=target_mean, rng=rng)
    fallback_scv = _scv_from_moments(fallback.moments)
    if min_scv <= fallback_scv <= max_scv:
        return fallback
    erlang = _gen_erlang_like_ph(size=size, target_mean=target_mean)
    erlang_scv = _scv_from_moments(erlang.moments)
    if min_scv <= erlang_scv <= max_scv:
        return erlang
    raise RuntimeError(f"Could not generate PH with SCV in [{min_scv}, {max_scv}] after {max_tries} tries.")


def _sample_ph_size(max_size: int, rng: np.random.Generator) -> int:
    """Sample PH size uniformly from 1..max_size."""
    if max_size <= 0:
        raise ValueError("max_size must be positive.")
    return int(rng.integers(1, max_size + 1))


def _sample_unique_policies(
    n_policies: int,
    rng: np.random.Generator,
    min_S: int = 5,
    max_S: int = 30,
) -> list[tuple[int, int]]:
    """Sample (s, S) pairs with min_S <= S <= max_S and 1 <= s <= S.

    - If n_policies <= number of feasible pairs, sampling is without replacement.
    - If n_policies is larger, all feasible pairs are reused in shuffled rounds.
    """
    candidates = [(s, S) for S in range(min_S, max_S + 1) for s in range(1, S + 1)]
    n_candidates = len(candidates)
    if n_policies <= n_candidates:
        idx = rng.choice(n_candidates, size=n_policies, replace=False)
        return [candidates[int(i)] for i in idx]

    # Overflow mode: keep randomness but allow repeats after all unique pairs are exhausted.
    out: list[tuple[int, int]] = []
    full_rounds = n_policies // n_candidates
    remainder = n_policies % n_candidates

    for _ in range(full_rounds):
        perm = rng.permutation(n_candidates)
        out.extend(candidates[int(i)] for i in perm)

    if remainder > 0:
        idx = rng.choice(n_candidates, size=remainder, replace=False)
        out.extend(candidates[int(i)] for i in idx)

    return out


def designated_ph_generator(
    size: int,
    rng: np.random.Generator,
    target_mean: float = 1.0,
    min_scv: float = 0.0,
    max_scv: float = 20.0,
    max_tries: int = 200,
) -> PHDistribution:
    """Designated PH-generation function (size-driven API)."""
    return generate_random_ph_wide(
        size=size,
        target_mean=target_mean,
        rng=rng,
        max_tries=max_tries,
        min_scv=min_scv,
        max_scv=max_scv,
    )


def exponential_ph(rate: float = 1.0) -> PHDistribution:
    """Build PH representation of Exp(rate)."""
    if rate <= 0:
        raise ValueError("rate must be positive.")
    alpha = np.array([1.0], dtype=float)
    T = np.array([[-float(rate)]], dtype=float)
    moments = np.array([factorial(k) / (rate**k) for k in range(1, 11)], dtype=float)
    rates, jump_cdfs = _build_jump_cdfs(T)
    return PHDistribution(alpha=alpha, T=T, moments=moments, rates=rates, jump_cdfs=jump_cdfs)


def sample_change_points_with_min_gap(
    horizon: int,
    n_changes: int,
    min_gap: int,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> np.ndarray:
    """Sample sorted change points in [1, horizon] with pairwise spacing >= min_gap."""
    if not (0 < n_changes <= horizon):
        raise ValueError("n_changes must be between 1 and horizon.")
    if min_gap < 1:
        raise ValueError("min_gap must be >= 1.")

    # Feasibility upper bound for simple spacing rule.
    max_feasible = (horizon - 1) // min_gap + 1
    if n_changes > max_feasible:
        raise ValueError("Requested n_changes/min_gap is infeasible for this horizon.")

    candidates = np.arange(1, horizon + 1, dtype=int)
    for _ in range(max_tries):
        pts = np.sort(rng.choice(candidates, size=n_changes, replace=False))
        if np.all(np.diff(pts) >= min_gap):
            return pts.astype(int)

    raise RuntimeError("Could not sample valid change points; try smaller n_changes or min_gap.")


def generate_dynamic_demand_plan(
    inter_size: int,
    horizon: int,
    rng: np.random.Generator,
    min_changes: int = 2,
    max_changes: int = 10,
    min_gap: int = 5,
    avg_scv_range: Optional[Tuple[float, float]] = None,
    avg_mean_range: Optional[Tuple[float, float]] = None,
    max_plan_tries: int = 1000,
) -> DynamicDemandPlan:
    """Generate dynamic inter-demand PHs with optional weighted-average controls."""
    if min_changes < 1 or max_changes < min_changes:
        raise ValueError("Need 1 <= min_changes <= max_changes.")
    max_feasible = (horizon - 1) // min_gap + 1
    max_changes = min(max_changes, max_feasible)
    if min_changes > max_changes:
        raise ValueError("Requested min_changes/min_gap is infeasible for this horizon.")
    avg_scv_range = _validate_float_range("avg_scv_range", avg_scv_range)
    avg_mean_range = _validate_float_range("avg_mean_range", avg_mean_range)
    if max_plan_tries < 1:
        raise ValueError("max_plan_tries must be >= 1.")

    mean_low, mean_high = avg_mean_range if avg_mean_range is not None else (0.1, 10.0)
    if mean_low <= 0:
        raise ValueError("Average inter-demand mean range must be positive.")
    ph_min_scv = 0.0
    ph_max_scv = max(20.0, avg_scv_range[1] if avg_scv_range is not None else 20.0)

    for _ in range(max_plan_tries):
        n_changes = int(rng.integers(min_changes, max_changes + 1))
        change_points = sample_change_points_with_min_gap(
            horizon=horizon,
            n_changes=n_changes,
            min_gap=min_gap,
            rng=rng,
        )

        means = np.zeros(n_changes + 1, dtype=float)
        phs = []
        prev_mean = None
        for seg in range(n_changes + 1):
            m = float(rng.uniform(mean_low, mean_high))
            while (mean_high > mean_low) and (prev_mean is not None) and (abs(m - prev_mean) < 1e-9):
                m = float(rng.uniform(mean_low, mean_high))
            means[seg] = m
            phs.append(
                _generate_ph_with_size_sampling(
                    max_size=inter_size,
                    target_mean=m,
                    rng=rng,
                    scv_range=(ph_min_scv, ph_max_scv),
                    max_tries=max_plan_tries,
                )
            )
            prev_mean = m

        plan = DynamicDemandPlan(
            change_points=change_points,
            means=means,
            phs=tuple(phs),
        )
        if _in_float_range(dynamic_plan_average_mean(plan, horizon), avg_mean_range) and _in_float_range(
            dynamic_plan_average_scv(plan, horizon),
            avg_scv_range,
        ):
            return plan

    raise RuntimeError("Could not generate dynamic demand plan inside requested average mean/SCV ranges.")


def generate_dynamic_exponential_demand_plan(
    horizon: int,
    rng: np.random.Generator,
    min_changes: int = 2,
    max_changes: int = 10,
    min_gap: int = 5,
) -> DynamicDemandPlan:
    """Generate piecewise-exponential inter-demand plan with random means in (0.1, 10)."""
    if min_changes < 1 or max_changes < min_changes:
        raise ValueError("Need 1 <= min_changes <= max_changes.")
    max_feasible = (horizon - 1) // min_gap + 1
    max_changes = min(max_changes, max_feasible)
    if min_changes > max_changes:
        raise ValueError("Requested min_changes/min_gap is infeasible for this horizon.")

    n_changes = int(rng.integers(min_changes, max_changes + 1))
    change_points = sample_change_points_with_min_gap(
        horizon=horizon,
        n_changes=n_changes,
        min_gap=min_gap,
        rng=rng,
    )

    means = np.zeros(n_changes + 1, dtype=float)
    phs = []
    prev_mean = None
    for seg in range(n_changes + 1):
        m = float(rng.uniform(0.1, 10.0))
        while (prev_mean is not None) and (abs(m - prev_mean) < 1e-9):
            m = float(rng.uniform(0.1, 10.0))
        means[seg] = m
        phs.append(exponential_ph(rate=1.0 / m))
        prev_mean = m

    return DynamicDemandPlan(
        change_points=change_points,
        means=means,
        phs=tuple(phs),
    )


def ph_shape_statistics(moments: Sequence[float]) -> Tuple[float, float, float]:
    """Return (SCV, skewness, kurtosis) from raw moments up to 4th."""
    m1, m2, m3, m4 = moments[:4]
    var = max(m2 - m1**2, 0.0)
    if var <= 0:
        return 0.0, 0.0, 0.0

    mu3 = m3 - 3 * m1 * m2 + 2 * (m1**3)
    mu4 = m4 - 4 * m1 * m3 + 6 * (m1**2) * m2 - 3 * (m1**4)
    scv = var / (m1**2)
    skew = mu3 / (var ** 1.5)
    kurt = mu4 / (var**2)
    return float(scv), float(skew), float(kurt)


class LostSalesInventorySystem:
    """Continuous-review lost-sales (s, S) inventory system."""

    def __init__(
        self,
        env: simpy.Environment,
        inter_demand_ph: PHDistribution,
        lead_time_ph: PHDistribution,
        s: int,
        S: int,
        horizon: int,
        rng: np.random.Generator,
        demand_changes: Optional[Tuple[Tuple[int, PHDistribution], ...]] = None,
    ) -> None:
        if not (0 <= s <= S):
            raise ValueError("Need 0 <= s <= S.")

        self.env = env
        self.inter_demand_ph = inter_demand_ph
        self.lead_time_ph = lead_time_ph
        self.s = int(s)
        self.S = int(S)
        self.horizon = int(horizon)
        self.rng = rng
        self.demand_changes = tuple(sorted(demand_changes or tuple(), key=lambda x: x[0]))
        self.demand_proc = None

        self.on_hand = int(S)
        self.order_outstanding = False
        self.orders_so_far = 0
        self.lost_sales_so_far = 0

        self.inv_path = np.zeros(self.horizon, dtype=np.int16)
        self.orders_path = np.zeros(self.horizon, dtype=np.int32)
        self.lost_path = np.zeros(self.horizon, dtype=np.int32)

    def maybe_place_order(self) -> None:
        # Single outstanding replenishment keeps inventory level bounded by S.
        if (not self.order_outstanding) and (self.on_hand <= self.s):
            qty = self.S - self.on_hand
            if qty > 0:
                self.orders_so_far += 1
                self.order_outstanding = True
                self.env.process(self.delivery_process(qty))

    def delivery_process(self, qty: int):
        lt = self.lead_time_ph.sample(self.rng)
        yield self.env.timeout(lt)
        self.on_hand = min(self.S, self.on_hand + qty)
        self.order_outstanding = False

    def demand_process(self):
        while True:
            try:
                dt = self.inter_demand_ph.sample(self.rng)
                yield self.env.timeout(dt)

                if self.on_hand > 0:
                    self.on_hand -= 1
                else:
                    self.lost_sales_so_far += 1

                self.maybe_place_order()
            except simpy.Interrupt:
                # Restart demand clock immediately from the new PH distribution.
                continue

    def demand_change_process(self):
        for t_change, new_ph in self.demand_changes:
            if t_change > self.horizon:
                break
            yield self.env.timeout(t_change - self.env.now)
            self.inter_demand_ph = new_ph
            if (self.demand_proc is not None) and self.demand_proc.is_alive:
                self.demand_proc.interrupt()

    def monitor_process(self):
        for t in range(1, self.horizon + 1):
            yield self.env.timeout(t - self.env.now)
            idx = t - 1
            self.inv_path[idx] = self.on_hand
            self.orders_path[idx] = self.orders_so_far
            self.lost_path[idx] = self.lost_sales_so_far


def run_one_replication(
    inter_demand_ph: PHDistribution,
    lead_time_ph: PHDistribution,
    s: int,
    S: int,
    horizon: int,
    rng: np.random.Generator,
    demand_changes: Optional[Tuple[Tuple[int, PHDistribution], ...]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = simpy.Environment()
    model = LostSalesInventorySystem(
        env=env,
        inter_demand_ph=inter_demand_ph,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        horizon=horizon,
        rng=rng,
        demand_changes=demand_changes,
    )
    model.demand_proc = env.process(model.demand_process())
    if demand_changes:
        env.process(model.demand_change_process())
    env.process(model.monitor_process())
    env.run(until=horizon + 1e-9)
    return model.inv_path, model.orders_path, model.lost_path


def aggregate_replications(
    inter_demand_ph: PHDistribution,
    lead_time_ph: PHDistribution,
    s: int,
    S: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run many replications and aggregate required outputs."""
    if S > 30 or S < 0:
        raise ValueError("This output format expects S in [0, 30].")

    inv_counts = np.zeros((horizon, 31), dtype=np.int64)
    orders_sum = np.zeros(horizon, dtype=np.float64)
    lost_sum = np.zeros(horizon, dtype=np.float64)

    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(n_replications)

    for child in children:
        rng = np.random.default_rng(child)
        inv, ords, lost = run_one_replication(
            inter_demand_ph=inter_demand_ph,
            lead_time_ph=lead_time_ph,
            s=s,
            S=S,
            horizon=horizon,
            rng=rng,
        )
        for t in range(horizon):
            level = int(inv[t])
            if 0 <= level <= 30:
                inv_counts[t, level] += 1
        orders_sum += ords
        lost_sum += lost

    inventory_distribution = inv_counts / float(n_replications)
    avg_orders_so_far = orders_sum / float(n_replications)
    avg_lost_sales_so_far = lost_sum / float(n_replications)
    return inventory_distribution, avg_orders_so_far, avg_lost_sales_so_far


def aggregate_replications_dynamic_demand(
    inter_size: int,
    lead_time_ph: PHDistribution,
    s: int,
    S: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
    min_changes: int = 2,
    max_changes: int = 10,
    min_gap: int = 5,
    demand_plan: Optional[DynamicDemandPlan] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DynamicDemandPlan]:
    """Aggregate outputs when inter-demand PH changes over time within each replication."""
    if S > 30 or S < 0:
        raise ValueError("This output format expects S in [0, 30].")

    inv_counts = np.zeros((horizon, 31), dtype=np.int64)
    orders_sum = np.zeros(horizon, dtype=np.float64)
    lost_sum = np.zeros(horizon, dtype=np.float64)

    if demand_plan is None:
        plan_rng = np.random.default_rng(seed)
        demand_plan = generate_dynamic_demand_plan(
            inter_size=inter_size,
            horizon=horizon,
            rng=plan_rng,
            min_changes=min_changes,
            max_changes=max_changes,
            min_gap=min_gap,
        )

    demand_changes = tuple(
        (int(t_change), demand_plan.phs[idx + 1]) for idx, t_change in enumerate(demand_plan.change_points)
    )

    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(n_replications)

    for child in children:
        rng = np.random.default_rng(child)
        inv, ords, lost = run_one_replication(
            inter_demand_ph=demand_plan.phs[0],
            lead_time_ph=lead_time_ph,
            s=s,
            S=S,
            horizon=horizon,
            rng=rng,
            demand_changes=demand_changes,
        )
        for t in range(horizon):
            level = int(inv[t])
            if 0 <= level <= 30:
                inv_counts[t, level] += 1
        orders_sum += ords
        lost_sum += lost

    inventory_distribution = inv_counts / float(n_replications)
    avg_orders_so_far = orders_sum / float(n_replications)
    avg_lost_sales_so_far = lost_sum / float(n_replications)
    return inventory_distribution, avg_orders_so_far, avg_lost_sales_so_far, demand_plan


def simulate_dynamic_demand_setting(
    inter_size: int,
    lead_size: int,
    s: int,
    S: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
    control_ranges: Optional[InputControlRanges] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DynamicDemandPlan]:
    """Simulate with dynamic inter-demand PH and fixed lead-time mean=1."""
    rng = np.random.default_rng(seed)
    control_ranges = control_ranges or InputControlRanges()
    lead_scv_range = _validate_float_range("lead_scv", control_ranges.lead_scv)
    inter_avg_scv_range = _validate_float_range("inter_avg_scv", control_ranges.inter_avg_scv)
    mean_ratio_range = _validate_float_range("mean_ratio", control_ranges.mean_ratio)
    lead_time_ph = _generate_ph_with_size_sampling(
        max_size=lead_size,
        target_mean=1.0,
        rng=rng,
        scv_range=lead_scv_range,
        max_tries=control_ranges.max_tries,
    )
    demand_plan = generate_dynamic_demand_plan(
        inter_size=inter_size,
        horizon=horizon,
        rng=rng,
        min_changes=2,
        max_changes=10,
        min_gap=5,
        avg_scv_range=inter_avg_scv_range,
        avg_mean_range=mean_ratio_range,
        max_plan_tries=control_ranges.max_tries,
    )
    inv_dist, avg_orders, avg_lost, sample_plan = aggregate_replications_dynamic_demand(
        inter_size=inter_size,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        n_replications=n_replications,
        horizon=horizon,
        seed=seed,
        min_changes=2,
        max_changes=10,
        min_gap=5,
        demand_plan=demand_plan,
    )
    input_vector = build_time_epoch_input_matrix(
        horizon=horizon,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        demand_plan=sample_plan,
    )
    return input_vector, inv_dist, avg_orders, avg_lost, sample_plan


def generate_controlled_setting(
    inter_size: int,
    lead_size: int,
    rng: np.random.Generator,
    control_ranges: Optional[InputControlRanges] = None,
) -> Tuple[np.ndarray, PHDistribution, PHDistribution, int, int]:
    """Generate one static setting while enforcing optional control ranges."""
    control_ranges = control_ranges or InputControlRanges()
    inter_avg_scv_range = _validate_float_range("inter_avg_scv", control_ranges.inter_avg_scv)
    lead_scv_range = _validate_float_range("lead_scv", control_ranges.lead_scv)
    mean_ratio_range = _validate_float_range("mean_ratio", control_ranges.mean_ratio)
    s_range = _validate_int_range("s", control_ranges.s, 0, 29)
    S_range = _validate_int_range("S", control_ranges.S, 1, 30)
    if control_ranges.max_tries < 1:
        raise ValueError("max_tries must be >= 1.")

    mean_low, mean_high = mean_ratio_range if mean_ratio_range is not None else (1.0, 1.0)

    last_error: Optional[Exception] = None
    for _ in range(control_ranges.max_tries):
        try:
            inter_mean = float(rng.uniform(mean_low, mean_high))
            inter_demand_ph = _generate_ph_with_size_sampling(
                max_size=inter_size,
                target_mean=inter_mean,
                rng=rng,
                scv_range=inter_avg_scv_range,
                max_tries=control_ranges.max_tries,
            )
            lead_time_ph = _generate_ph_with_size_sampling(
                max_size=lead_size,
                target_mean=1.0,
                rng=rng,
                scv_range=lead_scv_range,
                max_tries=control_ranges.max_tries,
            )
            s, S = _sample_policy_in_ranges(rng, s_range, S_range)
            input_vector = build_input_vector(inter_demand_ph, lead_time_ph, s=s, S=S)
            return input_vector, inter_demand_ph, lead_time_ph, s, S
        except RuntimeError as exc:
            last_error = exc

    detail = f" Last generator error: {last_error}" if last_error is not None else ""
    raise RuntimeError(f"Could not generate controlled setting after {control_ranges.max_tries} tries.{detail}")


def generate_random_setting(
    inter_size: int,
    lead_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, PHDistribution, PHDistribution, int, int]:
    """Generate one random setting and its model input vector."""
    inter_size_sample = _sample_ph_size(inter_size, rng)
    inter_demand_ph = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
    lead_mean = float(rng.uniform(0.1, 10.0))
    lead_size_sample = _sample_ph_size(lead_size, rng)
    lead_time_ph = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)

    S = int(rng.integers(5, 31))
    s = int(rng.integers(1, S + 1))

    input_vector = build_input_vector(inter_demand_ph, lead_time_ph, s=s, S=S)
    return input_vector, inter_demand_ph, lead_time_ph, s, S


def build_input_vector(
    inter_demand_ph: PHDistribution,
    lead_time_ph: PHDistribution,
    s: int,
    S: int,
) -> np.ndarray:
    """Build one input row with log-moments, policy, and raw percentile times."""
    eps = 1e-300
    inter_log = np.log(np.maximum(inter_demand_ph.moments, eps))
    lead_log = np.log(np.maximum(lead_time_ph.moments, eps))
    inter_percentiles = ph_percentiles(inter_demand_ph)
    lead_percentiles = ph_percentiles(lead_time_ph)
    return np.concatenate(
        [
            inter_log,
            lead_log,
            np.array([float(s), float(S)], dtype=float),
            inter_percentiles,
            lead_percentiles,
        ]
    )


def build_time_epoch_input_matrix(
    horizon: int,
    lead_time_ph: PHDistribution,
    s: int,
    S: int,
    inter_demand_ph: Optional[PHDistribution] = None,
    demand_plan: Optional[DynamicDemandPlan] = None,
) -> np.ndarray:
    """Build time-indexed input matrix x with shape (horizon, 44)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if (inter_demand_ph is None) and (demand_plan is None):
        raise ValueError("Provide either inter_demand_ph or demand_plan.")
    if (inter_demand_ph is not None) and (demand_plan is not None):
        raise ValueError("Provide only one of inter_demand_ph or demand_plan.")

    x = np.zeros((horizon, INPUT_FEATURE_COUNT), dtype=float)
    if inter_demand_ph is not None:
        row = build_input_vector(inter_demand_ph, lead_time_ph, s=s, S=S)
        x[:] = row
        return x

    cps = np.array(demand_plan.change_points, dtype=int)
    rows = [
        build_input_vector(inter_ph, lead_time_ph, s=s, S=S)
        for inter_ph in demand_plan.phs
    ]
    seg_idx = 0
    for t in range(1, horizon + 1):
        while seg_idx < cps.size and t >= cps[seg_idx]:
            seg_idx += 1
        x[t - 1] = rows[seg_idx]
    return x


def lead_scv_from_input_vector(x: np.ndarray) -> float:
    """Extract lead-time SCV from x (supports shape (44,) or (T,44), with log-moments)."""
    if x.ndim == 2:
        row = x[0]
    elif x.ndim == 1:
        row = x
    else:
        raise ValueError("x must be 1D or 2D.")
    if row.shape[0] < 12:
        raise ValueError("Input vector x must include at least first two lead moments.")

    # x stores log-moments, so map back.
    m1 = float(np.exp(row[10]))
    m2 = float(np.exp(row[11]))
    if m1 <= 0:
        raise ValueError("Lead-time first moment must be positive.")
    var = max(0.0, m2 - m1 * m1)
    return var / (m1 * m1)


def average_inter_mean_from_input_vector(x: np.ndarray) -> float:
    """Return the time-average inter-demand mean from log-moment input rows."""
    rows = x[None, :] if x.ndim == 1 else x
    return float(np.mean(np.exp(rows[:, 0])))


def average_inter_scv_from_input_vector(x: np.ndarray) -> float:
    """Return the time-average inter-demand SCV from log-moment input rows."""
    rows = x[None, :] if x.ndim == 1 else x
    m1 = np.exp(rows[:, 0])
    m2 = np.exp(rows[:, 1])
    var = np.maximum(0.0, m2 - m1 * m1)
    return float(np.mean(var / np.maximum(m1 * m1, 1e-300)))


def _format_float_for_filename(value: float) -> str:
    token = f"{value:.6f}".rstrip("0").rstrip(".")
    if token == "":
        token = "0"
    return token.replace("-", "m")


def save_io_pickles(
    x: np.ndarray,
    inv: np.ndarray,
    order: np.ndarray,
    loss: np.ndarray,
    scv_leadtime: float,
    number_demand_rates: int,
    model_number: int,
    S: int,
    s: int,
    model_num: int,
    inv_dir: Path,
    order_dir: Path,
    loss_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Save (x, inv), (x, order), (x, loss) pickles under requested naming convention."""
    inv_dir.mkdir(parents=True, exist_ok=True)
    order_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    scv_token = _format_float_for_filename(float(scv_leadtime))
    common = f"{scv_token}_{int(number_demand_rates)}_{int(model_number)}_{int(S)}_{int(s)}_{int(model_num)}"

    inv_path = inv_dir / f"inv_{common}.pkl"
    order_path = order_dir / f"order_{common}.pkl"
    loss_path = loss_dir / f"loss_{common}.pkl"

    with inv_path.open("wb") as f:
        pickle.dump((x, inv), f)
    with order_path.open("wb") as f:
        pickle.dump((x, order), f)
    with loss_path.open("wb") as f:
        pickle.dump((x, loss), f)

    return inv_path, order_path, loss_path


def simulate_given_setting(
    inter_demand_ph: PHDistribution,
    lead_time_ph: PHDistribution,
    s: int,
    S: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a fixed (inter PH, lead PH, s, S) setting and return all outputs."""
    input_vector = build_time_epoch_input_matrix(
        horizon=horizon,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        inter_demand_ph=inter_demand_ph,
    )
    inventory_distribution, avg_orders_so_far, avg_lost_sales_so_far = aggregate_replications(
        inter_demand_ph=inter_demand_ph,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        n_replications=n_replications,
        horizon=horizon,
        seed=seed,
    )
    return input_vector, inventory_distribution, avg_orders_so_far, avg_lost_sales_so_far


def simulate_single_setting(
    inter_size: int,
    lead_size: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one random setting and simulate it.

    Returns:
        input_vector: (100, 44)
        inventory_distribution: (100, 31)
        avg_orders_so_far: (100,)
        avg_lost_sales_so_far: (100,)
    """
    rng = np.random.default_rng(seed)
    _, inter_demand_ph, lead_time_ph, s, S = generate_random_setting(
        inter_size=inter_size,
        lead_size=lead_size,
        rng=rng,
    )
    input_vector = build_time_epoch_input_matrix(
        horizon=horizon,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        inter_demand_ph=inter_demand_ph,
    )

    inventory_distribution, avg_orders_so_far, avg_lost_sales_so_far = aggregate_replications(
        inter_demand_ph=inter_demand_ph,
        lead_time_ph=lead_time_ph,
        s=s,
        S=S,
        n_replications=n_replications,
        horizon=horizon,
        seed=seed,
    )
    return input_vector, inventory_distribution, avg_orders_so_far, avg_lost_sales_so_far


def simulate_multiple_settings(
    n_settings: int,
    inter_size: int,
    lead_size: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
    control_ranges: Optional[InputControlRanges] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset across multiple random settings.

    Returns:
        X_inputs: (n_settings, 100, 44)
        Y_inventory: (n_settings, 100, 31)
        Y_orders: (n_settings, 100)
        Y_lost_sales: (n_settings, 100)
    """
    master = np.random.SeedSequence(seed)
    setting_seeds = master.spawn(n_settings)

    X_inputs = np.zeros((n_settings, horizon, INPUT_FEATURE_COUNT), dtype=float)
    Y_inventory = np.zeros((n_settings, horizon, 31), dtype=float)
    Y_orders = np.zeros((n_settings, horizon), dtype=float)
    Y_lost = np.zeros((n_settings, horizon), dtype=float)
    policy_rng = np.random.default_rng(seed)
    unique_policies = None if control_ranges is not None else _sample_unique_policies(n_settings, policy_rng)

    for idx, ss in enumerate(setting_seeds):
        rng = np.random.default_rng(ss)
        if control_ranges is None:
            inter_size_sample = _sample_ph_size(inter_size, rng)
            inter_demand_ph = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
            lead_mean = float(rng.uniform(0.1, 10.0))
            lead_size_sample = _sample_ph_size(lead_size, rng)
            lead_time_ph = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)
            s, S = unique_policies[idx]
        else:
            _, inter_demand_ph, lead_time_ph, s, S = generate_controlled_setting(
                inter_size=inter_size,
                lead_size=lead_size,
                rng=rng,
                control_ranges=control_ranges,
            )
        input_vector = build_time_epoch_input_matrix(
            horizon=horizon,
            lead_time_ph=lead_time_ph,
            s=s,
            S=S,
            inter_demand_ph=inter_demand_ph,
        )
        inv_dist, avg_orders, avg_lost = aggregate_replications(
            inter_demand_ph=inter_demand_ph,
            lead_time_ph=lead_time_ph,
            s=s,
            S=S,
            n_replications=n_replications,
            horizon=horizon,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        X_inputs[idx] = input_vector
        Y_inventory[idx] = inv_dist
        Y_orders[idx] = avg_orders
        Y_lost[idx] = avg_lost

    return X_inputs, Y_inventory, Y_orders, Y_lost


def _enumerate_states_single_outstanding(s: int, S: int):
    """Enumerate CTMC states (on_hand, q_outstanding) for this simulator logic."""
    states = []
    index = {}

    # No outstanding order.
    for i in range(S + 1):
        st = (i, 0)
        index[st] = len(states)
        states.append(st)

    # One outstanding order with fixed quantity q.
    q_min = max(1, S - s)
    for q in range(q_min, S + 1):
        for i in range(S - q + 1):
            st = (i, q)
            index[st] = len(states)
            states.append(st)

    return states, index


def build_exponential_generator(
    s: int,
    S: int,
    demand_rate: float = 1.0,
    lead_rate: float = 1.0,
) -> Tuple[np.ndarray, list, dict]:
    """Build CTMC generator Q for exponential demand/lead-time case."""
    if not (1 <= s <= S <= 30):
        raise ValueError("Need 1 <= s <= S <= 30 for this setup.")
    if demand_rate <= 0 or lead_rate <= 0:
        raise ValueError("Rates must be positive.")

    states, index = _enumerate_states_single_outstanding(s=s, S=S)
    n = len(states)
    Q = np.zeros((n, n), dtype=float)

    for row, (i, q) in enumerate(states):
        # Demand event.
        if i > 0:
            ni = i - 1
            nq = q
            if q == 0 and ni <= s:
                nq = S - ni
            col = index[(ni, nq)]
            Q[row, col] += demand_rate
        elif q == 0:
            # Lost sale at i=0 triggers an order in this simulator implementation.
            col = index[(0, S)]
            Q[row, col] += demand_rate

        # Delivery event.
        if q > 0:
            col = index[(i + q, 0)]
            Q[row, col] += lead_rate

    row_sums = Q.sum(axis=1)
    Q[np.arange(n), np.arange(n)] = -row_sums
    return Q, states, index


def ctmc_transient_uniformization(
    Q: np.ndarray,
    p0: np.ndarray,
    times: np.ndarray,
    tol: float = 1e-13,
    max_terms: int = 5000,
) -> np.ndarray:
    """Transient CTMC probabilities p(t) for row-vector p0 using uniformization."""
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if p0.ndim != 1 or p0.shape[0] != Q.shape[0]:
        raise ValueError("p0 shape must match Q.")

    n = Q.shape[0]
    nu = float(np.max(-np.diag(Q)))
    if nu <= 0:
        return np.repeat(p0.reshape(1, n), repeats=len(times), axis=0)

    P = np.eye(n) + Q / nu
    out = np.zeros((len(times), n), dtype=float)

    for t_idx, t in enumerate(times):
        if t < 0:
            raise ValueError("Times must be nonnegative.")
        x = nu * float(t)

        pk = p0.copy()
        w = np.exp(-x)
        pt = w * pk

        # Around mean x, Poisson mass is concentrated in roughly x +- O(sqrt(x)).
        k_cap = int(min(max_terms, max(50, x + 14.0 * np.sqrt(max(x, 1e-12)) + 40.0)))
        for k in range(1, k_cap + 1):
            pk = pk @ P
            w *= x / k
            pt += w * pk

            if (k > x) and (w < tol):
                break

        pt = np.clip(pt, 0.0, None)
        total = pt.sum()
        if total > 0:
            pt /= total
        out[t_idx] = pt

    return out


def analytic_inventory_distribution_exponential(
    s: int,
    S: int,
    horizon: int = 100,
    demand_rate: float = 1.0,
    lead_rate: float = 1.0,
) -> np.ndarray:
    """Analytic transient inventory distribution for exponential case."""
    Q, states, index = build_exponential_generator(
        s=s, S=S, demand_rate=demand_rate, lead_rate=lead_rate
    )
    p0 = np.zeros(Q.shape[0], dtype=float)
    p0[index[(S, 0)]] = 1.0
    times = np.arange(1, horizon + 1, dtype=float)
    p_states = ctmc_transient_uniformization(Q=Q, p0=p0, times=times)

    inv_dist = np.zeros((horizon, 31), dtype=float)
    for state_idx, (i, _q) in enumerate(states):
        inv_dist[:, i] += p_states[:, state_idx]
    return inv_dist


def analytic_inventory_distribution_exponential_piecewise(
    s: int,
    S: int,
    change_points: Sequence[int],
    means: Sequence[float],
    horizon: int = 100,
    lead_rate: float = 1.0,
) -> np.ndarray:
    """Analytic transient inventory distribution for piecewise-exponential demand rates."""
    cps = np.array(sorted(int(x) for x in change_points), dtype=int)
    means_arr = np.array(means, dtype=float)
    if means_arr.size != cps.size + 1:
        raise ValueError("Need len(means) == len(change_points) + 1.")
    if np.any(means_arr <= 0):
        raise ValueError("All means must be positive.")
    if np.any((cps < 1) | (cps > horizon)):
        raise ValueError("change_points must be in [1, horizon].")

    boundaries = list(cps) + [horizon]
    p0 = None
    prev_t = 0
    inv_dist = np.zeros((horizon, 31), dtype=float)

    for seg_idx, end_t in enumerate(boundaries):
        if end_t <= prev_t:
            continue
        demand_rate = 1.0 / float(means_arr[seg_idx])
        Q, states, index = build_exponential_generator(
            s=s,
            S=S,
            demand_rate=demand_rate,
            lead_rate=lead_rate,
        )
        if p0 is None:
            p0 = np.zeros(Q.shape[0], dtype=float)
            p0[index[(S, 0)]] = 1.0

        local_times = np.arange(1, end_t - prev_t + 1, dtype=float)
        p_local = ctmc_transient_uniformization(Q=Q, p0=p0, times=local_times)

        seg_inv = np.zeros((end_t - prev_t, 31), dtype=float)
        for state_idx, (i, _q) in enumerate(states):
            seg_inv[:, i] += p_local[:, state_idx]

        inv_dist[prev_t:end_t, :] = seg_inv
        p0 = p_local[-1]
        prev_t = end_t

    return inv_dist


def simulate_exponential_with_analytic(
    s: int,
    S: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
    demand_rate: float = 1.0,
    lead_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run simulation + analytic transient results for exponential rates."""
    inter = exponential_ph(rate=demand_rate)
    lead = exponential_ph(rate=lead_rate)
    input_vector, sim_inv, sim_orders, sim_lost = simulate_given_setting(
        inter_demand_ph=inter,
        lead_time_ph=lead,
        s=s,
        S=S,
        n_replications=n_replications,
        horizon=horizon,
        seed=seed,
    )
    analytic_inv = analytic_inventory_distribution_exponential(
        s=s,
        S=S,
        horizon=horizon,
        demand_rate=demand_rate,
        lead_rate=lead_rate,
    )
    return input_vector, sim_inv, sim_orders, sim_lost, analytic_inv


def simulate_exponential_time_varying_with_analytic(
    s: int,
    S: int,
    n_replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
    lead_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, DynamicDemandPlan]:
    """Run simulation + analytic results for piecewise-exponential inter-demand rates."""
    rng = np.random.default_rng(seed)
    demand_plan = generate_dynamic_exponential_demand_plan(
        horizon=horizon,
        rng=rng,
        min_changes=2,
        max_changes=10,
        min_gap=5,
    )
    lead = exponential_ph(rate=lead_rate)
    sim_inv, sim_orders, sim_lost, _ = aggregate_replications_dynamic_demand(
        inter_size=1,
        lead_time_ph=lead,
        s=s,
        S=S,
        n_replications=n_replications,
        horizon=horizon,
        seed=seed,
        min_changes=2,
        max_changes=10,
        min_gap=5,
        demand_plan=demand_plan,
    )
    analytic_inv = analytic_inventory_distribution_exponential_piecewise(
        s=s,
        S=S,
        change_points=demand_plan.change_points,
        means=demand_plan.means,
        horizon=horizon,
        lead_rate=lead_rate,
    )
    input_vector = build_time_epoch_input_matrix(
        horizon=horizon,
        lead_time_ph=lead,
        s=s,
        S=S,
        demand_plan=demand_plan,
    )
    return input_vector, sim_inv, sim_orders, sim_lost, analytic_inv, demand_plan


def plot_inventory_level_probability(
    inventory_distribution: np.ndarray,
    level: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot P(Inventory=level at time t) for t=1..T."""
    if inventory_distribution.ndim != 2:
        raise ValueError("inventory_distribution must be a 2D array of shape (T, 31).")
    if level < 0 or level >= inventory_distribution.shape[1]:
        raise ValueError("Requested inventory level is outside the array columns.")

    import matplotlib.pyplot as plt

    horizon = inventory_distribution.shape[0]
    times = np.arange(1, horizon + 1)
    probs = inventory_distribution[:, level]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(times, probs, color="#1f77b4", linewidth=2.2)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"P(Inventory = {level})")
    ax.set_title(f"Probability of {level} Units in Inventory vs Time")
    ax.set_xlim(1, horizon)
    ax.grid(True, alpha=0.25)

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_simulation_vs_analytic_levels(
    simulation_inventory_distribution: np.ndarray,
    analytic_inventory_distribution: np.ndarray,
    levels: Sequence[int] = tuple(range(11)),
    change_points: Optional[Sequence[int]] = None,
    means: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot simulation vs analytic inventory probabilities for each requested level."""
    import matplotlib.pyplot as plt

    levels = list(levels)
    horizon = simulation_inventory_distribution.shape[0]
    times = np.arange(1, horizon + 1)

    n_levels = len(levels)
    ncols = 3
    nrows = int(np.ceil(n_levels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax_idx, level in enumerate(levels):
        ax = axes[ax_idx]
        ax.plot(
            times,
            analytic_inventory_distribution[:, level],
            color="#d62728",
            linewidth=2.0,
            label="Analytic",
        )
        ax.plot(
            times,
            simulation_inventory_distribution[:, level],
            color="#1f77b4",
            linewidth=1.8,
            linestyle="--",
            label="Simulation",
        )
        ax.set_title(f"Inventory={level}")
        ax.grid(True, alpha=0.25)

        if change_points is not None:
            for cp in change_points:
                if 1 <= int(cp) <= horizon:
                    ax.axvline(int(cp), color="black", linestyle="--", alpha=0.35, linewidth=0.9)

    for extra_idx in range(n_levels, len(axes)):
        axes[extra_idx].axis("off")

    if (change_points is not None) and (means is not None) and len(means) == len(change_points) + 1:
        y0, y1 = axes[0].get_ylim()
        ybase = y1 - 0.08 * (y1 - y0)
        for idx, cp in enumerate(change_points):
            rate = 1.0 / float(means[idx + 1])
            axes[0].text(
                int(cp) + 0.2,
                ybase - (idx % 3) * 0.08 * (y1 - y0),
                f"r~{rate:.3f}",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="black",
            )

        init_rate = 1.0 / float(means[0])
        title_suffix = f"\nPiecewise demand rates, initial r~{init_rate:.3f}"
    else:
        title_suffix = ""

    axes[0].legend(loc="best")
    fig.suptitle("Transient Inventory Probabilities: Simulation vs Analytic (Exponential Case)" + title_suffix)
    fig.supxlabel("Time")
    fig.supylabel("Probability")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_inventory_probabilities_0_to_S_with_changes(
    inventory_distribution: np.ndarray,
    S: int,
    change_points: Sequence[int],
    means: Sequence[float],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot P(Inventory=i) for i=0..S with demand-change markers and rates."""
    import matplotlib.pyplot as plt

    if S < 0 or S > 30:
        raise ValueError("S must be in [0, 30].")
    if inventory_distribution.ndim != 2 or inventory_distribution.shape[1] < (S + 1):
        raise ValueError("inventory_distribution must have columns 0..S.")

    horizon = inventory_distribution.shape[0]
    times = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.get_cmap("tab20", max(S + 1, 2))
    for i in range(S + 1):
        ax.plot(times, inventory_distribution[:, i], linewidth=1.5, color=cmap(i), label=f"i={i}")

    ymax = max(1e-9, float(np.max(inventory_distribution[:, : S + 1])))
    for idx, cp in enumerate(change_points):
        if 1 <= cp <= horizon:
            ax.axvline(cp, color="black", linestyle="--", alpha=0.45, linewidth=1.0)
            new_mean = float(means[idx + 1])
            new_rate = 1.0 / new_mean
            ax.text(
                cp + 0.25,
                ymax * (0.95 - 0.06 * (idx % 3)),
                f"t={cp}, r~{new_rate:.3f}",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="black",
            )

    initial_rate = 1.0 / float(means[0])
    ax.set_title(
        f"Transient Inventory Probabilities P(I(t)=i), i=0..{S}\n"
        f"Initial inter-demand equivalent rate ~ {initial_rate:.3f}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_xlim(1, horizon)
    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title="Inventory level",
        frameon=True,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _optional_float_range(name: str, low: Optional[float], high: Optional[float]) -> Optional[Tuple[float, float]]:
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"Provide both --{name}-min and --{name}-max.")
    return _validate_float_range(name, (float(low), float(high)))


def _optional_int_range(
    name: str,
    low: Optional[int],
    high: Optional[int],
    min_allowed: int,
    max_allowed: int,
) -> Optional[Tuple[int, int]]:
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"Provide both --{name}-min and --{name}-max.")
    return _validate_int_range(name, (int(low), int(high)), min_allowed, max_allowed)


def _build_control_ranges(args) -> Optional[InputControlRanges]:
    inter_avg_scv = _optional_float_range("inter-avg-scv", args.inter_avg_scv_min, args.inter_avg_scv_max)
    lead_scv = _optional_float_range("lead-scv", args.lead_scv_min, args.lead_scv_max)
    mean_ratio = _optional_float_range("mean-ratio", args.mean_ratio_min, args.mean_ratio_max)
    s_range = _optional_int_range("s", args.s_min, args.s_max, 0, 29)
    S_range = _optional_int_range("S", args.S_min, args.S_max, 1, 30)
    exact_s = None if args.s is None else int(args.s)
    exact_S = None if args.S is None else int(args.S)

    if s_range is not None and exact_s is not None:
        raise ValueError("Use either --s or --s-min/--s-max, not both.")
    if S_range is not None and exact_S is not None:
        raise ValueError("Use either --S or --S-min/--S-max, not both.")
    if s_range is None:
        s_range = (FIXED_S if exact_s is None else exact_s, FIXED_S if exact_s is None else exact_s)
    if S_range is None:
        S_range = (FIXED_CAP_S if exact_S is None else exact_S, FIXED_CAP_S if exact_S is None else exact_S)
    _validate_int_range("s", s_range, 0, 29)
    _validate_int_range("S", S_range, 1, 30)

    controls_requested = any(
        value is not None
        for value in (
            inter_avg_scv,
            lead_scv,
            mean_ratio,
            args.s_min,
            args.s_max,
            args.S_min,
            args.S_max,
            args.s,
            args.S,
        )
    )
    if not controls_requested:
        return None

    return InputControlRanges(
        inter_avg_scv=inter_avg_scv,
        lead_scv=lead_scv,
        mean_ratio=mean_ratio,
        s=s_range,
        S=S_range,
        max_tries=int(args.control_max_tries),
    )


def _sample_cli_policy(rng: np.random.Generator, control_ranges: Optional[InputControlRanges]) -> Tuple[int, int]:
    if control_ranges is None:
        return FIXED_S, FIXED_CAP_S
    return _sample_policy_in_ranges(rng, control_ranges.s, control_ranges.S)


def _print_control_summary(control_ranges: Optional[InputControlRanges]) -> None:
    if control_ranges is None:
        print(f"input controls: legacy defaults, fixed s={FIXED_S}, S={FIXED_CAP_S}")
        return
    print(
        "input controls: "
        f"inter_avg_scv={control_ranges.inter_avg_scv}, "
        f"lead_scv={control_ranges.lead_scv}, "
        f"mean_ratio={control_ranges.mean_ratio}, "
        f"s={control_ranges.s}, S={control_ranges.S}, "
        f"max_tries={control_ranges.max_tries}"
    )


def _run_partition_sampled_simulations(args, inv_dir: Path, order_dir: Path, loss_dir: Path) -> None:
    partition_csv_path = Path(args.partition_counts_csv)
    if not partition_csv_path.is_absolute():
        partition_csv_path = Path(__file__).resolve().parent / partition_csv_path
    partitions = _load_input_partitions(
        csv_path=partition_csv_path,
        max_num_files=int(args.partition_count_threshold),
    )
    rng = np.random.default_rng(args.seed)
    meta_rng = np.random.default_rng(None if args.seed is None else args.seed + 9173)
    saved_triplets: list[tuple[Path, Path, Path]] = []

    print(
        "partition-sampling mode: "
        f"{len(partitions)} eligible rows with num_files < {args.partition_count_threshold}; "
        f"running {args.partition_simulations} dynamic-demand simulations"
    )

    for idx in range(int(args.partition_simulations)):
        partition = partitions[int(rng.integers(0, len(partitions)))]
        control_ranges = _control_ranges_from_partition(partition, max_tries=int(args.control_max_tries))
        s, S = _sample_policy_from_partition(partition, rng)
        rep_seed = None if args.seed is None else int(rng.integers(0, 2**31 - 1))

        lead_scv_range = _validate_float_range("lead_scv", control_ranges.lead_scv)
        inter_avg_scv_range = _validate_float_range("inter_avg_scv", control_ranges.inter_avg_scv)
        mean_ratio_range = _validate_float_range("mean_ratio", control_ranges.mean_ratio)
        lead_time_ph = _generate_ph_with_size_sampling(
            max_size=args.lead_size,
            target_mean=1.0,
            rng=rng,
            scv_range=lead_scv_range,
            max_tries=control_ranges.max_tries,
        )
        demand_plan = generate_dynamic_demand_plan(
            inter_size=args.inter_size,
            horizon=args.horizon,
            rng=rng,
            min_changes=2,
            max_changes=10,
            min_gap=5,
            avg_scv_range=inter_avg_scv_range,
            avg_mean_range=mean_ratio_range,
            max_plan_tries=control_ranges.max_tries,
        )
        avg_inter_scv = dynamic_plan_average_scv(demand_plan, args.horizon)
        lead_scv = _scv_from_moments(lead_time_ph.moments)
        mean_ratio = dynamic_plan_average_mean(demand_plan, args.horizon) / float(lead_time_ph.moments[0])

        print(
            f"[partition run {idx + 1}/{args.partition_simulations}] selected before simulation -> "
            f"test_set={partition.test_set}, source_count={partition.num_files}, "
            f"D={partition.D}, L={partition.L}, rho={partition.rho}, S_part={partition.S}, s_part={partition.s}, "
            f"avg_inter_scv={avg_inter_scv:.6g}, lead_scv={lead_scv:.6g}, "
            f"mean_ratio={mean_ratio:.6g}, S={S}, s={s}",
            flush=True,
        )

        inv, orders, lost, sample_plan = aggregate_replications_dynamic_demand(
            inter_size=args.inter_size,
            lead_time_ph=lead_time_ph,
            s=s,
            S=S,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=rep_seed,
            min_changes=2,
            max_changes=10,
            min_gap=5,
            demand_plan=demand_plan,
        )
        x = build_time_epoch_input_matrix(
            horizon=args.horizon,
            lead_time_ph=lead_time_ph,
            s=s,
            S=S,
            demand_plan=sample_plan,
        )

        scv_leadtime = lead_scv_from_input_vector(x)
        avg_inter_scv = average_inter_scv_from_input_vector(x)
        avg_inter_mean = average_inter_mean_from_input_vector(x)
        lead_mean = float(np.exp(x[0, 10]))
        mean_ratio = avg_inter_mean / lead_mean
        number_demand_rates = int(len(sample_plan.means))
        model_num_i = int(meta_rng.integers(1, 1_000_001))

        inv_path, order_path, loss_path = save_io_pickles(
            x=x,
            inv=inv,
            order=orders,
            loss=lost,
            scv_leadtime=scv_leadtime,
            number_demand_rates=number_demand_rates,
            model_number=args.model_number,
            S=S,
            s=s,
            model_num=model_num_i,
            inv_dir=inv_dir,
            order_dir=order_dir,
            loss_dir=loss_dir,
        )
        saved_triplets.append((inv_path, order_path, loss_path))

        print(
            f"[partition run {idx + 1}/{args.partition_simulations}] "
            f"test_set={partition.test_set}, source_count={partition.num_files}, "
            f"D={partition.D}, L={partition.L}, rho={partition.rho}, S_part={partition.S}, s_part={partition.s}, "
            f"s={s}, S={S}, rates={number_demand_rates}, "
            f"avg_inter_scv={avg_inter_scv:.6g}, lead_scv={scv_leadtime:.6g}, mean_ratio={mean_ratio:.6g}"
        )

    print(f"Completed {len(saved_triplets)} partition-sampled settings and saved all pickle triplets.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run PH (s,S) simulation and plot P(Inventory=level).")
    parser.add_argument("--inter-size", type=int, default=100, help="Max inter-demand PH size (sampled in 1..inter-size).")
    parser.add_argument("--lead-size", type=int, default=100, help="Max lead-time PH size (sampled in 1..lead-size).")
    parser.add_argument("--replications", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. Default: None (non-deterministic run each time).",
    )
    parser.add_argument(
        "--n-settings",
        type=int,
        default=1,
        help="Number of random baseline PH settings to simulate and save in one run.",
    )
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--output", type=str, default="p_inv_0_to_S_dynamic.png")
    parser.add_argument(
        "--compare-output",
        type=str,
        default="exp_sim_vs_analytic_0_to_S.png",
        help="Output path for simulation-vs-analytic comparison figure.",
    )
    parser.add_argument("--show", action="store_true", help="Show the graph window.")
    parser.add_argument("--s", type=int, default=None, help=f"Fixed reorder point (default: {FIXED_S}).")
    parser.add_argument("--S", type=int, default=None, help=f"Fixed order-up-to level (default: {FIXED_CAP_S}).")
    parser.add_argument("--s-min", type=int, default=None, help="Minimum sampled reorder point.")
    parser.add_argument("--s-max", type=int, default=None, help="Maximum sampled reorder point.")
    parser.add_argument("--S-min", type=int, default=None, help="Minimum sampled order-up-to level.")
    parser.add_argument("--S-max", type=int, default=None, help="Maximum sampled order-up-to level.")
    parser.add_argument("--inter-avg-scv-min", type=float, default=None, help="Minimum time-weighted average inter-demand SCV.")
    parser.add_argument("--inter-avg-scv-max", type=float, default=None, help="Maximum time-weighted average inter-demand SCV.")
    parser.add_argument("--lead-scv-min", type=float, default=None, help="Minimum lead-time SCV.")
    parser.add_argument("--lead-scv-max", type=float, default=None, help="Maximum lead-time SCV.")
    parser.add_argument(
        "--mean-ratio-min",
        type=float,
        default=None,
        help="Minimum average inter-demand mean divided by average lead-time mean; lead-time mean is 1 in controlled runs.",
    )
    parser.add_argument(
        "--mean-ratio-max",
        type=float,
        default=None,
        help="Maximum average inter-demand mean divided by average lead-time mean; lead-time mean is 1 in controlled runs.",
    )
    parser.add_argument("--control-max-tries", type=int, default=1000, help="Maximum attempts for constrained sampling.")
    parser.add_argument(
        "--partition-counts-csv",
        type=str,
        default=None,
        help="CSV with Test Set,D,L,rho,S,s,num_files columns. Enables underrepresented partition sampling.",
    )
    parser.add_argument(
        "--partition-count-threshold",
        type=int,
        default=100000,
        help="Only CSV rows with num_files below this value are sampled.",
    )
    parser.add_argument(
        "--partition-simulations",
        type=int,
        default=5000,
        help="Number of simulations to run in partition-sampling mode.",
    )
    parser.add_argument("--model-number", type=int, default=0, help="Model number token used in file names.")
    parser.add_argument(
        "--model-num",
        type=int,
        default=None,
        help="Optional explicit random-model id in [1,1000000]; if omitted sampled once per run.",
    )
    parser.add_argument(
        "--inv-dir",
        type=str,
        default=None,
        help="Directory to store inventory pickle files (default: <code_dir>/inv).",
    )
    parser.add_argument(
        "--order-dir",
        type=str,
        default=None,
        help="Directory to store order pickle files (default: <code_dir>/order).",
    )
    parser.add_argument(
        "--loss-dir",
        type=str,
        default=None,
        help="Directory to store loss pickle files (default: <code_dir>/loss).",
    )
    parser.add_argument(
        "--exp-compare",
        action="store_true",
        help="Use exponential demand/lead (rate=1) and compare simulation vs analytic for levels 0..S.",
    )
    parser.add_argument(
        "--exp-varying-compare",
        action="store_true",
        help=(
            "Use piecewise-exponential inter-demand rates (random changes over time) with exponential "
            "lead time (rate=1), and compare simulation vs analytic for levels 0..S."
        ),
    )
    parser.add_argument(
        "--dynamic-demand",
        action="store_true",
        help=(
            "Dynamic inter-demand mode: inter-demand PH/mean changes at random discrete epochs; "
            "lead-time mean is fixed at 1."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    control_ranges = _build_control_ranges(args)
    mode_count = int(args.exp_compare) + int(args.dynamic_demand) + int(args.exp_varying_compare)
    if mode_count > 1:
        raise ValueError("Use only one mode flag: --exp-compare, --exp-varying-compare, or --dynamic-demand.")
    if args.n_settings < 1:
        raise ValueError("--n-settings must be >= 1.")
    if args.level < 0 or args.level > 30:
        raise ValueError("level must be between 0 and 30.")
    if args.partition_simulations < 1:
        raise ValueError("--partition-simulations must be >= 1.")
    if args.partition_count_threshold < 1:
        raise ValueError("--partition-count-threshold must be >= 1.")
    if (args.model_num is not None) and (not (1 <= args.model_num <= 1_000_000)):
        raise ValueError("--model-num must be in [1, 1000000].")
    if args.n_settings > 1 and args.model_num is not None:
        raise ValueError("--model-num is only valid for single-setting runs. Omit it when --n-settings>1.")
    if args.partition_counts_csv is not None and args.model_num is not None:
        raise ValueError("--model-num is not valid in partition-sampling mode.")
    if args.partition_counts_csv is not None and (args.exp_compare or args.exp_varying_compare):
        raise ValueError("Partition-sampling mode uses dynamic PH demand; do not combine it with exponential comparison modes.")

    print(
        "Running simulation with "
        f"inter_size={args.inter_size}, lead_size={args.lead_size}, "
        f"replications={args.replications}, horizon={args.horizon}, seed={args.seed}, "
        f"n_settings={args.n_settings}"
    )
    rng = np.random.default_rng(args.seed)
    meta_rng = np.random.default_rng(None if args.seed is None else args.seed + 9173)
    model_num = int(args.model_num) if args.model_num is not None else int(meta_rng.integers(1, 1_000_001))
    code_dir = Path(__file__).resolve().parent
    inv_dir = Path(args.inv_dir) if args.inv_dir else (code_dir / "inv")
    order_dir = Path(args.order_dir) if args.order_dir else (code_dir / "order")
    loss_dir = Path(args.loss_dir) if args.loss_dir else (code_dir / "loss")
    print(f"pickle dirs -> inv: {inv_dir} | order: {order_dir} | loss: {loss_dir}")

    if args.partition_counts_csv is not None:
        _run_partition_sampled_simulations(args, inv_dir=inv_dir, order_dir=order_dir, loss_dir=loss_dir)
        return

    _print_control_summary(control_ranges)

    if args.n_settings > 1:
        saved_triplets: list[tuple[Path, Path, Path]] = []
        for idx in range(args.n_settings):
            s, S = _sample_cli_policy(rng, control_ranges)
            rep_seed = None if args.seed is None else int(rng.integers(0, 2**31 - 1))
            mode_name = "baseline-ph"

            if args.dynamic_demand:
                mode_name = "dynamic-demand-ph"
                x, inv, orders, lost, sample_plan = simulate_dynamic_demand_setting(
                    inter_size=args.inter_size,
                    lead_size=args.lead_size,
                    s=s,
                    S=S,
                    n_replications=args.replications,
                    horizon=args.horizon,
                    seed=rep_seed,
                    control_ranges=control_ranges,
                )
                number_demand_rates = int(len(sample_plan.means))
            elif args.exp_varying_compare:
                mode_name = "exp-varying-compare"
                x, sim_inv, orders, lost, _analytic_inv, plan = simulate_exponential_time_varying_with_analytic(
                    s=s,
                    S=S,
                    n_replications=args.replications,
                    horizon=args.horizon,
                    seed=rep_seed,
                    lead_rate=1.0,
                )
                inv = sim_inv
                number_demand_rates = int(len(plan.means))
            elif args.exp_compare:
                mode_name = "exp-compare"
                x, sim_inv, orders, lost, _analytic_inv = simulate_exponential_with_analytic(
                    s=s,
                    S=S,
                    n_replications=args.replications,
                    horizon=args.horizon,
                    seed=rep_seed,
                    demand_rate=1.0,
                    lead_rate=1.0,
                )
                inv = sim_inv
                number_demand_rates = 1
            else:
                if control_ranges is None:
                    inter_size_sample = _sample_ph_size(args.inter_size, rng)
                    inter = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
                    lead_mean = float(rng.uniform(0.1, 10.0))
                    lead_size_sample = _sample_ph_size(args.lead_size, rng)
                    lead = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)
                else:
                    _x, inter, lead, s, S = generate_controlled_setting(
                        inter_size=args.inter_size,
                        lead_size=args.lead_size,
                        rng=rng,
                        control_ranges=control_ranges,
                    )
                x, inv, orders, lost = simulate_given_setting(
                    inter_demand_ph=inter,
                    lead_time_ph=lead,
                    s=s,
                    S=S,
                    n_replications=args.replications,
                    horizon=args.horizon,
                    seed=rep_seed,
                )
                number_demand_rates = 1

            scv_leadtime = lead_scv_from_input_vector(x)
            avg_inter_scv = average_inter_scv_from_input_vector(x)
            avg_inter_mean = average_inter_mean_from_input_vector(x)
            lead_mean = float(np.exp(x[0, 10]))
            mean_ratio = avg_inter_mean / lead_mean
            model_num_i = int(meta_rng.integers(1, 1_000_001))

            inv_path, order_path, loss_path = save_io_pickles(
                x=x,
                inv=inv,
                order=orders,
                loss=lost,
                scv_leadtime=scv_leadtime,
                number_demand_rates=number_demand_rates,
                model_number=args.model_number,
                S=S,
                s=s,
                model_num=model_num_i,
                inv_dir=inv_dir,
                order_dir=order_dir,
                loss_dir=loss_dir,
            )
            saved_triplets.append((inv_path, order_path, loss_path))
            print(
                f"[setting {idx + 1}/{args.n_settings}] saved -> "
                f"mode={mode_name}, s={s}, S={S}, model_num={model_num_i}, "
                f"n_demand_rates={number_demand_rates}, "
                f"avg_inter_scv={avg_inter_scv:.6g}, lead_scv={scv_leadtime:.6g}, "
                f"mean_ratio={mean_ratio:.6g}"
            )
            print(f"  inv: {inv_path}")
            print(f"  order: {order_path}")
            print(f"  loss: {loss_path}")

        print(f"Completed {len(saved_triplets)} settings and saved all pickle triplets.")
        return

    if args.exp_varying_compare:
        s_fixed, S_fixed = _sample_cli_policy(rng, control_ranges)
        print(
            "exponential varying-rate comparison mode: lead_rate=1, "
            f"random demand-rate changes over time, s={s_fixed}, S={S_fixed}"
        )
        x, sim_inv, orders, lost, analytic_inv, plan = simulate_exponential_time_varying_with_analytic(
            s=s_fixed,
            S=S_fixed,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
            lead_rate=1.0,
        )
        rates = 1.0 / np.array(plan.means, dtype=float)
        print("input matrix shape:", x.shape)
        print("simulation inventory distribution shape:", sim_inv.shape)
        print("analytic inventory distribution shape:", analytic_inv.shape)
        print("avg orders shape:", orders.shape)
        print("avg lost-sales shape:", lost.shape)
        print("change points:", plan.change_points.tolist())
        print("segment means:", np.round(plan.means, 4).tolist())
        print("segment rates:", np.round(rates, 4).tolist())

        max_abs_diff = np.max(np.abs(sim_inv[:, : S_fixed + 1] - analytic_inv[:, : S_fixed + 1]))
        mean_abs_diff = np.mean(np.abs(sim_inv[:, : S_fixed + 1] - analytic_inv[:, : S_fixed + 1]))
        print(f"max abs difference over levels 0..{S_fixed}: {max_abs_diff:.6f}")
        print(f"mean abs difference over levels 0..{S_fixed}: {mean_abs_diff:.6f}")

        scv_leadtime = lead_scv_from_input_vector(x)
        number_demand_rates = int(len(plan.means))
        inv_path, order_path, loss_path = save_io_pickles(
            x=x,
            inv=sim_inv,
            order=orders,
            loss=lost,
            scv_leadtime=scv_leadtime,
            number_demand_rates=number_demand_rates,
            model_number=args.model_number,
            S=S_fixed,
            s=s_fixed,
            model_num=model_num,
            inv_dir=inv_dir,
            order_dir=order_dir,
            loss_dir=loss_dir,
        )
        print(f"saved pickle (inv): {inv_path}")
        print(f"saved pickle (order): {order_path}")
        print(f"saved pickle (loss): {loss_path}")

        plot_simulation_vs_analytic_levels(
            simulation_inventory_distribution=sim_inv,
            analytic_inventory_distribution=analytic_inv,
            levels=tuple(range(S_fixed + 1)),
            change_points=plan.change_points,
            means=plan.means,
            save_path=args.compare_output,
            show=args.show,
        )
        print(f"saved comparison graph: {args.compare_output}")
        return

    if args.exp_compare:
        s_fixed, S_fixed = _sample_cli_policy(rng, control_ranges)
        print(f"exponential comparison mode: demand_rate=1, lead_rate=1, s={s_fixed}, S={S_fixed}")

        x, sim_inv, orders, lost, analytic_inv = simulate_exponential_with_analytic(
            s=s_fixed,
            S=S_fixed,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
            demand_rate=1.0,
            lead_rate=1.0,
        )

        print("input matrix shape:", x.shape)
        print("simulation inventory distribution shape:", sim_inv.shape)
        print("analytic inventory distribution shape:", analytic_inv.shape)
        print("avg orders shape:", orders.shape)
        print("avg lost-sales shape:", lost.shape)

        max_abs_diff = np.max(np.abs(sim_inv[:, : S_fixed + 1] - analytic_inv[:, : S_fixed + 1]))
        mean_abs_diff = np.mean(np.abs(sim_inv[:, : S_fixed + 1] - analytic_inv[:, : S_fixed + 1]))
        print(f"max abs difference over levels 0..{S_fixed}: {max_abs_diff:.6f}")
        print(f"mean abs difference over levels 0..{S_fixed}: {mean_abs_diff:.6f}")

        scv_leadtime = lead_scv_from_input_vector(x)
        number_demand_rates = 1
        inv_path, order_path, loss_path = save_io_pickles(
            x=x,
            inv=sim_inv,
            order=orders,
            loss=lost,
            scv_leadtime=scv_leadtime,
            number_demand_rates=number_demand_rates,
            model_number=args.model_number,
            S=S_fixed,
            s=s_fixed,
            model_num=model_num,
            inv_dir=inv_dir,
            order_dir=order_dir,
            loss_dir=loss_dir,
        )
        print(f"saved pickle (inv): {inv_path}")
        print(f"saved pickle (order): {order_path}")
        print(f"saved pickle (loss): {loss_path}")

        plot_simulation_vs_analytic_levels(
            simulation_inventory_distribution=sim_inv,
            analytic_inventory_distribution=analytic_inv,
            levels=tuple(range(S_fixed + 1)),
            save_path=args.compare_output,
            show=args.show,
        )
        print(f"saved comparison graph: {args.compare_output}")
        return

    if args.dynamic_demand:
        s_fixed, S_fixed = _sample_cli_policy(rng, control_ranges)
        print(
            "dynamic-demand mode: lead-time mean fixed at 1; "
            f"random demand changes (2..10 points, min-gap=5), s={s_fixed}, S={S_fixed}"
        )
        x, inv, orders, lost, sample_plan = simulate_dynamic_demand_setting(
            inter_size=args.inter_size,
            lead_size=args.lead_size,
            s=s_fixed,
            S=S_fixed,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
            control_ranges=control_ranges,
        )
        print("input matrix shape:", x.shape)
        print("inventory distribution shape:", inv.shape)
        print("avg orders shape:", orders.shape)
        print("avg lost-sales shape:", lost.shape)
        print("sample change points:", sample_plan.change_points.tolist())
        print("sample inter-demand means:", np.round(sample_plan.means, 4).tolist())
        print(
            "sample equivalent rates:",
            np.round(1.0 / np.array(sample_plan.means, dtype=float), 4).tolist(),
        )
        print(f"average inter-demand SCV: {average_inter_scv_from_input_vector(x):.6g}")
        print(f"lead-time SCV: {lead_scv_from_input_vector(x):.6g}")
        print(
            "average mean inter-demand / average lead time: "
            f"{average_inter_mean_from_input_vector(x) / float(np.exp(x[0, 10])):.6g}"
        )

        scv_leadtime = lead_scv_from_input_vector(x)
        number_demand_rates = int(len(sample_plan.means))
        inv_path, order_path, loss_path = save_io_pickles(
            x=x,
            inv=inv,
            order=orders,
            loss=lost,
            scv_leadtime=scv_leadtime,
            number_demand_rates=number_demand_rates,
            model_number=args.model_number,
            S=S_fixed,
            s=s_fixed,
            model_num=model_num,
            inv_dir=inv_dir,
            order_dir=order_dir,
            loss_dir=loss_dir,
        )
        print(f"saved pickle (inv): {inv_path}")
        print(f"saved pickle (order): {order_path}")
        print(f"saved pickle (loss): {loss_path}")

        plot_inventory_probabilities_0_to_S_with_changes(
            inventory_distribution=inv,
            S=S_fixed,
            change_points=sample_plan.change_points,
            means=sample_plan.means,
            save_path=args.output,
            show=args.show,
        )
        print(f"saved graph: {args.output}")
        return

    if control_ranges is None:
        inter_size_sample = _sample_ph_size(args.inter_size, rng)
        inter = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
        lead_mean = float(rng.uniform(0.1, 10.0))
        lead_size_sample = _sample_ph_size(args.lead_size, rng)
        lead = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)
        x, inv, orders, lost = simulate_given_setting(
            inter_demand_ph=inter,
            lead_time_ph=lead,
            s=FIXED_S,
            S=FIXED_CAP_S,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
        )
    else:
        _x, inter, lead, s_controlled, S_controlled = generate_controlled_setting(
            inter_size=args.inter_size,
            lead_size=args.lead_size,
            rng=rng,
            control_ranges=control_ranges,
        )
        x, inv, orders, lost = simulate_given_setting(
            inter_demand_ph=inter,
            lead_time_ph=lead,
            s=s_controlled,
            S=S_controlled,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
        )

    print("input matrix shape:", x.shape)
    print("inventory distribution shape:", inv.shape)
    print("avg orders shape:", orders.shape)
    print("avg lost-sales shape:", lost.shape)
    s = int(round(float(x[0, S_INPUT_COL])))
    S = int(round(float(x[0, CAP_S_INPUT_COL])))
    print(f"sampled policy: s={s}, S={S}")
    print(f"average inter-demand SCV: {average_inter_scv_from_input_vector(x):.6g}")
    print(f"lead-time SCV: {lead_scv_from_input_vector(x):.6g}")
    print(
        "average mean inter-demand / average lead time: "
        f"{average_inter_mean_from_input_vector(x) / float(np.exp(x[0, 10])):.6g}"
    )
    if args.level > S:
        print(
            f"note: requested level={args.level} is above S={S}, "
            "so probability can be zero at all times."
        )
    print(f"first 5 probabilities for inventory={args.level}:", inv[:5, args.level])

    scv_leadtime = lead_scv_from_input_vector(x)
    number_demand_rates = 1
    inv_path, order_path, loss_path = save_io_pickles(
        x=x,
        inv=inv,
        order=orders,
        loss=lost,
        scv_leadtime=scv_leadtime,
        number_demand_rates=number_demand_rates,
        model_number=args.model_number,
        S=S,
        s=s,
        model_num=model_num,
        inv_dir=inv_dir,
        order_dir=order_dir,
        loss_dir=loss_dir,
    )
    print(f"saved pickle (inv): {inv_path}")
    print(f"saved pickle (order): {order_path}")
    print(f"saved pickle (loss): {loss_path}")

    plot_inventory_level_probability(
        inv,
        level=args.level,
        save_path=args.output,
        show=args.show,
    )
    print(f"saved graph: {args.output}")


if __name__ == "__main__":
    main()

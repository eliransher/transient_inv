"""SimPy simulation for a lost-sales (s, S) inventory model with PH demand/lead times.

The main entry-point for one setting is:
    simulate_single_setting(...)
which returns:
    - input_vector: shape (100, 22)
    - inventory_distribution: shape (100, 31)
    - avg_orders_so_far: shape (100,)
    - avg_lost_sales_so_far: shape (100,)
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import simpy


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


def generate_random_ph_wide(
    size: int,
    target_mean: float,
    rng: np.random.Generator,
    max_tries: int = 200,
    max_scv: float = 20.0,
) -> PHDistribution:
    """Wide PH generator aligned with ph_summary_table.py family mix."""
    if size <= 0:
        raise ValueError("size must be positive.")
    if target_mean <= 0:
        raise ValueError("target_mean must be positive.")
    if max_scv <= 0:
        raise ValueError("max_scv must be positive.")

    families = ("base", "erlang", "hyperexp", "hyperexp_ultra", "coxian", "coxian_extreme")
    probs = np.array([0.20, 0.14, 0.18, 0.16, 0.16, 0.16], dtype=float)

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
                and scv <= max_scv
            ):
                return ph
        except np.linalg.LinAlgError:
            continue
        except FloatingPointError:
            continue

    fallback = generate_random_ph(size=size, target_mean=target_mean, rng=rng)
    if _scv_from_moments(fallback.moments) <= max_scv:
        return fallback
    # Guaranteed low-SCV fallback.
    return _gen_erlang_like_ph(size=size, target_mean=target_mean)


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
    """Sample unique (s, S) pairs with min_S <= S <= max_S and 1 <= s <= S."""
    candidates = [(s, S) for S in range(min_S, max_S + 1) for s in range(1, S + 1)]
    if n_policies > len(candidates):
        raise ValueError(
            f"Requested {n_policies} unique policies, but only {len(candidates)} available "
            f"under constraints {min_S}<=S<={max_S}, 1<=s<=S."
        )
    idx = rng.choice(len(candidates), size=n_policies, replace=False)
    return [candidates[int(i)] for i in idx]


def designated_ph_generator(
    size: int,
    rng: np.random.Generator,
    target_mean: float = 1.0,
    max_scv: float = 20.0,
) -> PHDistribution:
    """Designated PH-generation function (size-driven API)."""
    return generate_random_ph_wide(size=size, target_mean=target_mean, rng=rng, max_scv=max_scv)


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
) -> DynamicDemandPlan:
    """Generate dynamic inter-demand PHs and random means in (0.1, 10)."""
    if min_changes < 1 or max_changes < min_changes:
        raise ValueError("Need 1 <= min_changes <= max_changes.")

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
        seg_size = _sample_ph_size(inter_size, rng)
        phs.append(designated_ph_generator(size=seg_size, target_mean=m, rng=rng))
        prev_mean = m

    return DynamicDemandPlan(
        change_points=change_points,
        means=means,
        phs=tuple(phs),
    )


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DynamicDemandPlan]:
    """Simulate with dynamic inter-demand PH and fixed lead-time mean=1."""
    rng = np.random.default_rng(seed)
    lead_size_sample = _sample_ph_size(lead_size, rng)
    lead_time_ph = designated_ph_generator(size=lead_size_sample, target_mean=1.0, rng=rng)
    demand_plan = generate_dynamic_demand_plan(
        inter_size=inter_size,
        horizon=horizon,
        rng=rng,
        min_changes=2,
        max_changes=10,
        min_gap=5,
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
    """Build one input row: [log(10 inter moments), log(10 lead moments), s, S]."""
    eps = 1e-300
    inter_log = np.log(np.maximum(inter_demand_ph.moments, eps))
    lead_log = np.log(np.maximum(lead_time_ph.moments, eps))
    return np.concatenate(
        [
            inter_log,
            lead_log,
            np.array([float(s), float(S)], dtype=float),
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
    """Build time-indexed input matrix x with shape (horizon, 22)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if (inter_demand_ph is None) and (demand_plan is None):
        raise ValueError("Provide either inter_demand_ph or demand_plan.")
    if (inter_demand_ph is not None) and (demand_plan is not None):
        raise ValueError("Provide only one of inter_demand_ph or demand_plan.")

    x = np.zeros((horizon, 22), dtype=float)
    if inter_demand_ph is not None:
        row = build_input_vector(inter_demand_ph, lead_time_ph, s=s, S=S)
        x[:] = row
        return x

    cps = np.array(demand_plan.change_points, dtype=int)
    seg_idx = 0
    for t in range(1, horizon + 1):
        while seg_idx < cps.size and t >= cps[seg_idx]:
            seg_idx += 1
        row = build_input_vector(demand_plan.phs[seg_idx], lead_time_ph, s=s, S=S)
        x[t - 1] = row
    return x


def lead_scv_from_input_vector(x: np.ndarray) -> float:
    """Extract lead-time SCV from x (supports shape (22,) or (T,22), with log-moments)."""
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
        input_vector: (100, 22)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset across multiple random settings.

    Returns:
        X_inputs: (n_settings, 100, 22)
        Y_inventory: (n_settings, 100, 31)
        Y_orders: (n_settings, 100)
        Y_lost_sales: (n_settings, 100)
    """
    master = np.random.SeedSequence(seed)
    setting_seeds = master.spawn(n_settings)

    X_inputs = np.zeros((n_settings, horizon, 22), dtype=float)
    Y_inventory = np.zeros((n_settings, horizon, 31), dtype=float)
    Y_orders = np.zeros((n_settings, horizon), dtype=float)
    Y_lost = np.zeros((n_settings, horizon), dtype=float)
    policy_rng = np.random.default_rng(seed)
    unique_policies = _sample_unique_policies(n_settings, policy_rng)

    for idx, ss in enumerate(setting_seeds):
        rng = np.random.default_rng(ss)
        inter_size_sample = _sample_ph_size(inter_size, rng)
        inter_demand_ph = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
        lead_mean = float(rng.uniform(0.1, 10.0))
        lead_size_sample = _sample_ph_size(lead_size, rng)
        lead_time_ph = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)
        s, S = unique_policies[idx]
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
    parser.add_argument("--s", type=int, default=None, help="Optional fixed reorder point.")
    parser.add_argument("--S", type=int, default=None, help="Optional fixed order-up-to level.")
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
    mode_count = int(args.exp_compare) + int(args.dynamic_demand) + int(args.exp_varying_compare)
    if mode_count > 1:
        raise ValueError("Use only one mode flag: --exp-compare, --exp-varying-compare, or --dynamic-demand.")
    if args.n_settings < 1:
        raise ValueError("--n-settings must be >= 1.")
    if args.level < 0 or args.level > 30:
        raise ValueError("level must be between 0 and 30.")
    if (args.s is None) ^ (args.S is None):
        raise ValueError("Provide both --s and --S together, or neither.")
    if args.n_settings > 1 and (args.s is not None or args.S is not None):
        raise ValueError("--n-settings>1 expects fully random (s,S); do not pass --s/--S.")
    if args.S is not None:
        if not (5 <= args.S <= 30):
            raise ValueError("--S must be in [5, 30].")
        if not (1 <= args.s <= args.S):
            raise ValueError("--s must be in [1, S].")
    if (args.model_num is not None) and (not (1 <= args.model_num <= 1_000_000)):
        raise ValueError("--model-num must be in [1, 1000000].")
    if args.n_settings > 1 and args.model_num is not None:
        raise ValueError("--model-num is only valid for single-setting runs. Omit it when --n-settings>1.")

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

    if args.n_settings > 1:
        unique_policies = _sample_unique_policies(args.n_settings, rng)
        saved_triplets: list[tuple[Path, Path, Path]] = []
        for idx, (s, S) in enumerate(unique_policies):
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
                inter_size_sample = _sample_ph_size(args.inter_size, rng)
                inter = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
                lead_mean = float(rng.uniform(0.1, 10.0))
                lead_size_sample = _sample_ph_size(args.lead_size, rng)
                lead = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)
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
                f"n_demand_rates={number_demand_rates}"
            )
            print(f"  inv: {inv_path}")
            print(f"  order: {order_path}")
            print(f"  loss: {loss_path}")

        print(f"Completed {len(saved_triplets)} settings and saved all pickle triplets.")
        return

    if args.exp_varying_compare:
        s_fixed = args.s if args.s is not None else 5
        S_fixed = args.S if args.S is not None else 15
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
        s_fixed = args.s if args.s is not None else 5
        S_fixed = args.S if args.S is not None else 10
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
        s_fixed = args.s if args.s is not None else 5
        S_fixed = args.S if args.S is not None else 15
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

    if args.S is None:
        x, inv, orders, lost = simulate_single_setting(
            inter_size=args.inter_size,
            lead_size=args.lead_size,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
        )
    else:
        inter_size_sample = _sample_ph_size(args.inter_size, rng)
        inter = designated_ph_generator(size=inter_size_sample, target_mean=1.0, rng=rng)
        lead_mean = float(rng.uniform(0.1, 10.0))
        lead_size_sample = _sample_ph_size(args.lead_size, rng)
        lead = designated_ph_generator(size=lead_size_sample, target_mean=lead_mean, rng=rng)
        x, inv, orders, lost = simulate_given_setting(
            inter_demand_ph=inter,
            lead_time_ph=lead,
            s=args.s,
            S=args.S,
            n_replications=args.replications,
            horizon=args.horizon,
            seed=args.seed,
        )

    print("input matrix shape:", x.shape)
    print("inventory distribution shape:", inv.shape)
    print("avg orders shape:", orders.shape)
    print("avg lost-sales shape:", lost.shape)
    s = int(round(float(x[0, -2])))
    S = int(round(float(x[0, -1])))
    print(f"sampled policy: s={s}, S={S}")
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

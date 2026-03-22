"""Generate a wide-scatter PH summary table and save as pickle.

By default this script mixes several PH families to widen shape statistics:
- base generator from inventory_simpy_ph.generate_random_ph
- Erlang-like (low SCV)
- hyperexponential-like (high SCV/heavy tails)
- Coxian-like (asymmetric/heavy-tailed variants)

It still supports a strict "base-only" mode to use only the current project
PH generator.
"""

from __future__ import annotations

import argparse
import pickle
from math import factorial
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from inventory_simpy_ph import generate_random_ph, ph_shape_statistics


def _compute_ph_moments(alpha: np.ndarray, T: np.ndarray, k_max: int = 10) -> np.ndarray:
    """Compute first k_max raw moments of PH(alpha, T)."""
    n = T.shape[0]
    one = np.ones(n, dtype=float)
    A = -T
    v = one.copy()
    out = np.zeros(k_max, dtype=float)
    for k in range(1, k_max + 1):
        v = np.linalg.solve(A, v)
        out[k - 1] = factorial(k) * float(alpha @ v)
    return out


def _scale_T_to_mean(alpha: np.ndarray, T: np.ndarray, target_mean: float) -> np.ndarray:
    m1 = _compute_ph_moments(alpha, T, k_max=1)[0]
    scale = float(m1) / float(target_mean)
    return T * scale


def _gen_erlang_like(size: int, target_mean: float) -> np.ndarray:
    alpha = np.zeros(size, dtype=float)
    alpha[0] = 1.0
    T = np.zeros((size, size), dtype=float)
    lam = 1.0
    for i in range(size):
        T[i, i] = -lam
        if i < size - 1:
            T[i, i + 1] = lam
    T = _scale_T_to_mean(alpha, T, target_mean=target_mean)
    return _compute_ph_moments(alpha, T, k_max=10)


def _gen_hyperexp_heavy(size: int, target_mean: float, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.dirichlet(np.full(size, 0.12, dtype=float))
    rates = np.exp(rng.uniform(np.log(0.01), np.log(120.0), size=size))

    # Force a rare, very-slow branch for heavy tails.
    slow_idx = int(np.argmin(rates))
    p_slow = float(rng.uniform(0.01, 0.18))
    alpha = (1.0 - p_slow) * alpha
    alpha[slow_idx] += p_slow
    alpha /= alpha.sum()

    T = -np.diag(rates)
    T = _scale_T_to_mean(alpha, T, target_mean=target_mean)
    return _compute_ph_moments(alpha, T, k_max=10)


def _gen_hyperexp_ultra(size: int, target_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Very heavy-tailed hyperexponential-like PH for extreme shape values."""
    alpha = rng.dirichlet(np.full(size, 0.08, dtype=float))
    rates = np.exp(rng.uniform(np.log(0.002), np.log(250.0), size=size))

    slow_idx = int(np.argmin(rates))
    rates[slow_idx] *= float(rng.uniform(0.002, 0.04))

    p_slow = float(rng.uniform(0.002, 0.12))
    alpha = (1.0 - p_slow) * alpha
    alpha[slow_idx] += p_slow
    alpha /= alpha.sum()

    T = -np.diag(rates)
    T = _scale_T_to_mean(alpha, T, target_mean=target_mean)
    return _compute_ph_moments(alpha, T, k_max=10)


def _gen_coxian_like(size: int, target_mean: float, rng: np.random.Generator) -> np.ndarray:
    alpha = np.zeros(size, dtype=float)
    alpha[0] = 1.0
    rates = np.exp(rng.uniform(np.log(0.02), np.log(80.0), size=size))

    T = np.zeros((size, size), dtype=float)
    mode = str(rng.choice(["balanced", "tail-switch"], p=[0.55, 0.45]))
    if size <= 1:
        tail_start = 1
    else:
        low = max(1, size // 4)
        if low >= size:
            low = size - 1
        tail_start = int(rng.integers(low, size))
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
                    # Rare entry into a very slow tail chain.
                    cont = float(rng.uniform(0.003, 0.09))
                else:
                    cont = float(rng.uniform(0.985, 0.9999))
            T[i, i + 1] = lam * cont
        T[i, i] = -lam

    T = _scale_T_to_mean(alpha, T, target_mean=target_mean)
    return _compute_ph_moments(alpha, T, k_max=10)


def _gen_coxian_extreme(size: int, target_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Extreme heavy-tailed Coxian-like PH to increase skewness spread."""
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
                # Very small probability to enter extremely slow tail phases.
                cont = float(rng.uniform(0.001, 0.04))
            else:
                cont = float(rng.uniform(0.992, 0.99995))
            T[i, i + 1] = lam * cont
        T[i, i] = -lam

    T = _scale_T_to_mean(alpha, T, target_mean=target_mean)
    return _compute_ph_moments(alpha, T, k_max=10)


def _sample_moments_wide(
    size: int,
    target_mean: float,
    rng: np.random.Generator,
    forced_family: str | None = None,
    max_tries: int = 200,
) -> Tuple[np.ndarray, str]:
    """Sample PH moments from a diversified family mix."""
    families = ("base", "erlang", "hyperexp", "hyperexp_ultra", "coxian", "coxian_extreme")
    probs = np.array([0.20, 0.14, 0.18, 0.16, 0.16, 0.16], dtype=float)

    for _ in range(max_tries):
        fam = forced_family if forced_family is not None else str(rng.choice(families, p=probs))
        try:
            if fam == "base":
                moments = generate_random_ph(size=size, target_mean=target_mean, rng=rng).moments
            elif fam == "erlang":
                moments = _gen_erlang_like(size=size, target_mean=target_mean)
            elif fam == "hyperexp":
                moments = _gen_hyperexp_heavy(size=size, target_mean=target_mean, rng=rng)
            elif fam == "hyperexp_ultra":
                moments = _gen_hyperexp_ultra(size=size, target_mean=target_mean, rng=rng)
            elif fam == "coxian_extreme":
                moments = _gen_coxian_extreme(size=size, target_mean=target_mean, rng=rng)
            else:
                moments = _gen_coxian_like(size=size, target_mean=target_mean, rng=rng)

            if np.all(np.isfinite(moments)) and np.all(moments > 0) and np.max(moments) < 1e300:
                return moments, fam
        except np.linalg.LinAlgError:
            continue
        except FloatingPointError:
            continue

    # Fallback to base generator if diversified attempts fail.
    moments = generate_random_ph(size=size, target_mean=target_mean, rng=rng).moments
    return moments, "base_fallback"


def build_summary_table(
    n_ph: int = 2000,
    min_size: int = 1,
    max_size: int = 100,
    mean_value: float = 1.0,
    seed: int = 123,
    mode: str = "wide",
):
    """Return summary table and ranges."""
    if n_ph <= 0:
        raise ValueError("n_ph must be positive.")
    if min_size <= 0 or max_size < min_size:
        raise ValueError("Need positive sizes with max_size >= min_size.")
    if mean_value <= 0:
        raise ValueError("mean_value must be positive.")
    if mode not in {"wide", "base-only"}:
        raise ValueError("mode must be 'wide' or 'base-only'.")

    rng = np.random.default_rng(seed)
    sizes = rng.integers(min_size, max_size + 1, size=n_ph)
    family_schedule = None
    if mode == "wide":
        # Ensure each batch includes low/medium/high-variability PH families.
        base_sched = np.array(
            ["erlang", "base", "coxian", "hyperexp", "coxian_extreme", "hyperexp_ultra"],
            dtype=object,
        )
        reps = int(np.ceil(n_ph / base_sched.size))
        family_schedule = np.tile(base_sched, reps)[:n_ph]
        rng.shuffle(family_schedule)

    table = []
    scv_vals = []
    skew_vals = []
    kurt_vals = []
    fam_counts: Dict[str, int] = {}

    for idx, size in enumerate(sizes, start=1):
        if mode == "base-only":
            ph = generate_random_ph(size=int(size), target_mean=mean_value, rng=rng)
            moments = ph.moments
            family = "base"
        else:
            moments, family = _sample_moments_wide(
                size=int(size),
                target_mean=mean_value,
                rng=rng,
                forced_family=str(family_schedule[idx - 1]),
            )

        scv, skew, kurt = ph_shape_statistics(moments)
        row = {
            "ph_id": idx,
            "size": int(size),
            "family": family,
            "scv": float(scv),
            "skewness": float(skew),
            "kurtosis": float(kurt),
        }
        table.append(row)
        scv_vals.append(scv)
        skew_vals.append(skew)
        kurt_vals.append(kurt)
        fam_counts[family] = fam_counts.get(family, 0) + 1

    ranges = {
        "scv_range": (float(np.min(scv_vals)), float(np.max(scv_vals))),
        "skewness_range": (float(np.min(skew_vals)), float(np.max(skew_vals))),
        "kurtosis_range": (float(np.min(kurt_vals)), float(np.max(kurt_vals))),
    }
    return table, ranges, fam_counts


def save_scatter_plots(
    table,
    skew_output: Path,
    kurt_output: Path,
    use_log_scale: bool = False,
) -> None:
    """Save skewness-vs-SCV and kurtosis-vs-SCV scatter plots."""
    scv = np.array([row["scv"] for row in table], dtype=float)
    skew = np.array([row["skewness"] for row in table], dtype=float)
    kurt = np.array([row["kurtosis"] for row in table], dtype=float)
    fam = [row["family"] for row in table]

    families = sorted(set(fam))
    cmap = plt.cm.get_cmap("tab10", max(1, len(families)))
    color_of = {f: cmap(i) for i, f in enumerate(families)}
    colors = [color_of[f] for f in fam]

    # Skewness vs SCV
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(scv, skew, c=colors, s=16, alpha=0.75, edgecolors="none")
    if use_log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.set_xlabel("SCV")
    ax1.set_ylabel("Skewness")
    ax1.set_title(
        "Skewness as a Function of SCV (log-log)"
        if use_log_scale
        else "Skewness as a Function of SCV"
    )
    if use_log_scale:
        ax1.set_xlim(1e-3, 100)
        ax1.set_ylim(1e-3, 200)
    else:
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 200)
    ax1.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_of[f], markersize=7, label=f)
        for f in families
    ]
    ax1.legend(handles=handles, title="Family", loc="best", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(skew_output, dpi=170, bbox_inches="tight")
    plt.close(fig1)

    # Kurtosis vs SCV
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(scv, kurt, c=colors, s=16, alpha=0.75, edgecolors="none")
    if use_log_scale:
        ax2.set_xscale("log")
        ax2.set_yscale("log")
    ax2.set_xlabel("SCV")
    ax2.set_ylabel("Kurtosis")
    ax2.set_title(
        "Kurtosis as a Function of SCV (log-log)"
        if use_log_scale
        else "Kurtosis as a Function of SCV"
    )
    if use_log_scale:
        ax2.set_xlim(1e-3, 100)
        ax2.set_ylim(1e-3, 500)
    else:
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 500)
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=handles, title="Family", loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(kurt_output, dpi=170, bbox_inches="tight")
    plt.close(fig2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PH summary table and save as pickle.")
    parser.add_argument("--n-ph", type=int, default=2000, help="Number of PH distributions.")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum PH size (inclusive).")
    parser.add_argument("--max-size", type=int, default=100, help="Maximum PH size (inclusive).")
    parser.add_argument("--mean", type=float, default=1.0, help="Target mean for generated PHs.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["wide", "base-only"],
        default="wide",
        help="Use 'wide' for broad scatter or 'base-only' for original generator only.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ph_summary_table.pkl",
        help="Output pickle path for the summary table.",
    )
    parser.add_argument(
        "--skew-plot",
        type=str,
        default="skewness_vs_scv.png",
        help="Output image path for skewness-vs-SCV scatter.",
    )
    parser.add_argument(
        "--kurt-plot",
        type=str,
        default="kurtosis_vs_scv.png",
        help="Output image path for kurtosis-vs-SCV scatter.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log-log scale for both scatter plots (default: linear scale).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table, ranges, fam_counts = build_summary_table(
        n_ph=args.n_ph,
        min_size=args.min_size,
        max_size=args.max_size,
        mean_value=args.mean,
        seed=args.seed,
        mode=args.mode,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(table, f)

    skew_path = Path(args.skew_plot).resolve()
    kurt_path = Path(args.kurt_plot).resolve()
    skew_path.parent.mkdir(parents=True, exist_ok=True)
    kurt_path.parent.mkdir(parents=True, exist_ok=True)
    save_scatter_plots(
        table=table,
        skew_output=skew_path,
        kurt_output=kurt_path,
        use_log_scale=args.log_scale,
    )

    print(f"Mode: {args.mode}")
    print(f"Generated {len(table)} PH distributions.")
    print(f"Saved summary table pickle to: {out_path}")
    print(f"Saved skewness-vs-SCV scatter to: {skew_path}")
    print(f"Saved kurtosis-vs-SCV scatter to: {kurt_path}")
    print(f"Plot scale: {'log-log' if args.log_scale else 'linear'}")
    print(f"SCV range: {ranges['scv_range'][0]:.6f} to {ranges['scv_range'][1]:.6f}")
    print(f"Skewness range: {ranges['skewness_range'][0]:.6f} to {ranges['skewness_range'][1]:.6f}")
    print(f"Kurtosis range: {ranges['kurtosis_range'][0]:.6f} to {ranges['kurtosis_range'][1]:.6f}")
    print("Family counts:", fam_counts)


if __name__ == "__main__":
    main()

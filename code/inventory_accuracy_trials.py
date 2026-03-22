"""Repeatability/accuracy trial for inventory simulation results.

For one fixed setting (same inter-demand PH, lead-time PH, s, S), this script:
1) Runs the simulation multiple times (default: 10 trials).
2) Computes mean inventory level over time for each trial.
3) Computes relative error (%) of each trial from the across-trials mean.
4) Computes 95% confidence limits (CL) of the mean inventory across trials.

Output:
- A pandas DataFrame with `horizon` rows (default 100), one row per time epoch.
- Saved as both pickle and CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from inventory_simpy_ph import designated_ph_generator, simulate_given_setting


def _render_progress(current: int, total: int, width: int = 36) -> None:
    """Render a simple in-place progress bar."""
    frac = 0.0 if total <= 0 else float(current) / float(total)
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * max(0, width - filled)
    print(f"\rTrials progress: [{bar}] {current}/{total} ({100.0 * frac:5.1f}%)", end="", flush=True)
    if current >= total:
        print("")


def _format_float_for_filename(value: float) -> str:
    token = f"{value:.6f}".rstrip("0").rstrip(".")
    if token == "":
        token = "0"
    return token.replace("-", "m")


def _sample_setting(
    rng: np.random.Generator,
    inter_size_max: int,
    lead_size_max: int,
    s: Optional[int],
    S: Optional[int],
):
    """Sample one fixed model setting."""
    if inter_size_max < 1 or lead_size_max < 1:
        raise ValueError("inter_size_max and lead_size_max must be >= 1.")

    if (s is None) ^ (S is None):
        raise ValueError("Provide both s and S together, or neither.")

    if S is None:
        S_val = int(rng.integers(5, 31))
        s_val = int(rng.integers(1, S_val + 1))
    else:
        if not (5 <= S <= 30):
            raise ValueError("S must be in [5, 30].")
        if not (1 <= s <= S):
            raise ValueError("s must satisfy 1 <= s <= S.")
        S_val = int(S)
        s_val = int(s)

    inter_size = int(rng.integers(1, inter_size_max + 1))
    lead_size = int(rng.integers(1, lead_size_max + 1))
    inter_ph = designated_ph_generator(size=inter_size, target_mean=1.0, rng=rng)
    lead_mean = float(rng.uniform(0.1, 10.0))
    lead_ph = designated_ph_generator(size=lead_size, target_mean=lead_mean, rng=rng)
    m1 = float(lead_ph.moments[0])
    m2 = float(lead_ph.moments[1])
    var = max(0.0, m2 - m1 * m1)
    lead_scv = 0.0 if m1 <= 0 else var / (m1 * m1)

    return inter_ph, lead_ph, s_val, S_val, inter_size, lead_size, lead_mean, lead_scv


def _compute_mean_inventory_from_dist(inv_dist: np.ndarray) -> np.ndarray:
    """Compute E[I(t)] from inventory distribution over levels 0..30."""
    levels = np.arange(inv_dist.shape[1], dtype=float)
    return inv_dist @ levels


def run_accuracy_trial(
    trials: int = 10,
    replications: int = 50000,
    horizon: int = 100,
    seed: Optional[int] = None,
    inter_size_max: int = 100,
    lead_size_max: int = 100,
    s: Optional[int] = None,
    S: Optional[int] = None,
    show_progress: bool = True,
    model_num: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Run repeated simulations for one fixed setting and return DataFrame + metadata."""
    if trials < 2:
        raise ValueError("trials must be >= 2 for relative-error/CL analysis.")

    rng = np.random.default_rng(seed)
    inter_ph, lead_ph, s_val, S_val, inter_size, lead_size, lead_mean, lead_scv = _sample_setting(
        rng=rng,
        inter_size_max=inter_size_max,
        lead_size_max=lead_size_max,
        s=s,
        S=S,
    )
    if model_num is None:
        model_num_val = int(rng.integers(1, 100001))
    else:
        if not (1 <= int(model_num) <= 100000):
            raise ValueError("model_num must be in [1, 100000].")
        model_num_val = int(model_num)

    mean_inventory_trials = np.zeros((trials, horizon), dtype=float)
    trial_seeds = (
        [None] * trials
        if seed is None
        else [int(x) for x in rng.integers(0, 2**31 - 1, size=trials)]
    )

    if show_progress:
        _render_progress(0, trials)
    for j in range(trials):
        _, inv_dist, _, _ = simulate_given_setting(
            inter_demand_ph=inter_ph,
            lead_time_ph=lead_ph,
            s=s_val,
            S=S_val,
            n_replications=replications,
            horizon=horizon,
            seed=trial_seeds[j],
        )
        mean_inventory_trials[j] = _compute_mean_inventory_from_dist(inv_dist)
        if show_progress:
            _render_progress(j + 1, trials)

    avg_mean_inventory = mean_inventory_trials.mean(axis=0)
    eps = 1e-12
    rel_err_pct = (
        np.abs(mean_inventory_trials - avg_mean_inventory[None, :])
        / np.maximum(np.abs(avg_mean_inventory[None, :]), eps)
        * 100.0
    )

    std_t = mean_inventory_trials.std(axis=0, ddof=1)
    # 95% CL with t-critical for df=trials-1.
    # For trials=10 (default), t_0.975,9 ~= 2.262.
    t_crit = 2.262 if trials == 10 else 1.96
    half_width = t_crit * std_t / np.sqrt(float(trials))
    ci_low = avg_mean_inventory - half_width
    ci_high = avg_mean_inventory + half_width

    data = {"epoch": np.arange(1, horizon + 1, dtype=int)}
    for j in range(trials):
        data[f"mean_inv_trial_{j + 1}"] = mean_inventory_trials[j]
    data["avg_mean_inv_all_trials"] = avg_mean_inventory
    for j in range(trials):
        data[f"rel_err_pct_trial_{j + 1}"] = rel_err_pct[j]
    data["ci95_low"] = ci_low
    data["ci95_high"] = ci_high

    df = pd.DataFrame(data)

    meta = {
        "trials": trials,
        "replications": replications,
        "horizon": horizon,
        "seed": seed,
        "s": s_val,
        "S": S_val,
        "inter_size": inter_size,
        "lead_size": lead_size,
        "lead_mean": lead_mean,
        "lead_scv": lead_scv,
        "model_num": model_num_val,
    }
    return df, meta


def _parse_args():
    p = argparse.ArgumentParser(description="Repeatability test for mean inventory trajectory.")
    p.add_argument("--trials", type=int, default=10, help="Number of repeated runs per setting.")
    p.add_argument("--replications", type=int, default=50000, help="Replications per trial.")
    p.add_argument("--horizon", type=int, default=100, help="Time horizon.")
    p.add_argument("--seed", type=int, default=None, help="Optional top-level seed.")
    p.add_argument("--inter-size-max", type=int, default=100, help="Max inter-demand PH size.")
    p.add_argument("--lead-size-max", type=int, default=100, help="Max lead-time PH size.")
    p.add_argument("--s", type=int, default=None, help="Optional fixed s (must provide S too).")
    p.add_argument("--S", type=int, default=None, help="Optional fixed S in [5,30].")
    p.add_argument(
        "--output-prefix",
        type=str,
        default="inventory_accuracy_trial",
        help="Optional prefix for auxiliary outputs.",
    )
    p.add_argument(
        "--dump-dir",
        "--output-dir",
        dest="dump_dir",
        type=str,
        default=".",
        help="Directory where results are dumped (pickle + csv).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable trial progress bar.",
    )
    p.add_argument(
        "--model-num",
        type=int,
        default=None,
        help="Optional model id in [1,100000]. If omitted, sampled uniformly.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    df, meta = run_accuracy_trial(
        trials=args.trials,
        replications=args.replications,
        horizon=args.horizon,
        seed=args.seed,
        inter_size_max=args.inter_size_max,
        lead_size_max=args.lead_size_max,
        s=args.s,
        S=args.S,
        show_progress=not args.no_progress,
        model_num=args.model_num,
    )

    out_dir = Path(args.dump_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scv_token = _format_float_for_filename(float(meta["lead_scv"]))
    base = f"inv_{int(meta['S'])}_{int(meta['s'])}_{int(meta['replications'])}_{scv_token}_{int(meta['model_num'])}"
    pkl_path = out_dir / f"{base}.pkl"
    csv_path = out_dir / f"{base}.csv"

    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)

    print("Finished repeatability trial.")
    print("Setting:", meta)
    print(f"Dump directory: {out_dir}")
    print(f"Saved DataFrame pickle: {pkl_path}")
    print(f"Saved DataFrame csv: {csv_path}")
    print("DataFrame shape:", df.shape)
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

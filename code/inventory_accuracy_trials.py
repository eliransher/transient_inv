"""Repeatability/accuracy trial for inventory/order/loss simulation results.

For one fixed setting (same inter-demand PH, lead-time PH, s, S), this script:
1) Runs the simulation multiple times (default: 10 trials).
2) Computes mean inventory, mean orders, and mean loss trajectories for each trial.
3) Computes relative error (%) of each trial from the across-trials mean.
4) Computes 95% confidence limits (CL) across trials.

Output:
- A pandas DataFrame with `horizon` rows (default 100), one row per time epoch.
- Produced for each metric (inv/order/loss) and saved as both pickle and CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

try:
    from inventory_simpy_ph import designated_ph_generator, simulate_given_setting
except ModuleNotFoundError:
    # Support running via compatibility launcher from project root.
    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
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


def _build_metric_df(metric_trials: np.ndarray, metric_name: str) -> pd.DataFrame:
    """Build per-epoch repeatability DataFrame for one metric."""
    trials, horizon = metric_trials.shape
    avg = metric_trials.mean(axis=0)
    eps = 1e-12
    rel_err_pct = np.abs(metric_trials - avg[None, :]) / np.maximum(np.abs(avg[None, :]), eps) * 100.0

    std_t = metric_trials.std(axis=0, ddof=1)
    t_crit = 2.262 if trials == 10 else 1.96
    half_width = t_crit * std_t / np.sqrt(float(trials))
    ci_low = avg - half_width
    ci_high = avg + half_width

    data = {"epoch": np.arange(1, horizon + 1, dtype=int)}
    for j in range(trials):
        data[f"mean_{metric_name}_trial_{j + 1}"] = metric_trials[j]
    data[f"avg_mean_{metric_name}_all_trials"] = avg
    for j in range(trials):
        data[f"rel_err_pct_trial_{j + 1}"] = rel_err_pct[j]
    data["ci95_low"] = ci_low
    data["ci95_high"] = ci_high
    return pd.DataFrame(data)


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
    mean_order_trials = np.zeros((trials, horizon), dtype=float)
    mean_loss_trials = np.zeros((trials, horizon), dtype=float)
    trial_seeds = (
        [None] * trials
        if seed is None
        else [int(x) for x in rng.integers(0, 2**31 - 1, size=trials)]
    )

    if show_progress:
        _render_progress(0, trials)
    for j in range(trials):
        _, inv_dist, orders, loss = simulate_given_setting(
            inter_demand_ph=inter_ph,
            lead_time_ph=lead_ph,
            s=s_val,
            S=S_val,
            n_replications=replications,
            horizon=horizon,
            seed=trial_seeds[j],
        )
        mean_inventory_trials[j] = _compute_mean_inventory_from_dist(inv_dist)
        mean_order_trials[j] = orders
        mean_loss_trials[j] = loss
        if show_progress:
            _render_progress(j + 1, trials)

    df_inv = _build_metric_df(mean_inventory_trials, metric_name="inv")
    df_order = _build_metric_df(mean_order_trials, metric_name="order")
    df_loss = _build_metric_df(mean_loss_trials, metric_name="loss")

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
    return df_inv, df_order, df_loss, meta


def _parse_args():
    p = argparse.ArgumentParser(description="Repeatability test for mean inventory/order/loss trajectories.")
    p.add_argument("--trials", type=int, default=10, help="Number of repeated runs per setting.")
    p.add_argument("--replications", type=int, default=50000, help="Replications per trials.")
    p.add_argument("--horizon", type=int, default=100, help="Time horizon.")
    p.add_argument("--seed", type=int, default=None, help="Optional top-level seed.")
    p.add_argument("--inter-size-max", type=int, default=100, help="Max inter-demand PH size.")
    p.add_argument("--lead-size-max", type=int, default=100, help="Max lead-time PH size.")
    p.add_argument("--s", type=int, default=None, help="Optional fixed s (must provide S too).")
    p.add_argument("--S", type=int, default=None, help="Optional fixed S in [5,30].")
    p.add_argument(
        "--dump-dir",
        "--output-dir",
        dest="dump_dir",
        type=str,
        default=".",
        help="Fallback directory for outputs when specific metric dirs are not provided.",
    )
    p.add_argument(
        "--inv-dump-dir",
        type=str,
        default=r'C:\Users\Eshel\workspace\Elad_miklat\accuracy_results\inv',
        help="Directory for mean-inventory accuracy outputs.",
    )
    p.add_argument(
        "--order-dump-dir",
        type=str,
        default=r'C:\Users\Eshel\workspace\Elad_miklat\accuracy_results\order',
        help="Directory for mean-order accuracy outputs.",
    )
    p.add_argument(
        "--loss-dump-dir",
        type=str,
        default=r'C:\Users\Eshel\workspace\Elad_miklat\accuracy_results\loss',
        help="Directory for mean-loss accuracy outputs.",
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
    df_inv, df_order, df_loss, meta = run_accuracy_trial(
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

    default_out_dir = Path(args.dump_dir).resolve()
    inv_dir = Path(args.inv_dump_dir).resolve() if args.inv_dump_dir else default_out_dir
    order_dir = Path(args.order_dump_dir).resolve() if args.order_dump_dir else default_out_dir
    loss_dir = Path(args.loss_dump_dir).resolve() if args.loss_dump_dir else default_out_dir
    inv_dir.mkdir(parents=True, exist_ok=True)
    order_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    scv_token = _format_float_for_filename(float(meta["lead_scv"]))
    common = f"{int(meta['S'])}_{int(meta['s'])}_{int(meta['replications'])}_{scv_token}_{int(meta['model_num'])}"
    inv_pkl = inv_dir / f"inv_{common}.pkl"
    inv_csv = inv_dir / f"inv_{common}.csv"
    order_pkl = order_dir / f"order_{common}.pkl"
    order_csv = order_dir / f"order_{common}.csv"
    loss_pkl = loss_dir / f"loss_{common}.pkl"
    loss_csv = loss_dir / f"loss_{common}.csv"

    df_inv.to_pickle(inv_pkl)
    df_inv.to_csv(inv_csv, index=False)
    df_order.to_pickle(order_pkl)
    df_order.to_csv(order_csv, index=False)
    df_loss.to_pickle(loss_pkl)
    df_loss.to_csv(loss_csv, index=False)

    print("Finished repeatability trial.")
    print("Setting:", meta)
    print(f"Inventory dump directory: {inv_dir}")
    print(f"Order dump directory: {order_dir}")
    print(f"Loss dump directory: {loss_dir}")
    print(f"Saved inventory pickle: {inv_pkl}")
    print(f"Saved inventory csv: {inv_csv}")
    print(f"Saved order pickle: {order_pkl}")
    print(f"Saved order csv: {order_csv}")
    print(f"Saved loss pickle: {loss_pkl}")
    print(f"Saved loss csv: {loss_csv}")
    print("Inventory DataFrame shape:", df_inv.shape)
    print("Order DataFrame shape:", df_order.shape)
    print("Loss DataFrame shape:", df_loss.shape)
    print("Inventory head (first 5 rows):")
    print(df_inv.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

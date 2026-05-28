"""
Pivot Penalty ABM - Batch Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from model import PivotPenaltyModel

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

## Plotting defaults
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10, "figure.dpi": 150,
})

## Default batch parameters
# I chose 300 agents on 40 topics (7.5 per topic) as the standard density, with 8 seeds to reduce stochastic variance while keeping runtime manageable.
N_AGENTS = 300
N_TOPICS = 40
SKILL_DIM = 10
N_STEPS = 800
N_SEEDS = 16
BW = 3          # bin width for grouping pivot distances (3 → bins at 1, 4, 7, ...)
BURN_IN = 200   # discard early transient; convergence needs ~1/decay = 100 steps,
                # so 200 provides a 2x safety margin


## Ensemble runner
## Here I consulted AI + Mesa docs (Tutorial 5 "Collecting Data") for how to use DataCollector.get_agent_vars_dataframe() to extract agent-level panel data
## and combine results across multiple independent seeds into a single DataFrame for batch analysis
def run_ensemble(params, n_seeds=N_SEEDS, n_steps=N_STEPS):
    # run model n_seeds times with different random seeds, return combined DataFrame
    frames = []
    for s in range(n_seeds):
        # Each seed produces an independent replication
        p = {**params, "seed": s}
        model = PivotPenaltyModel(**p)
        for _ in range(n_steps):
            model.step()
        # Collect agent-level data from this replication
        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df["seed"] = s
        frames.append(df)
        if (s + 1) % 4 == 0:
            print(f"    ... {s + 1}/{n_seeds} seeds done")
    return pd.concat(frames, ignore_index=True)


## Analysis utilities
def prepare_df(df, val_col="Quality", burn_in=BURN_IN):
    # filter burn-in period and create a generic 'value' column
    df = df[df["Step"] >= burn_in].copy()
    df["value"] = df[val_col]
    return df


def binned_penalty(df, bw=BW, max_d=N_TOPICS // 2):
    # I bin pivots by distance to smooth the penalty curve — with ~5% pivot rate and BW=3, each bin aggregates enough observations for stable estimates.
    pivots = df[df["Did_Pivot"] == True].copy()
    stays = df[df["Did_Pivot"] == False]
    stay_mean = stays["value"].mean() if len(stays) > 0 else 1.0

    # Floor-divide into bins; center label = left edge + half width
    pivots["bin"] = (pivots["Pivot_Dist"] // bw) * bw + bw // 2
    pivots = pivots[pivots["bin"] <= max_d]

    # Per-seed bin means → cross-seed mean ± SEM.
    # Using per-seed aggregation gives honest confidence bands because observations within a single run are correlated (same agents, same environment). Pooled SEM would dramatically understate uncertainty.
    seed_bin = pivots.groupby(["seed", "bin"])["value"].mean().reset_index()
    agg = seed_bin.groupby("bin")["value"].agg(["mean", "sem"]).fillna(0)

    centers = sorted(agg.index)
    means = [agg.loc[c, "mean"] for c in centers]
    sems = [agg.loc[c, "sem"] for c in centers]
    counts = [len(pivots[pivots["bin"] == c]) for c in centers]
    return centers, means, sems, counts, stay_mean


## Fig 1: Baseline (skill mismatch only)
def fig1_baseline():
    # Baseline configuration: no agent interaction. I use this to establish the geometric baseline driven entirely by the AR(1) correlation structure.

    # rep_weight=0 disables reputation; crowding is off by default
    base = dict(n_agents=N_AGENTS, n_topics=N_TOPICS, skill_dim=SKILL_DIM,
                skill_decay=0.01, rep_weight=0.0)

    print("  Running baseline ensemble...")
    df = run_ensemble(base)
    # Baseline uses Quality (skill-only output) as the metric
    df = prepare_df(df, "Quality")
    centers, means, sems, counts, stay_mean = binned_penalty(df)
    # Normalize: ratio < 1 means pivoters do worse than stayers
    norm = [m / stay_mean for m in means]
    norm_sems = [s / stay_mean for s in sems]

    ## Two-panel figure: absolute + normalized
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: raw quality values
    axes[0].plot(centers, means, "o-", color="steelblue", markersize=5)
    axes[0].fill_between(centers,
        [m - 1.96 * s for m, s in zip(means, sems)],
        [m + 1.96 * s for m, s in zip(means, sems)],
        alpha=0.15, color="steelblue")
    axes[0].axhline(stay_mean, color="gray", ls="--", lw=1, label=f"Stay mean={stay_mean:.2f}")
    axes[0].set_xlabel("Pivot distance")
    axes[0].set_ylabel("Mean quality")
    axes[0].set_title("(A) Absolute penalty curve")
    axes[0].legend()

    # Panel B: ratio to stay mean — the standard penalty metric
    axes[1].plot(centers, norm, "o-", color="steelblue", markersize=5)
    axes[1].fill_between(centers,
        [n - 1.96 * s for n, s in zip(norm, norm_sems)],
        [n + 1.96 * s for n, s in zip(norm, norm_sems)],
        alpha=0.15, color="steelblue")
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Pivot distance")
    axes[1].set_ylabel("Pivot quality / Stay quality")
    axes[1].set_title("(B) Normalized (ratio < 1 = penalty)")
    axes[1].set_ylim(0, 1.1)

    plt.suptitle("Baseline: Skill Mismatch Penalty (No Agent Interaction)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig1_baseline.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print(f"  stay_mean={stay_mean:.4f}")
    for c, n_val, s in zip(centers, norm, norm_sems):
        print(f"    d={c:2d}: ratio={n_val:.4f} ± {1.96*s:.4f}")


## Fig 2: Baseline vs Reputation
def fig2_reputation():
    # I compare baseline and reputation on the same axes to show that reputation deepens the penalty without changing its monotone shape.
    base = dict(n_agents=N_AGENTS, n_topics=N_TOPICS, skill_dim=SKILL_DIM,
                skill_decay=0.01)

    # Baseline uses Quality (no reputation), Reputation uses Impact (includes rep amplification)
    # This is intentional: Impact = Quality when rep_weight=0, so the comparison measures quality on a fair basis for each configuration.
    configs = [
        ({"rep_weight": 0.0}, "Baseline (skill only)", "Quality", "steelblue"),
        ({"rep_weight": 0.5}, "Reputation ($w=0.5$)", "Impact", "crimson"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for extra, label, val_col, color in configs:
        print(f"  {label}...")
        params = {**base, **extra}
        df = run_ensemble(params)
        df = prepare_df(df, val_col)
        centers, means, sems, counts, stay_mean = binned_penalty(df)
        norm = [m / stay_mean for m in means]
        norm_sems = [s / stay_mean for s in sems]

        axes[0].plot(centers, means, "o-", color=color, markersize=5, label=label)
        axes[0].fill_between(centers,
            [m - 1.96 * s for m, s in zip(means, sems)],
            [m + 1.96 * s for m, s in zip(means, sems)],
            alpha=0.15, color=color)
        axes[1].plot(centers, norm, "o-", color=color, markersize=5, label=label)
        axes[1].fill_between(centers,
            [n - 1.96 * s for n, s in zip(norm, norm_sems)],
            [n + 1.96 * s for n, s in zip(norm, norm_sems)],
            alpha=0.15, color=color)

    axes[0].set_xlabel("Pivot distance")
    axes[0].set_ylabel("Mean output")
    axes[0].set_title("(A) Absolute penalty curve")
    axes[0].legend()

    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Pivot distance")
    axes[1].set_ylabel("Pivot / Stay ratio")
    axes[1].set_title("(B) Normalized — reputation deepens penalty")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    plt.suptitle("Baseline vs. Reputation: Peer-Relative Evaluation Amplifies Penalty",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig2_reputation.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


## Fig 3: Crowding (emergence test)
def fig3_crowding():
    # This is the core emergence test. I vary crowding strength gamma while keeping reputation off, to isolate the crowding mechanism's effect
    # I also add exo_pivot_prob=0.03 so that some random pivots still occur at all distances, ensuring measurement coverage even when scouting is active
    base = dict(n_agents=N_AGENTS, n_topics=N_TOPICS, skill_dim=SKILL_DIM,
                skill_decay=0.01, rep_weight=0.0,
                crowding_sensitivity=2.0, scout_radius=10, exo_pivot_prob=0.03)

    # gamma=0 serves as the baseline for direct comparison
    configs = [
        (0.0,  "No crowding (baseline)", "steelblue", "o-"),
        (0.02, "Crowding=0.02",    "#f39c12",   "s-"),
        (0.05, "Crowding=0.05",    "crimson",   "^-"),
        (0.10, "Crowding=0.10",    "#2ecc71",   "D-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for cs, label, color, marker in configs:
        print(f"  {label}...")
        params = {**base, "crowding_strength": cs}
        
        # When crowding is off, the model uses the simple pivot_prob parameter;
        # when crowding is on, it uses pivot_base_prob for the endogenous channel.
        if cs == 0:
            params["pivot_prob"] = 0.05
        else:
            params["pivot_base_prob"] = 0.05
        df = run_ensemble(params)
        
        # I use Impact here because it includes the crowding discount on production, which captures the full effect
        df = prepare_df(df, "Impact")
        centers, means, sems, counts, stay_mean = binned_penalty(df)
        norm = [m / stay_mean for m in means]
        norm_sems = [s / stay_mean for s in sems]

        axes[0].plot(centers, means, marker, color=color, markersize=5,
                     label=f"{label} (stay={stay_mean:.2f})", lw=1.5)
        axes[0].fill_between(centers,
            [m - 1.96 * s for m, s in zip(means, sems)],
            [m + 1.96 * s for m, s in zip(means, sems)],
            alpha=0.15, color=color)
        axes[1].plot(centers, norm, marker, color=color, markersize=5,
                     label=label, lw=1.5)
        axes[1].fill_between(centers,
            [n - 1.96 * s for n, s in zip(norm, norm_sems)],
            [n + 1.96 * s for n, s in zip(norm, norm_sems)],
            alpha=0.15, color=color)

    axes[0].set_xlabel("Pivot distance")
    axes[0].set_ylabel("Mean impact (post-crowding)")
    axes[0].set_title("(A) Absolute")
    axes[0].legend(fontsize=9)

    # y-axis goes to 1.5 because crowding can push short-pivot ratio above 1.0
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Pivot distance")
    axes[1].set_ylabel("Pivot / Stay ratio")
    axes[1].set_title("(B) Normalized — crowding reshapes the curve")
    axes[1].set_ylim(0, 1.5)
    axes[1].legend(fontsize=9)

    plt.suptitle("Crowding Configuration: Agent Interaction Reshapes Penalty (Emergence Test)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig3_crowding.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


## Fig 4: Density dependence (emergence)
def fig4_density():
    # Core emergence result: same agent rules, different population density, different penalty curve shape. 
    # I hold all parameters fixed and vary only N to test density dependence.

    # All mechanism parameters are fixed; only n_agents varies
    base = dict(n_topics=N_TOPICS, skill_dim=SKILL_DIM, skill_decay=0.01,
                rep_weight=0.0, crowding_strength=0.05,
                crowding_sensitivity=2.0, scout_radius=10,
                pivot_base_prob=0.05, exo_pivot_prob=0.03)

    # Three densities: 2.5, 5.0, and 10.0 scientists per topic
    densities = [
        (100,  "n=100 (2.5/topic)", "steelblue", "o-"),
        (200,  "n=200 (5/topic)",   "#f39c12",   "s-"),
        (400,  "n=400 (10/topic)",  "crimson",   "^-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for n, label, color, marker in densities:
        print(f"  {label}...")
        params = {**base, "n_agents": n}
        df = run_ensemble(params)
        # Impact captures both crowding discount and reputation amplification
        df = prepare_df(df, "Impact")
        centers, means, sems, counts, stay_mean = binned_penalty(df)
        norm = [m / stay_mean for m in means]
        norm_sems = [s / stay_mean for s in sems]

        axes[0].plot(centers, means, marker, color=color, markersize=5,
                     label=f"{label} (stay={stay_mean:.2f})", lw=1.5)
        axes[0].fill_between(centers,
            [m - 1.96 * s for m, s in zip(means, sems)],
            [m + 1.96 * s for m, s in zip(means, sems)],
            alpha=0.15, color=color)
        axes[1].plot(centers, norm, marker, color=color, markersize=5,
                     label=label, lw=1.5)
        axes[1].fill_between(centers,
            [n - 1.96 * s for n, s in zip(norm, norm_sems)],
            [n + 1.96 * s for n, s in zip(norm, norm_sems)],
            alpha=0.15, color=color)

    axes[0].set_xlabel("Pivot distance")
    axes[0].set_ylabel("Mean impact (post-crowding)")
    axes[0].set_title("(A) Absolute — stay level drops with density")
    axes[0].legend(fontsize=9)

    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Pivot distance")
    axes[1].set_ylabel("Pivot / Stay ratio")
    axes[1].set_title("(B) Normalized — curve shape is density-dependent")
    axes[1].set_ylim(0, 1.5)
    axes[1].legend(fontsize=9)

    plt.suptitle("Density Dependence: Same Rules, Different Density → Different Penalty",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig4_density.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


## Fig 5: Convergence
def fig5_convergence():
    # If the curves overlap for T=400, 600, and 1000, the model has reached steady state and results are not artifacts of transient dynamics.
    # Baseline only, simplest case to verify convergence
    base = dict(n_agents=N_AGENTS, n_topics=N_TOPICS, skill_dim=SKILL_DIM,
                rep_weight=0.0, skill_decay=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Three simulation lengths; curves should overlap if converged
    for n_steps, color, marker in [(400, "steelblue", "o-"),
                                   (600, "crimson", "s-"),
                                   (1000, "#2ecc71", "^-")]:
        print(f"  T={n_steps}...")
        df = run_ensemble(base, n_steps=n_steps)
        df = prepare_df(df, "Quality")
        centers, means, sems, counts, stay_mean = binned_penalty(df)
        norm = [m / stay_mean for m in means]
        norm_sems = [s / stay_mean for s in sems]

        axes[0].plot(centers, means, marker, color=color, markersize=5,
                     label=f"T={n_steps} (stay={stay_mean:.2f})", lw=1.5)
        axes[0].fill_between(centers,
            [m - 1.96 * s for m, s in zip(means, sems)],
            [m + 1.96 * s for m, s in zip(means, sems)],
            alpha=0.15, color=color)
        axes[1].plot(centers, norm, marker, color=color, markersize=5,
                     label=f"T={n_steps}", lw=1.5)
        axes[1].fill_between(centers,
            [n - 1.96 * s for n, s in zip(norm, norm_sems)],
            [n + 1.96 * s for n, s in zip(norm, norm_sems)],
            alpha=0.15, color=color)

    axes[0].set_xlabel("Pivot distance")
    axes[0].set_ylabel("Mean quality")
    axes[0].set_title("(A) Absolute")
    axes[0].legend()

    # If normalized curves overlap, the penalty shape is time-invariant
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Pivot distance")
    axes[1].set_ylabel("Pivot / Stay ratio")
    axes[1].set_title("(B) Normalized — curves overlap = converged")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    plt.suptitle("Convergence check (skill_decay=0.01)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig5_convergence.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


## Fig 6: Full model (all three mechanisms)
def fig6_full_model():
    # I test what happens when all three mechanisms operate simultaneously.
    # This is the most realistic configuration and shows whether the crowding-induced non-monotonicity survives the addition of reputation.

    base = dict(n_agents=N_AGENTS, n_topics=N_TOPICS, skill_dim=SKILL_DIM,
                skill_decay=0.01, crowding_strength=0.05,
                crowding_sensitivity=2.0, scout_radius=10,
                pivot_base_prob=0.05, exo_pivot_prob=0.03)

    configs = [
        (0.0, "Crowding only ($w=0$)",       "steelblue", "o-"),
        (0.5, "Full model ($w=0.5$)",         "crimson",   "s-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for w, label, color, marker in configs:
        print(f"  {label}...")
        params = {**base, "rep_weight": w}
        df = run_ensemble(params)
        df = prepare_df(df, "Impact")
        centers, means, sems, counts, stay_mean = binned_penalty(df)
        norm = [m / stay_mean for m in means]
        norm_sems = [s / stay_mean for s in sems]

        axes[0].plot(centers, means, marker, color=color, markersize=5,
                     label=f"{label} (stay={stay_mean:.2f})", lw=1.5)
        axes[0].fill_between(centers,
            [m - 1.96 * s for m, s in zip(means, sems)],
            [m + 1.96 * s for m, s in zip(means, sems)],
            alpha=0.15, color=color)
        axes[1].plot(centers, norm, marker, color=color, markersize=5,
                     label=label, lw=1.5)
        axes[1].fill_between(centers,
            [n - 1.96 * s for n, s in zip(norm, norm_sems)],
            [n + 1.96 * s for n, s in zip(norm, norm_sems)],
            alpha=0.15, color=color)

    axes[0].set_xlabel("Pivot distance")
    axes[0].set_ylabel("Mean impact")
    axes[0].set_title("(A) Absolute")
    axes[0].legend(fontsize=9)

    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Pivot distance")
    axes[1].set_ylabel("Pivot / Stay ratio")
    axes[1].set_title("(B) Normalized — non-monotonicity survives reputation")
    axes[1].set_ylim(0, 1.5)
    axes[1].legend(fontsize=9)

    plt.suptitle("Full Model: All Three Mechanisms Active",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "fig6_full_model.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


## Main
if __name__ == "__main__":
    print("=" * 60)
    print("Pivot Penalty ABM — Batch Analysis")
    print(f"  {N_AGENTS} agents, {N_TOPICS} topics, {N_SEEDS} seeds, {N_STEPS} steps")
    print("=" * 60)

    # Figures are ordered as a progressive mechanism comparison:
    # Baseline (no interaction) → Reputation (weak) → Crowding (strong)
    print("\nFig 1: Baseline (skill mismatch only)...")
    fig1_baseline()

    print("\nFig 2: Baseline vs Reputation (peer interaction)...")
    fig2_reputation()

    print("\nFig 3: Crowding (agent interaction, emergence test)...")
    fig3_crowding()

    print("\nFig 4: Density dependence (emergence)...")
    fig4_density()

    print("\nFig 5: Convergence check...")
    fig5_convergence()

    print("\nFig 6: Full model (all three mechanisms)...")
    fig6_full_model()

    print("\nAll figures saved to:", OUT_DIR)

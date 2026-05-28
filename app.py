"""
Pivot Penalty ABM - App
"""

import numpy as np
import solara
from matplotlib.figure import Figure

from model import PivotPenaltyModel
from mesa.visualization import SolaraViz, make_plot_component, Slider
from mesa.visualization.utils import update_counter


## Custom visualization: Topic Ring
## Since scientists live on a 1D topic ring (not a Grid), I draw it as a polar scatter plot. Each dot is a scientist at their topic;
## Red = just pivoted, blue = stayed. I added radial jitter to reduce overlap.
## Here I consulted AI + Mesa docs (Tutorial 10 "Visualization — Custom Components", section "Building Custom Components"; API Reference → mesa.visualization)
## For how to build a custom Solara component. The key pattern from the tutorial: @solara.component decorator, update_counter.get() to subscribe to model step changes, thread-safe
## Figure() instead of plt.figure(), and solara.FigureMatplotlib(fig) to render into the dashboard.


@solara.component
def TopicRing(model):
    update_counter.get()  # subscribe to model step updates

    fig = Figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)

    n_topics = model.n_topics
    angles = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)

    ## Collect agent positions and pivot status
    stay_topics, pivot_topics = [], []
    for agent in model.agents:
        if agent.did_pivot:
            pivot_topics.append(agent.topic)
        else:
            stay_topics.append(agent.topic)

    ## Draw agents with radial jitter to reduce overplotting
    rng = np.random.default_rng(0)
    if stay_topics:
        theta_s = angles[stay_topics] + rng.uniform(-0.04, 0.04, len(stay_topics))
        r_s = 1.0 + rng.uniform(-0.15, 0.15, len(stay_topics))
        ax.scatter(theta_s, r_s, c="tab:blue", s=8, alpha=0.4, label="Stay")
    if pivot_topics:
        theta_p = angles[pivot_topics] + rng.uniform(-0.04, 0.04, len(pivot_topics))
        r_p = 1.0 + rng.uniform(-0.15, 0.15, len(pivot_topics))
        ax.scatter(theta_p, r_p, c="tab:red", s=25, alpha=0.8,
                   marker="*", label="Pivot")

    ## Draw topic markers on the ring
    ax.scatter(angles, np.ones(n_topics), c="gray", s=3, alpha=0.3, zorder=0)

    ## Formatting
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    tick_step = max(1, n_topics // 8)
    ax.set_xticks(angles[::tick_step])
    ax.set_xticklabels([str(i) for i in range(0, n_topics, tick_step)], fontsize=8)
    ax.set_title(
        f"Topic Ring  (step {model._step_count},  pivots: {len(pivot_topics)})",
        fontsize=11, pad=15,
    )
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.3, 1.1))

    solara.FigureMatplotlib(fig)


@solara.component
def TopicHistogram(model):
    # bar chart of topic occupancy — highlights crowded topics in red
    update_counter.get()  # subscribe to model step updates

    fig = Figure(figsize=(6, 2.5))
    ax = fig.add_subplot(111)

    n_topics = model.n_topics
    counts = np.zeros(n_topics)
    for agent in model.agents:
        counts[agent.topic] += 1

    uniform = len(model.agents) / n_topics
    colors = ["tab:red" if c > uniform * 1.5 else "tab:blue" for c in counts]
    ax.bar(range(n_topics), counts, color=colors, alpha=0.7, width=1.0)
    ax.axhline(uniform, color="gray", ls="--", lw=1, label="Uniform")
    ax.set_xlabel("Topic", fontsize=10)
    ax.set_ylabel("# Scientists", fontsize=10)
    ax.set_title(f"Topic Occupancy  (step {model._step_count})", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()

    solara.FigureMatplotlib(fig)


## Parameter sliders
## I exposed only the parameters a reviewer would want to test interactively. rep_weight=0 gives baseline; 
## increasing it activates reputation. crowding_strength>0 activates crowding + scouting.

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    # Environment
    "n_agents": Slider("Number of Scientists (N)", value=200, min=50, max=500, step=50),
    "n_topics": Slider("Number of Topics (T)", value=40, min=10, max=80, step=10),
    "spatial_corr": Slider(
        "Topic Correlation (ρ)", value=0.7, min=0.1, max=0.95, step=0.05
    ),
    # Skill accumulation
    "skill_decay": Slider("Skill Decay (δ)", value=0.01, min=0.0, max=0.05, step=0.005),
    "learning_rate": Slider("Learning Rate (λ₀)", value=0.12, min=0.01, max=0.30, step=0.01),
    "pivot_prob": Slider(
        "Base Pivot Probability (p₀)", value=0.05, min=0.01, max=0.20, step=0.01
    ),
    # Reputation layer (0 = off)
    "rep_weight": Slider(
        "Reputation Weight w (0=off)", value=0.0, min=0.0, max=1.0, step=0.1
    ),
    # Crowding layer (0 = off)
    "crowding_strength": Slider(
        "Crowding Strength γ (0=off)", value=0.0, min=0.0, max=0.20, step=0.01
    ),
    "crowding_sensitivity": Slider(
        "Crowding → Pivot Sensitivity (κ)", value=2.0, min=0.0, max=5.0, step=0.5
    ),
}

## Time-series plots
# quality and impact overlap when rep_weight=0; diverge when rep_weight>0
quality_plot = make_plot_component(
    {"Mean_Quality": "tab:blue", "Mean_Impact": "tab:purple"},
    backend="matplotlib",
)

# stay vs pivot on impact — the core penalty visualization, when rep_weight=0, impact == quality so this works for all configurations
penalty_plot = make_plot_component(
    {"Stay_Impact": "tab:blue", "Pivot_Impact": "tab:red"},
    backend="matplotlib",
)

# skill norm should converge to steady state when skill_decay > 0
skill_norm_plot = make_plot_component(
    {"Mean_Skill_Norm": "tab:green"},
    backend="matplotlib",
)

# pivot rate sanity check — should hover around p0 = 0.05
pivot_rate_plot = make_plot_component(
    {"Pivot_Rate": "tab:orange"},
    backend="matplotlib",
)


model1 = PivotPenaltyModel()

page = SolaraViz(
    model1,
    components=[
        TopicRing, TopicHistogram,
        quality_plot, penalty_plot,
        skill_norm_plot, pivot_rate_plot,
    ],
    model_params=model_params,
    name="Pivot Penalty in Science",
    play_interval=150,
)

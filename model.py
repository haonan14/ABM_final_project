"""
Pivot Penalty ABM - Model
"""

import numpy as np
from collections import Counter, defaultdict

import mesa
from agents import Scientist


## I use an AR(1) process to generate topic skill vectors on the ring. Adjacent topics are correlated (rho = spatial_corr), so short-distance pivots preserve more skill alignment than long-distance ones.
##
## Here I consulted AI for how to implement a stationary AR(1) process whose marginal variance stays constant across topics. The key is the sqrt(1 - rho^2) scaling on the innovation term, which is the standard result ensuring
## Var(raw[j]) = Var(raw[0]) for all j. I then normalize each vector to unit length so the dot product measures pure angular alignment between skill profiles.
def generate_topic_skills(n_topics, skill_dim, spatial_corr, rng):
    rho = spatial_corr
    raw = np.zeros((n_topics, skill_dim))
    raw[0] = rng.standard_normal(skill_dim)
    for j in range(1, n_topics):
        innovation = rng.standard_normal(skill_dim)
        raw[j] = rho * raw[j - 1] + np.sqrt(1 - rho**2) * innovation
    # Normalize to unit length so dot-product measures pure alignment
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return raw / norms


class PivotPenaltyModel(mesa.Model):
    ## Helper functions for DataCollector reporters

    # stay vs pivot split lets me measure the penalty directly in the GUI
    def _stay_mean_quality(self):
        vals = [a.base_quality for a in self.agents if not a.did_pivot]
        return float(np.mean(vals)) if vals else 0.0

    def _pivot_mean_quality(self):
        vals = [a.base_quality for a in self.agents if a.did_pivot]
        return float(np.mean(vals)) if vals else 0.0

    def _stay_mean_impact(self):
        vals = [a.impact for a in self.agents if not a.did_pivot]
        return float(np.mean(vals)) if vals else 0.0

    def _pivot_mean_impact(self):
        vals = [a.impact for a in self.agents if a.did_pivot]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_quality(self):
        return float(np.mean([a.base_quality for a in self.agents])) if len(self.agents) > 0 else 0.0

    def _mean_impact(self):
        return float(np.mean([a.impact for a in self.agents])) if len(self.agents) > 0 else 0.0

    def _pivot_rate(self):
        return sum(1 for a in self.agents if a.did_pivot) / max(1, len(self.agents))

    # I track skill norm to verify that skill decay produces a stable steady state
    def _mean_skill_norm(self):
        return float(np.mean([np.linalg.norm(a.skills) for a in self.agents])) if len(self.agents) > 0 else 0.0

    ## Define initiation, inherit seed property from parent class
    def __init__(
        self,
        n_agents=200,
        n_topics=40,
        skill_dim=10,
        learning_rate=0.12,
        learning_decay=0.03,
        skill_decay=0.01,
        spatial_corr=0.7,
        pivot_prob=0.05,
        noise_std=0.1,

        # Reputation: setting rep_weight=0 gives the baseline (skill mismatch only)
        rep_weight=0.0,
        rep_accumulation=0.01,
        rep_decay=0.02,

        # Crowding: setting crowding_strength>0 activates crowding + scouting
        crowding_strength=0.0,
        pivot_base_prob=0.05,
        crowding_sensitivity=0.0,
        scout_radius=10,
        exo_pivot_prob=0.0,
        funding_enabled=False,
        funding_fraction=0.3,
        funding_boost=1.5,
        funding_window=20,

        # Heterogeneity in learning speed
        sigma_ability=0.0,
        seed=None,
    ):
        super().__init__(rng=seed)

        ## Instantiate model parameters
        self.n_agents_init = n_agents
        self.n_topics = n_topics
        self.skill_dim = skill_dim
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.skill_decay = skill_decay
        self.spatial_corr = spatial_corr
        self.pivot_prob = pivot_prob
        self.noise_std = noise_std

        # Reputation
        self.rep_weight = rep_weight
        self.rep_accumulation = rep_accumulation
        self.rep_decay = rep_decay

        # Crowding
        self.crowding_strength = crowding_strength
        self.use_crowding = crowding_strength > 0
        self.pivot_base_prob = pivot_base_prob
        self.crowding_sensitivity = crowding_sensitivity
        self.scout_radius = scout_radius
        self.exo_pivot_prob = exo_pivot_prob
        self.funding_enabled = funding_enabled
        self.funding_fraction = funding_fraction
        self.funding_boost = funding_boost
        self.funding_window = funding_window

        self.sigma_ability = sigma_ability

        self.running = True
        self.topic_counts = {}
        self.topic_avg_quality = {}   # Reputation: peer benchmark per topic
        self._step_count = 0

        ## Generate topic space: skill requirements and intrinsic value
        self.topic_skills = generate_topic_skills(
            n_topics, skill_dim, spatial_corr, self.rng
        )
        self.topic_value = self.rng.uniform(0.5, 1.5, size=n_topics)

        ## Draw agent abilities from log-normal distribution
        if sigma_ability > 0:
            abilities = self.rng.lognormal(0, sigma_ability, size=n_agents)
        else:
            abilities = np.ones(n_agents)

        ## Create scientist agents and assign random starting topics
        for i in range(n_agents):
            starting_topic = self.random.randrange(n_topics)
            Scientist(self, starting_topic, ability=float(abilities[i]))

        ## Define data collector with model-level and agent-level reporters
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Mean_Quality": self._mean_quality,
                "Mean_Impact": self._mean_impact,
                "Stay_Quality": self._stay_mean_quality,
                "Pivot_Quality": self._pivot_mean_quality,
                "Stay_Impact": self._stay_mean_impact,
                "Pivot_Impact": self._pivot_mean_impact,
                "Pivot_Rate": self._pivot_rate,
                "Mean_Skill_Norm": self._mean_skill_norm,
            },
            agent_reporters={
                "Topic": "topic",
                "Quality": "base_quality",
                "Impact": "impact",
                "Pivot_Dist": "pivot_dist",
                "Did_Pivot": "did_pivot",
                "Tenure": "tenure",
                "Skill_Norm": lambda a: float(np.linalg.norm(a.skills)),
                "Reputation": lambda a: float(a.reputation[a.topic]),
                "Ability": "ability",
            },
        )
        self.datacollector.collect(self)

    ## Define step: snapshot occupancy, allocate funding, all agents act, then collect data
    def step(self):
        ## Snapshot topic occupancy for crowding calculations
        if self.use_crowding:
            self.topic_counts = Counter(a.topic for a in self.agents)

        ## Reputation: compute average quality per topic as peer benchmark
        ## Each agent's reputation will grow relative to this benchmark, creating coupling: A's reputation depends on B, C, D's output.
        if self.rep_weight > 0 and self._step_count > 0:
            topic_quals = defaultdict(list)
            for a in self.agents:
                topic_quals[a.topic].append(a.base_quality)
            self.topic_avg_quality = {
                t: float(np.mean(qs)) for t, qs in topic_quals.items()
            }

        ## Funding allocation: rank agents by recent output, top fraction gets boost
        if (
            self.funding_enabled
            and self._step_count > 0
            and self._step_count % self.funding_window == 0
        ):
            ranked = sorted(
                self.agents, key=lambda a: a.recent_output, reverse=True
            )
            cutoff = int(self.funding_fraction * len(ranked))
            for i, ag in enumerate(ranked):
                ag.funded = i < cutoff

        ## All agents act (pivot, produce, learn, update reputation)
        self.agents.shuffle_do("step")

        self._step_count += 1

        ## Collect model and agent-level data
        self.datacollector.collect(self)

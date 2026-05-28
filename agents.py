"""
Pivot Penalty ABM - Scientist Agent
"""

import numpy as np
from mesa import Agent


## Helper function to get ring distance between two topics
def topic_distance(a, b, n):
    d = abs(a - b)
    return min(d, n - d)


class Scientist(Agent):
    ## Initiate agent, inherit model property from parent class
    def __init__(self, model, topic, ability=1.0):
        super().__init__(model)
        self.topic = topic
        self.skills = np.zeros(model.skill_dim)
        self.reputation = np.zeros(model.n_topics)
        self.ability = ability
        self.tenure = 0

        ## Per-step outputs, reset at start of each step
        self.did_pivot = False
        self.pivot_dist = 0
        self.base_quality = 0.0
        self.impact = 0.0

        ## Crowding: funding attributes
        self.funded = False
        self.recent_output = 0.0

    ## Define step: pivot decision, produce, learn, update reputation
    def step(self):
        model = self.model

        ## Reset per-step state
        self.did_pivot = False
        self.pivot_dist = 0

        ## 1. Pivot decision
        if model.use_crowding:
            self._pivot_endogenous()
        else:
            # Baseline/Reputation: pivot to a random topic.
            # Reputation push-out: below-average reputation increases pivot probability.
            # Sensitivity is higher than impact weight because career decisions
            # respond more strongly to failure than output quality does.
            p_pivot = model.pivot_prob
            if model.rep_weight > 0:
                rep_push = max(0.0, -self.reputation[self.topic])
                p_pivot = min(0.5, p_pivot * (1 + 3.0 * rep_push))
            if self.random.random() < p_pivot:
                new_topic = self.random.randrange(model.n_topics - 1)
                if new_topic >= self.topic:
                    new_topic += 1
                self.pivot_dist = topic_distance(
                    self.topic, new_topic, model.n_topics
                )
                self.topic = new_topic
                self.tenure = 0
                self.did_pivot = True

        ## 2. Produce (before learning)
        # Quality = alignment between skill vector and topic requirement
        match = max(0.0, np.dot(self.skills, model.topic_skills[self.topic]))
        noise = max(0.0, 1 + model.rng.normal(0, model.noise_std))
        self.base_quality = match * model.topic_value[self.topic] * noise

        # Crowding: more agents on a topic means diminishing returns
        crowded_q = self.base_quality
        if model.use_crowding:
            n_on = model.topic_counts.get(self.topic, 1)
            crowded_q /= 1 + model.crowding_strength * n_on

        # Reputation: relative standing affects impact.
        # Positive rep (above-average) boosts; negative rep (below-average) penalizes.
        rep = self.reputation[self.topic]
        self.impact = max(0.0, crowded_q * (1 + model.rep_weight * rep))

        ## 3. Learn (after producing)
        # Skill decay: all components fade each step, preventing unbounded accumulation
        if model.skill_decay > 0:
            self.skills *= 1 - model.skill_decay

        # Diminishing returns to learning: rate decreases with tenure on topic
        # Ability multiplier allows heterogeneous agents to learn at different speeds
        lr = (
            model.learning_rate
            * self.ability
            / (1 + model.learning_decay * self.tenure)
        )
        if model.funding_enabled and self.funded:
            lr *= model.funding_boost

        self.skills += lr * model.topic_skills[self.topic]
        self.tenure += 1

        ## 4. Update reputation
        # Reputation: grows relative to PEER AVERAGE on topic.
        # If the output > topic average → reputation increases
        # If the output < topic average → reputation decreases
        peer_avg = model.topic_avg_quality.get(self.topic, 0.0)
        rep_delta = model.rep_accumulation * (self.base_quality - peer_avg)
        self.reputation[self.topic] += rep_delta
        # Reputation on all other topics decays
        mask = np.ones(model.n_topics, dtype=bool)
        mask[self.topic] = False
        self.reputation[mask] *= 1 - model.rep_decay

        ## Update rolling output average (used for crowding-config funding ranking)
        alpha = 2 / (model.funding_window + 1)
        self.recent_output = alpha * self.impact + (1 - alpha) * self.recent_output

    ## Crowding-driven pivot: I chose to make agents in crowded topics more likely to pivot, and when they do, they scout nearby topics for better
    ## opportunities. Attractiveness balances topic value, crowding, and skill fit, this makes agents boundedly rational.
    def _pivot_endogenous(self):
        model = self.model
        n_here = model.topic_counts.get(self.topic, 1)

        # How crowded is the current topic relative to uniform allocation?
        crowding_ratio = n_here / (model.n_agents_init / model.n_topics)

        # Push-out: below-average reputation increases pivot probability.
        # Scientists failing in competitive fields have more incentive to switch.
        rep_push = max(0.0, -self.reputation[self.topic]) if model.rep_weight > 0 else 0.0
        # Endogenous pivot probability increases with both crowding and negative rep
        p_endo = min(
            0.5,
            model.pivot_base_prob
            * (1 + model.crowding_sensitivity * max(0, crowding_ratio - 1)
               + 3.0 * rep_push),
        )

        if self.random.random() < p_endo:
            ## Scout nearby topics within a limited radius
            candidates = []
            for off in range(-model.scout_radius, model.scout_radius + 1):
                if off == 0:
                    continue
                c = (self.topic + off) % model.n_topics
                nc = model.topic_counts.get(c, 0)
                d = topic_distance(self.topic, c, model.n_topics)
                # Skill fit: how well do my current skills match the candidate?
                skill_fit = max(0.01, np.dot(self.skills, model.topic_skills[c]))
                # Attractiveness trades off value, crowding, skill fit, and distance
                attractiveness = (
                    model.topic_value[c] / (1 + model.crowding_strength * nc)
                    * skill_fit
                    - 0.1 * d
                )
                candidates.append((c, attractiveness, d))

            ## Select destination via softmax over top-5 candidates
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_k = candidates[:5]
                weights = np.array([x[1] for x in top_k])
                # Softmax: subtract max for numerical stability
                weights = np.exp(weights - weights.max())
                weights /= weights.sum()
                idx = model.rng.choice(len(top_k), p=weights)
                new_topic, _, self.pivot_dist = top_k[idx]
                self.topic = new_topic
                self.tenure = 0
                self.did_pivot = True

        ## Exogenous channel: non-strategic random pivots for measurement coverage
        # Without this, scouting concentrates pivots at short distances
        if not self.did_pivot and model.exo_pivot_prob > 0:
            if self.random.random() < model.exo_pivot_prob:
                # Exclude current topic to avoid distance-0 "pivots"
                new_topic = self.random.randrange(model.n_topics - 1)
                if new_topic >= self.topic:
                    new_topic += 1
                self.pivot_dist = topic_distance(
                    self.topic, new_topic, model.n_topics
                )
                self.topic = new_topic
                self.tenure = 0
                self.did_pivot = True

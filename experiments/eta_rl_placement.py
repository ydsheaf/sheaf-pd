#!/usr/bin/env python3
"""
η-guided RL Placement — Phase 2 (Issue #12)
============================================

PPO agent learns cell placement using η as a dense reward signal.

Environment:
  - Cells are placed iteratively into a congested region
  - State: per-cell local features [Δ̄_local, η_local, n_overlaps, rel_pos(2),
    nearest_neighbor_dist, cell_size(2)] = 8 dims
  - Action: displacement (dx, dy) from initial position — continuous 2D
  - Reward: -overlap_area_delta - α·η_delta + β·overlap_resolved_bonus

Key insight from sheaf-swarm: scalar gain RL doesn't beat η-shield (Theorem 11).
But placement has per-cell action space — richer than scalar. The agent can
learn cell-specific displacements that a uniform shield cannot.

Baselines:
  1. random:      random displacement within ±range
  2. greedy:      move away from highest-overlap neighbor
  3. eta_shield:  Dykstra projection with step *= (1-η)
  4. ppo:         learned policy

Usage:
  python experiments/eta_rl_placement.py
  python experiments/eta_rl_placement.py --episodes 300 --design gcd_nangate45
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)
from eta_shield_placement import (
    compute_overlaps, compute_gcell_metrics, dykstra_projection,
    create_congested_placement, ETA_GAIN_FLOOR,
)

# Check for torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not found — PPO mode disabled, baselines only")


# ─── RL Parameters ───
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
LR = 3e-4
PPO_EPOCHS = 4
BATCH_SIZE = 64
EPISODES_DEFAULT = 200
STEPS_PER_EPISODE = 50      # placement iterations per episode
ACTION_SCALE = 1.0           # max displacement per step (μm)
OBS_DIM = 8                  # per-cell observation dimension
ACTION_DIM = 2               # (dx, dy)

# Reward weights
R_OVERLAP_AREA = -10.0       # penalty per μm² overlap area
R_ETA_DELTA = -5.0           # penalty for increasing η
R_OVERLAP_RESOLVED = 0.5     # bonus per resolved overlap
R_CONVERGENCE = 10.0         # bonus for reaching 0 overlaps


# ═══════════════════════════════════════════════════════════════════
# Placement Environment
# ═══════════════════════════════════════════════════════════════════

class PlacementEnv:
    """Placement legalization environment.

    Each episode:
    1. Load a design with localized compression (congested placement)
    2. Agent outputs per-G-cell gain factors for Dykstra projection
    3. Reward reflects overlap reduction and η reduction

    The agent learns spatially-varying gains — the generalization of
    η-shield's uniform gain. This is the placement analog of learning
    a non-scalar intervention in sheaf-swarm.

    Observation: per-G-cell features [Δ̄, η, n_cells, n_overlaps, density]
    Action: per-G-cell gain factors (gs*gs values in [ε, 1])
    """

    def __init__(self, design_name="gcd_nangate45", compress_factor=0.3,
                 compress_region="center", gs=6, max_cells=200, seed=42,
                 max_steps=STEPS_PER_EPISODE):
        self.design_name = design_name
        self.compress_factor = compress_factor
        self.compress_region = compress_region
        self.gs = gs
        self.max_cells = max_cells
        self.seed = seed

        # Load design once
        positions, widths, heights, die_area, _ = load_design(design_name)
        self.orig_positions = positions
        self.widths = widths
        self.heights = heights
        self.die_area = die_area

        N = len(positions)
        med_w = np.median(widths)
        med_h = np.median(heights)
        self.diag = np.sqrt(med_w**2 + med_h**2)
        self.r_interact = self.diag * 1.5

        # Subsample if needed (for tractable RL)
        if N > max_cells:
            rng = np.random.default_rng(seed)
            cx, cy = np.mean(positions, axis=0)
            dists = np.sqrt((positions[:, 0] - cx)**2 +
                            (positions[:, 1] - cy)**2)
            idx = np.argsort(dists)[:max_cells]
            self.orig_positions = positions[idx]
            self.widths = widths[idx]
            self.heights = heights[idx]

        self.N = len(self.orig_positions)
        self.max_steps = max_steps
        self.step_count = 0

        # State
        self.positions = None
        self.anchor_positions = None
        self.prev_overlap_area = 0.0
        self.prev_eta_max = 0.0
        self.prev_n_overlaps = 0

    def reset(self, seed=None):
        """Reset to a new congested placement."""
        if seed is not None:
            self.seed = seed

        # Create congested placement
        self.positions, _, _ = create_congested_placement(
            self.orig_positions, self.widths, self.heights, self.die_area,
            compress_region=self.compress_region,
            compress_factor=self.compress_factor,
            seed=self.seed,
        )
        self.anchor_positions = self.positions.copy()

        # Compute initial metrics
        overlaps = compute_overlaps(self.positions, self.widths, self.heights)
        self.prev_n_overlaps = len(overlaps)
        self.prev_overlap_area = sum(a for _, _, a in overlaps)

        gains, eta_map, sigma_map, dbar_map, gcell_assign, ncells_map = \
            compute_gcell_metrics(self.positions, self.widths, self.heights,
                                  self.die_area, self.gs, self.r_interact)
        self.prev_eta_max = float(np.max(eta_map))
        self._gcell_assign = gcell_assign
        self._eta_map = eta_map
        self._dbar_map = dbar_map
        self._ncells_map = ncells_map

        self.step_count = 0

        return self._get_obs()

    @property
    def obs_dim(self):
        """Per-G-cell observation: [Δ̄, η, n_cells, n_overlaps, density]."""
        return 5

    @property
    def action_dim(self):
        """Per-G-cell gain factor."""
        return self.gs * self.gs

    def _get_obs(self):
        """Flat observation: per-G-cell features.

        For each G-cell: [Δ̄/10, η, n_cells/20, n_overlaps/20, density]
        Flattened to (gs*gs * 5,).
        """
        gs = self.gs
        obs = np.zeros((gs, gs, 5), dtype=np.float32)

        # Per-G-cell overlap count
        overlaps = compute_overlaps(self.positions, self.widths, self.heights)
        gcell_overlaps = np.zeros((gs, gs))
        for i, j, area in overlaps:
            gx_i, gy_i = self._gcell_assign.get(i, (0, 0))
            gcell_overlaps[gy_i, gx_i] += 1
            gx_j, gy_j = self._gcell_assign.get(j, (0, 0))
            gcell_overlaps[gy_j, gx_j] += 1

        for gx in range(gs):
            for gy in range(gs):
                obs[gy, gx, 0] = self._dbar_map[gy, gx] / 10.0
                obs[gy, gx, 1] = self._eta_map[gy, gx]
                obs[gy, gx, 2] = self._ncells_map[gy, gx] / 20.0
                obs[gy, gx, 3] = gcell_overlaps[gy, gx] / 20.0
                obs[gy, gx, 4] = self._ncells_map[gy, gx] / max(1, self.N) * gs * gs

        return obs.reshape(-1)  # flat (gs*gs*5,)

    def step(self, gains_action):
        """Apply Dykstra projection with per-G-cell gains from agent.

        Args:
            gains_action: (gs*gs,) array of raw gain values.
                          Transformed via sigmoid to [ε, 1].

        Returns:
            obs, reward, done, info
        """
        gs = self.gs

        # Transform actions to gains in [ε, 1]
        gains_flat = 1.0 / (1.0 + np.exp(-gains_action))  # sigmoid
        gains_flat = ETA_GAIN_FLOOR + (1.0 - ETA_GAIN_FLOOR) * gains_flat
        gains_2d = gains_flat.reshape(gs, gs)

        # Dykstra projection
        overlaps = compute_overlaps(self.positions, self.widths, self.heights)
        if overlaps:
            proj_disp = dykstra_projection(self.positions, self.widths,
                                            self.heights, overlaps, n_rounds=5)

            # Apply per-G-cell gains
            per_cell_gain = np.ones(self.N)
            for idx in range(self.N):
                gx, gy = self._gcell_assign.get(idx, (0, 0))
                per_cell_gain[idx] = gains_2d[gy, gx]

            displacement = proj_disp * 0.3 * per_cell_gain[:, np.newaxis]

            # Add small anchor force
            displacement += 0.05 * (self.anchor_positions - self.positions)

            self.positions += displacement

        # Clamp to die area
        if self.die_area is not None:
            self.positions[:, 0] = np.clip(self.positions[:, 0],
                                           self.die_area["x_min"],
                                           self.die_area["x_max"])
            self.positions[:, 1] = np.clip(self.positions[:, 1],
                                           self.die_area["y_min"],
                                           self.die_area["y_max"])

        # Compute new metrics
        overlaps_new = compute_overlaps(self.positions, self.widths, self.heights)
        n_overlaps = len(overlaps_new)
        overlap_area = sum(a for _, _, a in overlaps_new)

        self.step_count += 1
        if self.step_count % 3 == 0 or self.step_count <= 3:
            _, eta_map, _, dbar_map, gcell_assign, ncells_map = \
                compute_gcell_metrics(self.positions, self.widths,
                                      self.heights, self.die_area,
                                      self.gs, self.r_interact)
            self._gcell_assign = gcell_assign
            self._eta_map = eta_map
            self._dbar_map = dbar_map
            self._ncells_map = ncells_map

        eta_max = float(np.max(self._eta_map))

        # Reward: area reduction + overlap reduction
        area_ratio = overlap_area / max(0.01, self.prev_overlap_area)
        overlaps_resolved = self.prev_n_overlaps - n_overlaps

        reward = -2.0 * area_ratio  # penalize remaining area
        reward += 0.1 * max(0, overlaps_resolved)  # bonus for resolving
        if n_overlaps == 0:
            reward += 10.0

        self.prev_overlap_area = overlap_area
        self.prev_eta_max = eta_max
        self.prev_n_overlaps = n_overlaps

        done = self.step_count >= self.max_steps or n_overlaps == 0

        info = {
            "n_overlaps": n_overlaps,
            "overlap_area": float(overlap_area),
            "eta_max": float(eta_max),
            "overlaps_resolved": overlaps_resolved,
            "step": self.step_count,
        }

        return self._get_obs(), float(reward), done, info


# ═══════════════════════════════════════════════════════════════════
# Baseline Agents
# ═══════════════════════════════════════════════════════════════════

class UniformGainAgent:
    """Fixed gain = 1.0 for all G-cells (standard Dykstra, no shield)."""
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return np.zeros(self.env.action_dim, dtype=np.float32) + 2.0  # sigmoid(2)≈0.88→gain≈0.9


class EtaShieldAgent:
    """η-shield: gain = max(ε, 1-η_theory) per G-cell (Phase 1 baseline)."""
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        gs = self.env.gs
        gains_raw = np.zeros(gs * gs, dtype=np.float32)
        for gx in range(gs):
            for gy in range(gs):
                dbar_local = self.env._dbar_map[gy, gx]
                eta_th = theory_eta(dbar_local, 2)
                target_gain = max(ETA_GAIN_FLOOR, 1.0 - eta_th)
                # Invert sigmoid: raw = log(p/(1-p)) where p = (gain-ε)/(1-ε)
                p = (target_gain - ETA_GAIN_FLOOR) / (1.0 - ETA_GAIN_FLOOR)
                p = np.clip(p, 0.01, 0.99)
                gains_raw[gy * gs + gx] = np.log(p / (1 - p))
        return gains_raw


class FullGainAgent:
    """Full gain = 1.0 everywhere (maximum aggressiveness)."""
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return np.ones(self.env.action_dim, dtype=np.float32) * 5.0  # sigmoid(5)≈1→gain≈1


# ═══════════════════════════════════════════════════════════════════
# PPO Agent (requires PyTorch)
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class GainPolicy(nn.Module):
        """Policy: observe per-G-cell features, output per-G-cell gains.

        Input: (gs*gs*5,) flattened G-cell features
        Output: (gs*gs,) raw gain logits (sigmoid → actual gains)
        """
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(64, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, obs):
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            h = self.net(obs)
            mean = self.mean_head(h)
            std = self.log_std.exp().expand_as(mean)
            return mean, std

        def get_action(self, obs_np):
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std = self.forward(obs_t)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            return action.squeeze(0).numpy(), log_prob.item()

        def evaluate(self, obs_t, actions_t):
            mean, std = self.forward(obs_t)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_probs, entropy

    class GainValue(nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, obs):
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            return self.net(obs).squeeze(-1)

    def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
        """Compute GAE advantages."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else 0
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        returns = advantages + np.array(values[:T])
        return advantages, returns

    def ppo_update(policy, value_net, optimizer, obs_batch, act_batch,
                   old_log_probs, advantages, returns):
        """One PPO update epoch."""
        obs_t = torch.tensor(obs_batch, dtype=torch.float32)
        act_t = torch.tensor(act_batch, dtype=torch.float32)
        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            # Shuffle
            idx = torch.randperm(len(obs_t))
            for start in range(0, len(idx), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(idx))
                mb = idx[start:end]

                log_probs, entropy = policy.evaluate(obs_t[mb], act_t[mb])
                values = value_net(obs_t[mb])

                ratio = (log_probs - old_lp_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[mb]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[mb])
                entropy_loss = -entropy.mean()

                loss = policy_loss + VALUE_COEFF * value_loss + ENTROPY_COEFF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_net.parameters()), 0.5)
                optimizer.step()

        return float(policy_loss), float(value_loss)


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_agent(env, agent, n_episodes=10, label="agent"):
    """Run agent for n_episodes, return metrics."""
    all_final_overlaps = []
    all_final_areas = []
    all_final_etas = []
    all_total_rewards = []

    for ep in range(n_episodes):
        obs = env.reset(seed=42 + ep * 7)
        total_reward = 0
        done = False
        info = {}

        while not done:
            if hasattr(agent, 'act'):
                actions = agent.act(obs)
            elif HAS_TORCH and hasattr(agent, 'get_action'):
                actions, _ = agent.get_action(obs)
            else:
                actions = np.zeros(env.action_dim, dtype=np.float32)

            obs, reward, done, info = env.step(actions)
            total_reward += reward

        all_final_overlaps.append(info.get("n_overlaps", 0))
        all_final_areas.append(info.get("overlap_area", 0))
        all_final_etas.append(info.get("eta_max", 0))
        all_total_rewards.append(total_reward)

    return {
        "label": label,
        "mean_final_overlaps": float(np.mean(all_final_overlaps)),
        "std_final_overlaps": float(np.std(all_final_overlaps)),
        "mean_final_area": float(np.mean(all_final_areas)),
        "mean_final_eta": float(np.mean(all_final_etas)),
        "mean_reward": float(np.mean(all_total_rewards)),
        "raw_overlaps": [int(x) for x in all_final_overlaps],
        "raw_areas": [float(x) for x in all_final_areas],
    }


# ═══════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════

def train_ppo(env, n_episodes=EPISODES_DEFAULT):
    """Train PPO agent on per-G-cell gains."""
    if not HAS_TORCH:
        print("ERROR: PyTorch required for PPO training")
        return None

    obs_dim = env.gs * env.gs * 5  # flattened G-cell features
    act_dim = env.action_dim        # gs*gs gain factors

    policy = GainPolicy(obs_dim, act_dim)
    value_net = GainValue(obs_dim)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=LR
    )

    ep_rewards = []
    ep_overlaps = []
    ep_areas = []
    best_reward = -float('inf')

    print(f"\n{'='*60}")
    print(f"PPO Training: {n_episodes} episodes, {env.max_steps} steps/ep")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        obs = env.reset(seed=42 + ep)

        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values, all_dones = [], [], []

        done = False
        while not done:
            action, log_prob = policy.get_action(obs)

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                value = value_net(obs_t).item()

            obs_next, reward, done, info = env.step(action)

            all_obs.append(obs)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
            all_values.append(value)
            all_dones.append(done)

            obs = obs_next

        total_reward = sum(all_rewards)
        final_overlaps = info.get("n_overlaps", 0)
        final_area = info.get("overlap_area", 0)
        ep_rewards.append(total_reward)
        ep_overlaps.append(final_overlaps)
        ep_areas.append(final_area)

        # PPO update
        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones)

        obs_batch = np.array(all_obs)
        act_batch = np.array(all_actions)
        lp_batch = np.array(all_log_probs)

        p_loss, v_loss = ppo_update(
            policy, value_net, optimizer,
            obs_batch, act_batch, lp_batch, advantages, returns)

        if total_reward > best_reward:
            best_reward = total_reward

        if ep % 20 == 0 or ep == n_episodes - 1:
            recent_r = np.mean(ep_rewards[-20:])
            recent_ov = np.mean(ep_overlaps[-20:])
            recent_a = np.mean(ep_areas[-20:])
            print(f"  ep {ep:4d}: r={total_reward:7.1f} "
                  f"ov={final_overlaps:4d} area={final_area:7.2f} "
                  f"| avg20: r={recent_r:7.1f} ov={recent_ov:.0f} "
                  f"a={recent_a:.2f} | ploss={p_loss:.4f}")

    return policy, ep_rewards, ep_overlaps, ep_areas


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def plot_training(ep_rewards, ep_overlaps, ep_areas, eval_results,
                  design_name, results_dir):
    """Plot training curves and baseline comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"η-RL Placement: {design_name}", fontsize=14)

    # (a) Training reward
    ax = axes[0, 0]
    ax.plot(ep_rewards, alpha=0.3, color='blue')
    window = min(20, len(ep_rewards) // 3)
    if window > 1:
        smoothed = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ep_rewards)), smoothed, color='blue', lw=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("(a) Training reward")
    ax.grid(True, alpha=0.3)

    # (b) Training overlaps
    ax = axes[0, 1]
    ax.plot(ep_overlaps, alpha=0.3, color='red')
    if window > 1:
        smoothed = np.convolve(ep_overlaps, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ep_overlaps)), smoothed, color='red', lw=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final overlap count")
    ax.set_title("(b) Final overlaps during training")
    ax.grid(True, alpha=0.3)

    # (c) Baseline comparison — overlaps
    ax = axes[1, 0]
    labels_list = [r["label"] for r in eval_results]
    means = [r["mean_final_overlaps"] for r in eval_results]
    stds = [r["std_final_overlaps"] for r in eval_results]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax.bar(labels_list, means, yerr=stds, color=colors[:len(labels_list)],
                  capsize=5, alpha=0.8)
    ax.set_ylabel("Mean final overlaps")
    ax.set_title("(c) Baseline comparison: overlaps")
    ax.grid(True, alpha=0.3, axis='y')

    # (d) Baseline comparison — area
    ax = axes[1, 1]
    areas = [r["mean_final_area"] for r in eval_results]
    bars = ax.bar(labels_list, areas, color=colors[:len(labels_list)], alpha=0.8)
    ax.set_ylabel("Mean final overlap area (μm²)")
    ax.set_title("(d) Baseline comparison: overlap area")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"eta_rl_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="η-RL Placement")
    parser.add_argument("--design", type=str, default="gcd_nangate45")
    parser.add_argument("--episodes", type=int, default=EPISODES_DEFAULT)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--compress", type=float, default=0.3)
    parser.add_argument("--region", type=str, default="center")
    parser.add_argument("--max-cells", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    steps_per_ep = args.steps

    results_dir = os.path.join(os.path.dirname(__file__), "results", "rl")
    os.makedirs(results_dir, exist_ok=True)

    # Create environment
    env = PlacementEnv(
        design_name=args.design,
        compress_factor=args.compress,
        compress_region=args.region,
        max_cells=args.max_cells,
        max_steps=steps_per_ep,
    )
    print(f"\n  Environment: {env.N} cells, gs={env.gs}")

    # Evaluate baselines
    print(f"\n{'='*60}")
    print("Evaluating baselines...")
    print(f"{'='*60}")

    eval_results = []

    # Uniform gain (no shield)
    print("\n  [uniform]")
    r = evaluate_agent(env, UniformGainAgent(env),
                       n_episodes=args.eval_episodes, label="uniform")
    eval_results.append(r)
    print(f"    overlaps: {r['mean_final_overlaps']:.1f} ± {r['std_final_overlaps']:.1f}, "
          f"area: {r['mean_final_area']:.2f}")

    # Full gain (aggressive)
    print("\n  [full-gain]")
    r = evaluate_agent(env, FullGainAgent(env),
                       n_episodes=args.eval_episodes, label="full-gain")
    eval_results.append(r)
    print(f"    overlaps: {r['mean_final_overlaps']:.1f} ± {r['std_final_overlaps']:.1f}, "
          f"area: {r['mean_final_area']:.2f}")

    # η-shield (theory)
    print("\n  [η-shield]")
    r = evaluate_agent(env, EtaShieldAgent(env),
                       n_episodes=args.eval_episodes, label="η-shield")
    eval_results.append(r)
    print(f"    overlaps: {r['mean_final_overlaps']:.1f} ± {r['std_final_overlaps']:.1f}, "
          f"area: {r['mean_final_area']:.2f}")

    # PPO
    if not args.baselines_only and HAS_TORCH:
        print(f"\n{'='*60}")
        print("Training PPO...")
        print(f"{'='*60}")

        policy, ep_rewards, ep_overlaps, ep_areas = train_ppo(
            env, n_episodes=args.episodes)

        print("\n  [ppo] Evaluating trained policy...")
        r = evaluate_agent(env, policy, n_episodes=args.eval_episodes, label="ppo")
        eval_results.append(r)
        print(f"    overlaps: {r['mean_final_overlaps']:.1f} ± {r['std_final_overlaps']:.1f}, "
              f"area: {r['mean_final_area']:.2f}")
    else:
        ep_rewards, ep_overlaps, ep_areas = [], [], []

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.design}")
    print(f"{'='*60}")
    print(f"  {'Agent':>12s}  {'Overlaps':>12s}  {'Area':>10s}  {'Reward':>10s}")
    print(f"  {'─'*50}")
    for r in eval_results:
        print(f"  {r['label']:>12s}  "
              f"{r['mean_final_overlaps']:>8.1f}±{r['std_final_overlaps']:<3.1f}  "
              f"{r['mean_final_area']:>10.2f}  "
              f"{r['mean_reward']:>10.1f}")

    # Save
    summary = {
        "design": args.design,
        "N": env.N,
        "compress": args.compress,
        "region": args.region,
        "episodes": args.episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "eval_results": eval_results,
        "training": {
            "rewards": [float(x) for x in ep_rewards],
            "overlaps": [int(x) for x in ep_overlaps],
            "areas": [float(x) for x in ep_areas],
        } if ep_rewards else None,
    }

    json_path = os.path.join(results_dir, f"eta_rl_{args.design}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    if ep_rewards or len(eval_results) > 2:
        plot_training(ep_rewards, ep_overlaps, ep_areas, eval_results,
                      args.design, results_dir)


if __name__ == "__main__":
    main()

"""
Run a naive multi-agent MPPI example where two robots start near each other,
head to the same goal, and treat the other robot as a static obstacle during
rollout cost evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Tuple, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from controller.mppi import MPPI
from envs.navigation_2d import Navigation2DEnv
from envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles

COLORS = ["blue", "orange", "green", "purple", "red", "cyan"]


def _make_shared_map(device: torch.device, dtype: torch.dtype, seed: int) -> ObstacleMap:
    """
    Build a fixed map with a straight canal whose width pinches then widens.
    """
    obstacle_map = ObstacleMap(map_size=(20, 20), cell_size=0.1, device=device, dtype=dtype)

    # Canal walls made of rectangles along the diagonal y = x.
    # Width profile: wide -> narrow -> wide to create a choke point.
    segment_centers = np.linspace(-10, 10, num=20)
    half_widths = np.linspace(2.5, 1.5, num=len(segment_centers) // 2).tolist() + \
                  np.linspace(1.5, 2.5, num=len(segment_centers) - len(segment_centers) // 2).tolist()
    rect_length_along_canal = 2
    rect_thickness = 1.0

    for t, hw in zip(segment_centers, half_widths):
        # normal to the diagonal (1,1) is (1,-1)/sqrt(2)
        offset = hw / np.sqrt(2)
        center_up = np.array([t + offset, t - offset])
        center_down = np.array([t - offset, t + offset])
        obstacle_map.add_rectangle_obstacle(center=center_up, width=rect_length_along_canal, height=rect_thickness)
        obstacle_map.add_rectangle_obstacle(center=center_down, width=rect_length_along_canal, height=rect_thickness)

    obstacle_map.convert_to_torch()
    return obstacle_map


def _make_envs(
    num_agents: int = 4,
    seed: int = 2025,
    peer_safe_distance: float = 0.4,
) -> List[Navigation2DEnv]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    shared_map = _make_shared_map(device=device, dtype=dtype, seed=seed)
    goal = torch.tensor([9.0, 9.0], device=device, dtype=dtype)

    # Cluster initial positions near the canal entrance so agents block each other unless they separate.
    base_start = torch.tensor([-8.5, -8.5], device=device, dtype=dtype)
    offsets = [
        torch.tensor([0.0, 0.0], device=device, dtype=dtype),
        torch.tensor([1, 1], device=device, dtype=dtype),
        torch.tensor([0.5, 0.5], device=device, dtype=dtype),
        torch.tensor([0.2, 0.3], device=device, dtype=dtype),
    ]
    envs: List[Navigation2DEnv] = []
    for idx in range(num_agents):
        start_pos = base_start + offsets[idx % len(offsets)]
        envs.append(
            Navigation2DEnv(
                v_min=0.0,
                v_max=2.0,
                omega_min=-1.0,
                omega_max=1.0,
                seed=seed + idx,
                device=device,
                dtype=dtype,
                start_pos=start_pos,
                goal_pos=goal,
                obstacle_map=shared_map,
                peer_safe_distance=peer_safe_distance,
            )
        )

    return envs


def _make_solver(env: Navigation2DEnv, num_samples: int = 200) -> MPPI:
    device = env.u_min.device
    dtype = env.u_min.dtype
    sigmas = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
    return MPPI(
        env=env,
        horizon=20,
        num_samples=num_samples,
        sigmas=sigmas,
        lambda_=1.0,
    )


def _render_frame(
    obstacle_map: ObstacleMap,
    goal: np.ndarray,
    current_positions: np.ndarray,
    predicted: Optional[List[np.ndarray]] = None,
    sample_sets: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> np.ndarray:
    """
    Render a frame with current positions, predicted optimal rollouts, and sampled rollouts.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    obstacle_map.render(ax, zorder=0)

    ax.scatter(goal[0], goal[1], c="black", marker="*", s=120, label="goal")

    for i, pos in enumerate(current_positions):
        color = COLORS[i % len(COLORS)]
        # current location
        ax.scatter(pos[0], pos[1], color=color, marker="o", s=30, label=f"agent {i+1}")

        # optimal predicted rollout from current step
        if predicted is not None and predicted[i] is not None:
            ax.plot(predicted[i][:, 0], predicted[i][:, 1], color=color, linestyle="--", alpha=0.9)

        # sampled rollouts
        if sample_sets is not None and sample_sets[i] is not None:
            samples, weights = sample_sets[i]
            if len(weights) > 0:
                weights = weights / np.max(weights)
                weights = np.clip(weights, 0.05, 1.0)
            for j, sample in enumerate(samples):
                alpha = 0.2 if len(weights) == 0 else 0.1 + 0.6 * weights[j]
                ax.plot(sample[:, 0], sample[:, 1], color=color, linewidth=1, alpha=alpha)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Multi-agent MPPI navigation (current + rollouts)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def run_multi_agent_navigation(
    max_steps: int = 200,
    num_agents: int = 4,
    peer_safe_distance: float = 0.8,
    save_gif_path: str | None = None,
    progress_fn: Optional[Callable[[int], None]] = None,
    use_moving_obstacles: bool = False,
    communication_radius: float = 4.0,
) -> List[np.ndarray]:
    """
    Drive multiple MPPI agents toward the same goal while keeping them separated.
    Returns:
        List of numpy arrays containing xy waypoints for each agent.
    """
    envs = _make_envs(num_agents=num_agents, peer_safe_distance=peer_safe_distance)
    solvers = [_make_solver(env) for env in envs]

    states = [env.reset() for env in envs]
    trajectories: List[List[np.ndarray]] = [[s[:2].cpu().numpy()] for s in states]

    goal = envs[0]._goal_pos.cpu().numpy()
    starts = np.stack([env._start_pos.cpu().numpy() for env in envs], axis=0)
    frames = []

    prev_predicted_trajs: List[Optional[np.ndarray]] = [None] * num_agents

    for step_idx in range(max_steps):
        positions = torch.stack([s[:2] for s in states], dim=0)

        infos: List[dict] = []
        for idx in range(num_agents):
            info = {}
            if use_moving_obstacles and any(traj is not None for traj in prev_predicted_trajs):
                neighbor_trajs = []
                for j in range(num_agents):
                    if j == idx:
                        continue
                    if prev_predicted_trajs[j] is None:
                        continue
                    if torch.norm(positions[idx] - positions[j]) <= communication_radius:
                        neighbor_trajs.append(prev_predicted_trajs[j][..., :2])
                if len(neighbor_trajs) > 0:
                    info["moving_agents"] = torch.stack(
                        [torch.as_tensor(traj) for traj in neighbor_trajs], dim=0
                    )
                    info["moving_agent_radius"] = peer_safe_distance
                else:
                    mask = torch.ones(num_agents, dtype=torch.bool, device=positions.device)
                    mask[idx] = False
                    peer_positions = positions[mask]
                    info["static_agents"] = peer_positions
                    info["static_agent_radius"] = peer_safe_distance
            else:
                mask = torch.ones(num_agents, dtype=torch.bool, device=positions.device)
                mask[idx] = False
                peer_positions = positions[mask]
                info["static_agents"] = peer_positions
                info["static_agent_radius"] = peer_safe_distance

            infos.append(info)

        next_states: List[torch.Tensor] = []
        reached_flags: List[bool] = []
        predicted_state_seqs: List[np.ndarray] = [None] * num_agents
        top_sample_sets: List[Tuple[np.ndarray, np.ndarray]] = [None] * num_agents
        for idx, (env, solver, info) in enumerate(zip(envs, solvers, infos)):
            action_seq, state_seq = solver.forward(state=states[idx], info=info)
            next_state, reached = env.step(action_seq[0, :])
            next_states.append(next_state)
            reached_flags.append(reached)
            trajectories[idx].append(next_state[:2].cpu().numpy())

            # keep collision check for debugging/visualization if needed
            _ = env.collision_check(
                state_seq,
                static_agents=info.get("static_agents"),
                moving_agents=info.get("moving_agents"),
            )

            predicted_state_seqs[idx] = state_seq.squeeze(0).detach().cpu().numpy()
            try:
                top_samples, top_weights = solver.get_top_samples(num_samples=min(30, solver._num_samples))
                top_sample_sets[idx] = (
                    top_samples.detach().cpu().numpy(),
                    top_weights.detach().cpu().numpy(),
                )
            except Exception:
                top_sample_sets[idx] = None

        states = next_states
        prev_predicted_trajs = predicted_state_seqs

        if save_gif_path is not None:
            frame = _render_frame(
                obstacle_map=envs[0]._obstacle_map,
                goal=goal,
                current_positions=positions.cpu().numpy(),
                predicted=predicted_state_seqs,
                sample_sets=top_sample_sets,
            )
            frames.append(frame)

        if all(reached_flags):
            break

        if progress_fn is not None:
            progress_fn(step_idx + 1)

    if save_gif_path is not None and len(frames) > 0:
        out_path = Path(save_gif_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=10)

    return [np.stack(traj, axis=0) for traj in trajectories]


if __name__ == "__main__":
    trajectories = run_multi_agent_navigation(num_agents=4, save_gif_path="video/multi_agent.gif")
    for idx, traj in enumerate(trajectories, start=1):
        print(f"Agent {idx} trajectory length: {len(traj)} steps")

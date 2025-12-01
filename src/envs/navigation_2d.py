"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from matplotlib import pyplot as plt

import torch
import numpy as np
import os

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles

from IPython import display

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Navigation2DEnv:
    def __init__(
        self,
        v_min=0.0,
        v_max=2.0,
        omega_min=-1.0,
        omega_max=1.0,
        seed: int = 42,
        device=torch.device("cuda"),
        dtype=torch.float32,
        start_pos: Optional[Union[Tuple[float, float], torch.Tensor]] = None,
        goal_pos: Optional[Union[Tuple[float, float], torch.Tensor]] = None,
        obstacle_map: Optional[ObstacleMap] = None,
        peer_safe_distance: float = 0.75,
        peer_cost_weight: float = 10000.0,
    ) -> None:
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        self._seed = seed

        if obstacle_map is None:
            self._obstacle_map = ObstacleMap(
                map_size=(20, 20), cell_size=0.1, device=self._device, dtype=self._dtype
            )

            generate_random_obstacles(
                obstacle_map=self._obstacle_map,
                random_x_range=(-7.5, 7.5),
                random_y_range=(-7.5, 7.5),
                num_circle_obs=7,
                radius_range=(1, 1),
                num_rectangle_obs=7,
                width_range=(2, 2),
                height_range=(2, 2),
                max_iteration=1000,
                seed=seed,
            )
            self._obstacle_map.convert_to_torch()
        else:
            # reuse a pre-generated obstacle map so multiple agents can share the same world
            self._obstacle_map = obstacle_map
            if getattr(self._obstacle_map, "_map_torch", None) is None:
                self._obstacle_map.convert_to_torch()

        start_pos = (
            start_pos
            if start_pos is not None
            else torch.tensor([-9.0, -9.0], device=self._device, dtype=self._dtype)
        )
        goal_pos = (
            goal_pos
            if goal_pos is not None
            else torch.tensor([9.0, 9.0], device=self._device, dtype=self._dtype)
        )

        self._start_pos = torch.as_tensor(start_pos, device=self._device, dtype=self._dtype)
        self._goal_pos = torch.as_tensor(goal_pos, device=self._device, dtype=self._dtype)

        self._peer_safe_distance = peer_safe_distance
        self._peer_cost_weight = peer_cost_weight

        self._robot_state = torch.zeros(3, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        # u: [v, omega] (m/s, rad/s)
        self.u_min = torch.tensor([v_min, omega_min], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([v_max, omega_max], device=self._device, dtype=self._dtype)

    def reset(self) -> torch.Tensor:
        """
        Reset robot state.
        Returns:
            torch.Tensor: shape (3,) [x, y, theta]
        """
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._rendered_frames = []

        return self._robot_state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update robot state based on differential drive dynamics.
        Args:
            u (torch.Tensor): control batch tensor, shape (2) [v, omega]
        Returns:
            Tuple[torch.Tensor, bool]: Tuple of robot state and is goal reached.
        """
        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        # goal check
        goal_threshold = 0.5
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        )

        return self._robot_state, is_goal_reached

    def plot(self) -> None:
        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")
        """plot the environment using matplotlib"""
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        # obstacle map
        self._obstacle_map.render(self._ax, zorder=10)

        # start and goal
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )
        
    def render(
        self,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        mode: str = "human",
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        # obstacle map
        self._obstacle_map.render(self._ax, zorder=10)

        # start and goal
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )

        # robot
        self._ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            marker="o",
            color="green",
            zorder=100,
        )

        # visualize top samples with different alpha based on weights
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )

        # predicted trajectory
        if predicted_trajectory is not None:
            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        if mode == "human":
            # online rendering
            display.clear_output(wait=True)
            display.display(self._fig)
            self._ax.cla()
        elif mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            buf = self._fig.canvas.buffer_rgba()
            data_rgba = np.asarray(buf)
            data_rgb = data_rgba[..., :3]
            self._ax.cla()
            self._rendered_frames.append(data_rgb.copy())

    def close(self, path: str = None) -> None:
        if path is None:
            # mkdir video if not exists

            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/" + "navigation_2d" + ".gif"

        if len(self._rendered_frames) > 0:
            # save animation
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            # clip.write_videofile(path, fps=10)
            clip.write_gif(path, fps=10)
        plt.close(self._fig)

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        omega = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        new_x = x + v * torch.cos(theta) * delta_t
        new_y = y + v * torch.sin(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        result = torch.cat([clamped_x, clamped_y, new_theta], dim=1)

        return result

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
        Returns:
            torch.Tensor: shape (batch_size,)
        """

        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)

        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(1)  # (batch_size,)

        # optional static-agent penalty so other robots can be treated as obstacles
        static_agents = info.get("static_agents", None)
        static_agent_radius = info.get("static_agent_radius", self._peer_safe_distance)
        static_agent_weight = info.get("static_agent_weight", self._peer_cost_weight)
        peer_cost = 0.0
        if static_agents is not None:
            peer_positions = static_agents
            if not torch.is_tensor(peer_positions):
                peer_positions = torch.tensor(peer_positions, device=self._device, dtype=self._dtype)
            else:
                peer_positions = peer_positions.to(self._device, self._dtype)

            if peer_positions.dim() == 1:
                peer_positions = peer_positions.unsqueeze(0)

            peer_positions = peer_positions[..., :2]
            peer_positions = peer_positions.unsqueeze(0)  # (1, num_peer, 2)
            agent_positions = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)

            peer_dist = torch.norm(agent_positions - peer_positions, dim=2)
            proximity = torch.clamp(static_agent_radius - peer_dist, min=0.0)
            peer_cost = torch.sum(proximity**2, dim=1)

        # optional moving-agent penalty (time-varying trajectories)
        moving_agents = info.get("moving_agents", None)
        moving_agent_radius = info.get("moving_agent_radius", self._peer_safe_distance)
        moving_agent_weight = info.get("moving_agent_weight", self._peer_cost_weight)
        moving_cost = 0.0
        if moving_agents is not None:
            trajs = moving_agents
            if not torch.is_tensor(trajs):
                trajs = torch.tensor(trajs, device=self._device, dtype=self._dtype)
            else:
                trajs = trajs.to(self._device, self._dtype)

            # expected shape: (num_peers, horizon_len, 2 or 3)
            if trajs.dim() == 2:
                trajs = trajs.unsqueeze(0)
            trajs = trajs[..., :2]

            # pick the waypoint at the current rollout time
            t_idx = info.get("t", 0)
            t_idx = int(t_idx)
            t_idx = max(0, min(t_idx, trajs.shape[1] - 1))
            peer_positions_t = trajs[:, t_idx, :]  # (num_peers, 2)
            peer_positions_t = peer_positions_t.unsqueeze(0)  # (1, num_peers, 2)

            agent_positions = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
            peer_dist = torch.norm(agent_positions - peer_positions_t, dim=2)
            proximity = torch.clamp(moving_agent_radius - peer_dist, min=0.0)
            moving_cost = torch.sum(proximity**2, dim=1)

        cost = (
            goal_cost
            + 10000 * obstacle_cost
            + static_agent_weight * peer_cost
            + moving_agent_weight * moving_cost
        )

        return cost


    def collision_check(
        self,
        state: torch.Tensor,
        static_agents: Optional[torch.Tensor] = None,
        moving_agents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, traj_size , 3) [x, y, theta]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        pos_batch = state[:, :, :2]
        is_collisions = self._obstacle_map.compute_cost(pos_batch).squeeze(1)

        if static_agents is not None:
            peer_positions = static_agents
            if not torch.is_tensor(peer_positions):
                peer_positions = torch.tensor(peer_positions, device=self._device, dtype=self._dtype)
            else:
                peer_positions = peer_positions.to(self._device, self._dtype)

            if peer_positions.dim() == 1:
                peer_positions = peer_positions.unsqueeze(0)

            peer_positions = peer_positions[..., :2]
            agent_positions = pos_batch.unsqueeze(2)  # (batch_size, traj, 1, 2)
            peer_positions = peer_positions.view(1, 1, -1, 2)  # (1, 1, num_peer, 2)
            peer_dist = torch.norm(agent_positions - peer_positions, dim=3)
            peer_collisions = torch.any(
                peer_dist < self._peer_safe_distance, dim=2
            ).to(self._dtype)
            is_collisions = torch.maximum(is_collisions, peer_collisions)

        if moving_agents is not None:
            trajs = moving_agents
            if not torch.is_tensor(trajs):
                trajs = torch.tensor(trajs, device=self._device, dtype=self._dtype)
            else:
                trajs = trajs.to(self._device, self._dtype)

            if trajs.dim() == 2:
                trajs = trajs.unsqueeze(0)
            trajs = trajs[..., :2]  # (num_peers, horizon, 2)

            horizon_len = min(trajs.shape[1], pos_batch.shape[1])
            # align time steps
            peer_positions = trajs[:, :horizon_len, :]  # (num_peers, T, 2)
            peer_positions = peer_positions.permute(1, 0, 2).unsqueeze(0)  # (1, T, num_peers, 2)
            agent_positions = pos_batch[:, :horizon_len, :].unsqueeze(2)  # (batch, T, 1, 2)
            peer_dist = torch.norm(agent_positions - peer_positions, dim=3)
            peer_collisions = torch.any(
                peer_dist < self._peer_safe_distance, dim=2
            ).to(self._dtype)
            is_collisions = torch.maximum(is_collisions, peer_collisions)

        return is_collisions

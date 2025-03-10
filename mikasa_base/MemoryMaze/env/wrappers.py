from typing import Any, Dict, List

import dm_env
import numpy as np
from dm_env import specs


class Wrapper(dm_env.Environment):
    """Base class for dm_env.Environment wrapper."""

    def __init__(self, env: dm_env.Environment):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(f"Attempted to get missing private attribute {name}")
        return getattr(self.env, name)

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(action)

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def action_spec(self) -> Any:
        return self.env.action_spec()

    def discount_spec(self) -> Any:
        return self.env.discount_spec()

    def observation_spec(self) -> Any:
        return self.env.observation_spec()

    def reward_spec(self) -> Any:
        return self.env.reward_spec()

    def close(self):
        return self.env.close()


class ObservationWrapper(Wrapper):
    """Base class for observation wrapper."""

    def observation_spec(self):
        raise NotImplementedError

    def observation(self, obs: Any) -> Any:
        raise NotImplementedError

    def step(self, action) -> dm_env.TimeStep:
        step_type, discount, reward, observation = self.env.step(action)
        return dm_env.TimeStep(
            step_type, discount, reward, self.observation(observation)
        )

    def reset(self) -> dm_env.TimeStep:
        step_type, discount, reward, observation = self.env.reset()
        return dm_env.TimeStep(
            step_type, discount, reward, self.observation(observation)
        )


class RemapObservationWrapper(ObservationWrapper):
    """Select a subset of dictionary observation keys and rename them."""

    def __init__(self, env: dm_env.Environment, mapping: Dict[str, str]):
        super().__init__(env)
        self.mapping = mapping

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        return {key: spec[key_orig] for key, key_orig in self.mapping.items()}

    def observation(self, obs):
        assert isinstance(obs, dict)
        return {key: obs[key_orig] for key, key_orig in self.mapping.items()}


class TargetsPositionWrapper(ObservationWrapper):
    """Collects and postporcesses walker/target_rel_{i} relative position vectors into
    targets_vec (n_targets,2) tensor, and walker/targets_abs_{i} absolute positions
    into targets_pos tensor."""

    def __init__(self, env: dm_env.Environment, maze_xy_scale, maze_width, maze_height):
        super().__init__(env)
        self.maze_xy_scale = maze_xy_scale
        self.center_ji = np.array([maze_width - 2.0, maze_height - 2.0]) / 2.0

        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert "walker/target_rel_0" in spec
        assert "walker/target_abs_0" in spec
        assert "target_index" in spec

        i = 0
        while f"walker/target_rel_{i}" in spec:
            assert f"walker/target_abs_{i}" in spec
            i += 1

        self.n_targets = i

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        # All targets
        spec["targets_vec"] = specs.Array((self.n_targets, 2), float, "targets_vec")
        spec["targets_pos"] = specs.Array((self.n_targets, 2), float, "targets_pos")
        # Current target
        spec["target_vec"] = specs.Array((2,), float, "target_vec")
        spec["target_pos"] = specs.Array((2,), float, "target_pos")
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        # All targets
        x_rel = np.zeros((self.n_targets, 2))
        x_abs = np.zeros((self.n_targets, 2))
        for i in range(self.n_targets):
            x_rel[i] = obs[f"walker/target_rel_{i}"][:2] / self.maze_xy_scale
            x_abs[i] = (
                obs[f"walker/target_abs_{i}"][:2] / self.maze_xy_scale + self.center_ji
            )
        obs["targets_vec"] = x_rel
        obs["targets_pos"] = x_abs
        # Current target
        target_ix = int(obs["target_index"])
        obs["target_vec"] = x_rel[target_ix]
        obs["target_pos"] = x_abs[target_ix]
        return obs


class AgentPositionWrapper(ObservationWrapper):
    """Postprocesses absolute_position and absolute_orientation."""

    def __init__(self, env: dm_env.Environment, maze_xy_scale, maze_width, maze_height):
        super().__init__(env)
        self.maze_xy_scale = maze_xy_scale
        self.center_ji = np.array([maze_width - 2.0, maze_height - 2.0]) / 2.0

    def observation_spec(self):
        spec = self.env.observation_spec()
        # absolute_position and absolute_orientation should already be generated by the environment.
        assert (
            isinstance(spec, dict)
            and "absolute_position" in spec
            and "absolute_orientation" in spec
        )
        # Add agent_pos, measured in grid coordinates
        spec["agent_pos"] = specs.Array((2,), float, "agent_pos")
        # Add agent_dir as 2-vector
        spec["agent_dir"] = specs.Array((2,), float, "agent_dir")
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        walker_xy = obs["absolute_position"][:2]
        walker_ji = walker_xy / self.maze_xy_scale + self.center_ji
        # agent_pos, measured in grid coordinates, where bottom-left coordinate is (0.1,0.1),
        # and top-right coordinate for a 15x15 maze is (14.9,14.9)
        obs["agent_pos"] = walker_ji
        # Pick orientation vector such, that going forward increases agent_pos in the direction of agent_dir.
        obs["agent_dir"] = obs["absolute_orientation"][:2, 1]
        return obs


class MazeLayoutWrapper(ObservationWrapper):
    """Postprocesses maze_layout observation."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        # maze_layout should already be generated by the environment
        assert isinstance(spec, dict) and "maze_layout" in spec
        # Change char array to binary array, removing outer walls
        n, m = spec["maze_layout"].shape
        spec["maze_layout"] = specs.BoundedArray(
            (n - 2, m - 2), np.uint8, 0, 1, "maze_layout"
        )
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        maze = obs["maze_layout"]
        maze = maze[1:-1, 1:-1]  # Remove outer walls
        maze = np.flip(
            maze, 0
        )  # Flip vertical axis so that bottom-left is at maze[0,0]
        nonwalls = (maze == " ") | (maze == "P") | (maze == "G")
        obs["maze_layout"] = nonwalls.astype(np.uint8)
        return obs


class ImageOnlyObservationWrapper(ObservationWrapper):
    """Select one of the dictionary observation keys as observation."""

    def __init__(self, env: dm_env.Environment, key: str = "image"):
        super().__init__(env)
        self.key = key

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        return spec[self.key]

    def observation(self, obs):
        assert isinstance(obs, dict)
        return obs[self.key]


class DiscreteActionSetWrapper(Wrapper):
    """Change action space from continuous to discrete with given set of action vectors."""

    def __init__(self, env: dm_env.Environment, action_set: List[np.ndarray]):
        super().__init__(env)
        self.action_set = action_set

    def action_spec(self):
        return specs.DiscreteArray(len(self.action_set))

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(self.action_set[action])


class TargetColorAsBorderWrapper(ObservationWrapper):
    """MemoryMaze-specific wrapper, which draws target_color as border on the image."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert "target_color" in spec
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        assert "target_color" in obs and "image" in obs
        target_color = obs["target_color"]
        img = obs["image"]
        B = int(2 * np.sqrt(img.shape[0] // 64))
        img[:, :B] = target_color * 255 * 0.7
        img[:, -B:] = target_color * 255 * 0.7
        img[:B, :] = target_color * 255 * 0.7
        img[-B:, :] = target_color * 255 * 0.7
        return obs

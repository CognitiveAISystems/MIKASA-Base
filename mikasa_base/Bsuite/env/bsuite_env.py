"""Bsuite adapter for OpenAI gym run-loops."""

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym

from dm_env import specs

from .discounting_chain import DiscountingChain
from .memory_chain import MemoryChain

# OpenAI gym step format (Gymnasium): observation, reward, terminated, truncated, info
GymTimestep = Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]


class BsuiteGymWrapper(gym.Env):
    """Wrapper that converts a dm_env.Environment to a gym.Env.

    Allows bsuite environments to be used with Gymnasium API.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env_id: str, **kwargs):
        """Initialize the BsuiteGymWrapper.

        Args:
            env_id: Identifier of the environment ('MemoryLength' or 'DiscountingChain').
            **kwargs: Additional arguments to create the specific environment.
        """
        if "MemoryLength" in env_id:
            memory_length = kwargs.get("memory_length", 10)
            num_bits = kwargs.get("num_bits", 1)
            self._env = MemoryChain(memory_length, num_bits)
            self.max_episode_steps = kwargs.get("max_episode_steps", memory_length + 1)
        elif "DiscountingChain" in env_id:
            mapping_seed = kwargs.get("mapping_seed", None)
            self._env = DiscountingChain(mapping_seed)
            self.max_episode_steps = kwargs.get("max_episode_steps", 100)
        else:
            raise ValueError(f"Unknown environment identifier: {env_id}")

        self._last_observation: Optional[np.ndarray] = None
        self.viewer = None
        self.game_over = False  # Needed for Dopamine agents.

    def step(self, action: int) -> GymTimestep:
        """Performs a step in the environment.

        Args:
            action: The chosen action.

        Returns:
            A tuple (observation, reward, terminated, truncated, info).
        """
        timestep = self._env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.0
        terminated = timestep.last()
        truncated = timestep.last()
        return (
            timestep.observation.flatten(),
            reward,
            terminated,
            truncated,
            {},
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to an initial state.

        Args:
            seed: (Optional) Seed for the random number generator.
            **kwargs: Additional parameters for resetting the environment.

        Returns:
            A tuple (observation, info) where observation is the initial observation.
        """
        info = {}
        self.game_over = False
        timestep = self._env.reset(seed=seed)
        self._last_observation = timestep.observation
        return timestep.observation.flatten(), info

    def render(self, mode: str = "rgb_array") -> Union[np.ndarray, bool]:
        """Renders the current state of the environment.

        Args:
            mode: Rendering mode ('rgb_array' or 'human').

        Returns:
            If mode == 'rgb_array': the image array;
            if mode == 'human': the viewer status.
        """
        if self._last_observation is None:
            raise ValueError("Environment not ready for rendering. Call reset() first.")

        if mode == "rgb_array":
            return self._last_observation

        if mode == "human":
            if self.viewer is None:
                # Import inside the method to avoid circular dependencies.
                from gym.envs.classic_control import rendering

                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._last_observation)
            return self.viewer.isopen

        raise ValueError(f"Unknown render mode: {mode}")

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Returns the action space."""
        action_spec = self._env.action_spec()  # type: specs.DiscreteArray
        return gym.spaces.Discrete(action_spec.num_values)

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Returns the observation space."""
        obs_spec = self._env.observation_spec()  # type: specs.Array
        # TODO: May need further adjustments depending on obs_spec type.
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(obs_spec.shape[1],),
            dtype=obs_spec.dtype,
        )

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Returns the reward range."""
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float("inf"), float("inf")

    def __getattr__(self, attr: str) -> Any:
        """Delegates attribute access to the underlying environment."""
        return getattr(self._env, attr)


def space2spec(space: gym.Space, name: Optional[str] = None) -> Any:
    """Converts a Gym space to a dm_env spec or nested structure of specs.

    Converts:
    - Discrete to DiscreteArray;
    - Box, MultiBinary, MultiDiscrete to BoundedArray;
    - Tuple and Dict recursively to tuple and dict of specs.

    Args:
        space: The Gym space to convert.
        name: Optional name to apply to all returned spec(s).

    Returns:
        The corresponding dm_env spec or nested structure of specs.
    """
    if isinstance(space, gym.spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, gym.spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            name=name,
        )

    elif isinstance(space, gym.spaces.MultiBinary):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=0.0,
            maximum=1.0,
            name=name,
        )

    elif isinstance(space, gym.spaces.MultiDiscrete):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=np.zeros(space.shape),
            maximum=space.nvec,
            name=name,
        )

    elif isinstance(space, gym.spaces.Tuple):
        return tuple(space2spec(s, name) for s in space.spaces)

    elif isinstance(space, gym.spaces.Dict):
        return {key: space2spec(value, name) for key, value in space.spaces.items()}

    else:
        raise ValueError(f"Unexpected gym space: {space}")

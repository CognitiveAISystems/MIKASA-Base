from typing import Any, Dict, Optional

import numpy as np
import dm_env
from dm_env import specs
from .base import Environment


class DiscountingChain(Environment):
    """Simple diagnostic discounting challenge.

    Observation consists of two values: (context, time_to_live).

    - `context` is -1 on the first step and then equals the action selected in
    the first step. The agent follows a "chain" for that action.
    - A reward of +1 is given at one of the following time steps: 1, 3, 10, 30, 100.
    - Depending on the seed, one of these chains has a 10% bonus.
    """

    def __init__(self, mapping_seed: Optional[int] = None):
        """Initializes the Discounting Chain environment.

        Args:
            mapping_seed: Optional integer specifying which reward receives a bonus.
        """
        super().__init__()
        self._episode_len = 100
        self._reward_timestep = [1, 3, 10, 30, 100]
        self._n_actions = len(self._reward_timestep)

        if mapping_seed is None:
            mapping_seed = np.random.randint(0, self._n_actions)
        else:
            mapping_seed %= self._n_actions

        self._rewards = np.ones(self._n_actions)
        self._rewards[mapping_seed] += 0.1

        self._timestep = 0
        self._context = -1
        self.bsuite_num_episodes = 10_000  # Number of episodes for benchmarking.

    def _get_observation(self) -> np.ndarray:
        """Generates the current observation.

        Returns:
            A numpy array of shape (1, 2) containing context and normalized time.
        """
        obs = np.zeros(shape=(1, 2), dtype=np.float32)
        obs[0, 0] = self._context
        obs[0, 1] = self._timestep / self._episode_len
        return obs

    def _reset(self) -> dm_env.TimeStep:
        """Resets the environment to its initial state.

        Returns:
            A dm_env.TimeStep representing the initial observation.
        """
        self._timestep = 0
        self._context = -1
        observation = self._get_observation()
        return dm_env.restart(observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        """Takes a step in the environment given an action.

        Args:
            action: The action chosen by the agent.

        Returns:
            A dm_env.TimeStep representing the new state and reward.
        """
        if self._timestep == 0:
            self._context = action

        self._timestep += 1
        reward = (
            self._rewards[self._context]
            if self._timestep == self._reward_timestep[self._context]
            else 0.0
        )

        observation = self._get_observation()
        if self._timestep == self._episode_len:
            return dm_env.termination(reward=reward, observation=observation)
        return dm_env.transition(reward=reward, observation=observation)

    def observation_spec(self) -> specs.Array:
        """Returns the observation specification.

        Returns:
            A dm_env specification object describing the observation space.
        """
        return specs.Array(shape=(1, 2), dtype=np.float32, name="observation")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action specification.

        Returns:
            A dm_env specification object describing the action space.
        """
        return specs.DiscreteArray(self._n_actions, name="action")

    def _save(self, observation: np.ndarray) -> None:
        """Saves the raw observation as an 8-bit image representation.

        Args:
            observation: A numpy array containing the raw observation.
        """
        self._raw_observation = (observation * 255).astype(np.uint8)

    @property
    def optimal_return(self) -> float:
        """Returns the maximum total reward achievable in an episode."""
        return 1.1

    def bsuite_info(self) -> Dict[str, Any]:
        """Returns benchmarking information.

        Returns:
            An empty dictionary (can be extended for logging purposes).
        """
        return {}

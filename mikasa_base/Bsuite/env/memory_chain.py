from typing import Any, Dict, Optional

import numpy as np
import dm_env
from dm_env import specs
from .base import Environment


class MemoryChain(Environment):
    """Simple diagnostic memory challenge.

    Observation consists of (context, time_to_live, query, and num_bits of context).

    - `context` is nonzero only in the first step, where each bit is randomly +1 or -1.
    - Actions have no effect until `time_to_live=0`, at which point the agent
    must recall and repeat the observed bits.
    """

    def __init__(
        self, memory_length: int = 10, num_bits: int = 1, seed: Optional[int] = None
    ):
        """Initializes the MemoryChain environment.

        Args:
            memory_length: The number of steps before recall.
            num_bits: The number of bits in the context.
            seed: Random seed for context generation.
        """
        super().__init__()
        self._memory_length = memory_length
        self._num_bits = num_bits
        self._rng = np.random.RandomState(seed)

        # Contextual information per episode
        self._timestep = 0
        self._context = self._rng.binomial(1, 0.5, num_bits)
        self._query = self._rng.randint(num_bits)

        # Logging info
        self._total_perfect = 0
        self._total_regret = 0
        self._episode_mistakes = 0

        # Experiment episode count
        self.bsuite_num_episodes = 10_000

    def _get_observation(self) -> np.ndarray:
        """Generates the current observation.

        Returns:
            A numpy array containing time, query, and context bits.
        """
        obs = np.zeros(shape=(1, self._num_bits + 2), dtype=np.float32)
        obs[0, 0] = 1 - self._timestep / self._memory_length  # Time decay
        if self._timestep == self._memory_length - 1:
            obs[0, 1] = self._query  # Query shown on the last step
        if self._timestep == 0:
            obs[0, 2:] = 2 * self._context - 1  # Context shown on the first step
        return obs

    def _step(self, action: int) -> dm_env.TimeStep:
        """Takes a step in the environment given an action.

        Args:
            action: The action chosen by the agent.

        Returns:
            A dm_env.TimeStep representing the new state and reward.
        """
        observation = self._get_observation()
        self._timestep += 1

        if self._timestep - 1 < self._memory_length:
            return dm_env.transition(reward=0.0, observation=observation)
        if self._timestep - 1 > self._memory_length:
            raise RuntimeError("Invalid state encountered.")

        reward = 1.0 if action == self._context[self._query] else -1.0
        if reward == 1.0:
            self._total_perfect += 1
        else:
            self._total_regret += 2.0

        return dm_env.termination(reward=reward, observation=observation)

    def _reset(self, seed) -> dm_env.TimeStep:
        """Resets the environment to its initial state.

        Returns:
            A dm_env.TimeStep representing the initial observation.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._timestep = 0
        self._episode_mistakes = 0
        self._context = self._rng.binomial(1, 0.5, self._num_bits)
        self._query = self._rng.randint(self._num_bits)
        observation = self._get_observation()
        return dm_env.restart(observation)

    def observation_spec(self) -> specs.Array:
        """Returns the observation specification.

        Returns:
            A dm_env specification describing the observation space.
        """
        return specs.Array(
            shape=(1, self._num_bits + 2), dtype=np.float32, name="observation"
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action specification.

        Returns:
            A dm_env specification describing the action space.
        """
        return specs.DiscreteArray(2, name="action")

    def _save(self, observation: np.ndarray) -> None:
        """Saves the raw observation as an 8-bit image representation.

        Args:
            observation: A numpy array containing the raw observation.
        """
        self._raw_observation = (observation * 255).astype(np.uint8)

    def bsuite_info(self) -> Dict[str, Any]:
        """Returns benchmarking information.

        Returns:
            A dictionary containing total perfect responses and total regret.
        """
        return {
            "total_perfect": self._total_perfect,
            "total_regret": self._total_regret,
        }

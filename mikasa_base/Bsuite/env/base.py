""" "Base class for bsuite environments.

This inherits from the dm_env base class, with two major differences:

- Includes bsuite-specific metadata:
- `bsuite_info` returns metadata for logging, e.g. for computing regret/score.
- `bsuite_num_episodes` specifies how long the experiment should run for.
- Implements the auto-reset behavior specified by the environment API.
That is, stepping an environment after a LAST timestep should return the
first timestep of a new episode.
"""

import abc
from typing import Any, Dict

import dm_env
import numpy as np


class Environment(dm_env.Environment, abc.ABC):
    """Base clas for bsuite environments.

    A bsuite environment is a dm_env environment with extra metadata:
    - bsuite_info method.
    - bsuite_num_episodes attribute.

    A bsuite environment also has auto-reset behavior.
    This class implements the required `step()` and `reset()` methods.

    It instead requires users to implement `_step()` and `_reset()`. This class
    handles the reset behaviour automatically when it detects a LAST timestep.
    """

    # Number of episodes that this environment should be run for.
    bsuite_num_episodes: int

    def __init__(self):
        self._reset_next_step = True

    def reset(self, seed=None) -> dm_env.TimeStep:
        """Resets the environment, calling the underlying _reset() method."""
        self._reset_next_step = False
        return self._reset(seed)

    def step(self, action: int) -> dm_env.TimeStep:
        """Steps the environment and implements the auto-reset behavior."""
        if self._reset_next_step:
            return self.reset()
        timestep = self._step(action)
        self._reset_next_step = timestep.last()
        return timestep

    @abc.abstractmethod
    def _reset(self) -> dm_env.TimeStep:
        """Returns a `timestep` namedtuple as per the regular `reset()` method."""

    @abc.abstractmethod
    def _step(self, action: int) -> dm_env.TimeStep:
        """Returns a `timestep` namedtuple as per the regular `step()` method."""

    @abc.abstractmethod
    def bsuite_info(self) -> Dict[str, Any]:
        """Returns metadata specific to this environment for logging/scoring."""

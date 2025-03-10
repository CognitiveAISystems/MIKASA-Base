"""A game very similar to Simon, but backwards.

The agent receives a sequence of cards, and must output the cards it saw
in reverse order. E.g., seeing [1, 2, 3] means I should output them in the order
[3, 2, 1].
"""

import enum
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from .core.deck import Deck
from .core.env import POPGymEnv


class Mode(enum.IntEnum):
    PLAY = 0
    WATCH = 1


class Autoencode(POPGymEnv):
    """A game very similar to Simon, but backwards.

    The agent receives a sequence of cards, and must output the cards it saw
    in reverse order. E.g., seeing [1, 2, 3] means I should output them in the order
    [3, 2, 1].

    Examples:
        >>> env = Autoencode()
        >>> env.reset()

    Args:
        num_decks: The maximum number of decks the agent must memorize

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.deck = Deck(num_decks)
        self.deck.add_players("system")
        self.max_episode_length = self.deck.num_cards * 2 - 1
        self.action_space = self.deck.get_obs_space(["suits"])
        # self.observation_space = gym.spaces.Tuple(
        #     (
        #         gym.spaces.Discrete(2),
        #         self.action_space,
        #     )
        # )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.int8
        )
        self.state_space = gym.spaces.Tuple(
            (
                gym.spaces.MultiDiscrete([4] * self.deck.num_cards),
                gym.spaces.Discrete(2),
                gym.spaces.Box(0, 1, (1,)),
            )
        )
        self.mode = Mode.WATCH

    def make_obs(self, card_idx) -> Tuple[int, int]:
        card_suit = self.deck.suits_idx[card_idx].reshape(-1)
        return np.array([int(self.mode.value), card_suit.item()])

    def get_state(self) -> Tuple[np.ndarray, int, np.ndarray]:
        cards = self.deck.suits_idx[self.deck.idx].copy()
        mode = int(self.mode.value)
        pos = np.array(
            [len(self.deck["system"]) / self.deck.num_cards], dtype=np.float32
        )
        return cards, mode, pos

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        terminated = truncated = False
        reward = 0
        # TODO: This is causing flaky tests, make sure we are not
        # off-by-one in step
        reward_scale = 1 / self.deck.num_cards

        if self.mode == Mode.WATCH:
            self.deck.deal("system", 1)
            if len(self.deck) == 0:
                self.mode = Mode.PLAY
            obs = self.make_obs(self.deck["system"][-1])
        else:
            # Recited all cards
            terminated = len(self.deck["system"]) == 1
            correct_card = self.deck.suits_idx[self.deck["system"].pop(-1)]
            if action == correct_card:
                reward = reward_scale
            else:
                reward = -reward_scale

            obs = self.make_obs(np.array([0]))

        info: dict = {}

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)
        self.deck.deal("system", 1)

        self.mode = Mode.WATCH
        obs = self.make_obs(self.deck["system"][-1])
        return obs, {}


class AutoencodeEasy(Autoencode):
    pass


class AutoencodeMedium(Autoencode):
    def __init__(self):
        super().__init__(num_decks=2)


class AutoencodeHard(Autoencode):
    def __init__(self):
        super().__init__(num_decks=3)

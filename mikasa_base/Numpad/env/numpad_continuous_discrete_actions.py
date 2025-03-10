from unittest import skip
from .numpad_continuous import Numpad2DContinuous as NumpadContinuous
from .numpad_base import Numpad
import numpy as np
from gymnasium import spaces

from .numpad_continuous import Config as ContinuousConfig


class Config(ContinuousConfig):

    def __init__(self):
        super().__init__()
        [setattr(self, k, v) for k, v in vars(Config).items() if not k.startswith("_")]


class NumpadContinuousDiscreteActions(Numpad):
    UP, RIGHT, DOWN, LEFT, UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )
    NUM_ACTIONS = 8

    def __init__(self, config: Config = Config()):
        super().__init__(config)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.numpad_continuous = NumpadContinuous(config)
        self.observation_space = self.numpad_continuous.observation_space

    def step(self, action):
        if action == self.UP:
            return self.numpad_continuous.step(np.array((-1.0, 0.0)))
        if action == self.DOWN:
            return self.numpad_continuous.step(np.array((1.0, 0.0)))
        if action == self.RIGHT:
            return self.numpad_continuous.step(np.array((0.0, 1.0)))
        if action == self.LEFT:
            return self.numpad_continuous.step(np.array((0.0, -1.0)))
        if action == self.UP_LEFT:
            return self.numpad_continuous.step(np.array((-1.0, -1.0)))
        if action == self.UP_RIGHT:
            return self.numpad_continuous.step(np.array((-1.0, 1.0)))
        if action == self.DOWN_LEFT:
            return self.numpad_continuous.step(np.array((1.0, -1.0)))
        if action == self.DOWN_RIGHT:
            return self.numpad_continuous.step(np.array((1.0, 1.0)))
        raise ValueError(f"Action {action} is not in action space!")

    def reset(self):
        return self.numpad_continuous.reset()

    def close(self):
        return self.numpad_continuous.close()

    def render(
        self,
        mode="file",
        img_filepath="numpad_continuous_discrete_actions",
        frames_freq=5,
        remove_frame_files=True,
    ):
        return self.numpad_continuous.render(
            mode=mode,
            img_filepath=img_filepath,
            frames_freq=frames_freq,
            remove_frame_files=remove_frame_files,
        )

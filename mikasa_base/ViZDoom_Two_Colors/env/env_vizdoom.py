from math import isclose
import itertools
import vizdoom
from vizdoom import (
    DoomGame,
    ScreenResolution,
    GameVariable,
    Button,
    AutomapMode,
    Mode,
    doom_fixed_to_double,
)
import numpy as np
from cv2 import resize
import cv2
import math
import argparse
from itertools import count
import time
import gymnasium as gym
from gymnasium import spaces
from .common_wrappers import ImgObsWrapper, FrameStack, PrevActionAndReward
from enum import IntEnum

import os


class DoomEnvironment(gym.Env):
    """
    A wrapper class for the Doom Maze Environment
    """

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        back = 3
        left_forward = 4
        left_backward = 5
        right_forward = 6
        right_backward = 7

    class PlayerInfo:
        """
        Small class to hold player position info etc

        """

        def __init__(self, x, y, theta):
            self.x = x
            self.y = y
            self.theta = theta  # in radians
            self.starting_theta = theta

            self.dx, self.dy, self.dtheta = 0.0, 0.0, 0.0
            self.vx, self.vy, self.dv = 0.0, 0.0, 0.0
            self.origin_x = x
            self.origin_y = y

        def update(self, x, y, theta):
            # recording of agents position and rotation during a rollout
            # We do some calculations in the agents reference frame which are not relavant
            # for the moment but may be useful for future work
            self.dtheta = theta - self.theta
            self.theta = theta

            # the calculations below will fail if the agent has not moved
            if x == self.x and y == self.y:
                self.dx = 0
                self.dy = 0
                return

            # dx and dy are all in the agents current frame of reference
            world_dx = self.x - x  # swapped due to mismatch in world coord frame
            world_dy = y - self.y

            # the hypotenus of the triangle between the agents previous and current position

            h = math.sqrt(world_dx**2 + world_dy**2)
            theta_tilda = math.atan2(world_dy, world_dx)
            theta_prime = math.pi - theta_tilda - theta
            # theta_prime = theta - theta_tilda this should be correct but the coordinate system in Doom in inverted

            self.dx = h * math.sin(theta_prime)
            self.dy = h * math.cos(theta_prime)
            # changes in x and y are all relative
            self.x = x
            self.y = y
            self.theta = theta

    SCREEN_WIDTH = 112
    SCREEN_HEIGHT = 64
    SCREEN_RESOLUTION = ScreenResolution.RES_320X180
    SCREEN_SIZE = ()

    def __init__(
        self,
        idx=0,
        scenario="scenarios/two_colors_hard.cfg",
        get_extra_obs=False,
        use_shaping=False,  # coeficient of shaping rewards
        frame_skip=4,
        show_window=False,
        screen_size=None,
        no_backward_movement=False,
        resolution=SCREEN_RESOLUTION,
        use_info=False,
        reward_scaling=1.0,
        seed=None,  # all future episodes will be fixed if seed is given
    ):

        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        full_path = os.path.join(current_dir, scenario)
        self.scenario = os.path.normpath(full_path)

        self.idx = idx
        self.use_shaping = use_shaping
        self.reward_scaling = reward_scaling
        self.show_window = show_window
        self.get_extra_obs = get_extra_obs
        self.screen_resolution = resolution
        self.use_info = use_info

        self.game = self._create_game(self.scenario, show_window, get_extra_obs)
        self.screen_width = screen_size[0] if screen_size else self.SCREEN_WIDTH
        self.screen_height = screen_size[1] if screen_size else self.SCREEN_HEIGHT

        self.frame_skip = frame_skip

        self.action_map = self._gen_actions(self.game, no_backward_movement)
        self.num_actions = len(self.action_map)
        self.actions = DoomEnvironment.Actions
        self.game_state = None

        self.seed(seed)
        self.empty_image = np.zeros(
            (3, self.screen_height, self.screen_width), dtype=np.uint8
        )

        self.player_info = self.PlayerInfo(
            self.game.get_game_variable(GameVariable.POSITION_X),
            self.game.get_game_variable(GameVariable.POSITION_Y),
            math.radians(self.game.get_game_variable(GameVariable.ANGLE)),
        )

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            0.0, 255.0, (3, self.screen_height, self.screen_width), dtype=np.uint8
        )
        self.reward_range = (-2.0, 2.0)

    def render(self, mode="human", **kwargs):
        raise NotImplementedError()

    @property
    def unwrapped(self):
        return self

    def _create_game(self, scenario, show_window, get_extra_info=False):
        game = DoomGame()

        game.load_config(scenario)

        game.set_screen_resolution(self.screen_resolution)

        game.set_sound_enabled(False)
        # game.add_game_args("+vid_forcesurface 1")
        game.set_window_visible(show_window)

        if show_window:
            game.set_mode(Mode.SPECTATOR)
            game.add_game_args("+freelook 1")

        # Player variables for prediction of position etc
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.add_available_game_variable(GameVariable.POSITION_Z)
        game.add_available_game_variable(GameVariable.VELOCITY_X)
        game.add_available_game_variable(GameVariable.VELOCITY_Y)
        game.add_available_game_variable(GameVariable.VELOCITY_Z)
        game.add_available_game_variable(GameVariable.ANGLE)
        game.add_available_game_variable(GameVariable.PITCH)
        game.add_available_game_variable(GameVariable.ROLL)
        game.add_available_game_variable(GameVariable.USER2)

        if get_extra_info:
            game.set_labels_buffer_enabled(True)
            game.set_automap_buffer_enabled(True)
            game.set_automap_mode(AutomapMode.OBJECTS)
            game.set_automap_rotate(True)
            game.set_automap_render_textures(False)
            game.set_depth_buffer_enabled(True)

        game.init()

        if GameVariable.HEALTH in game.get_available_game_variables():
            self.previous_health = game.get_game_variable(GameVariable.HEALTH)

        if self.use_shaping:
            self.shaping_reward = doom_fixed_to_double(
                game.get_game_variable(GameVariable.USER1)
            )

        # if params.disable_head_bob: #up and down head movement during walking
        #    game.send_game_command('movebob 0.0')
        return game

    def _gen_actions(self, game, no_backward_movement):
        buttons = game.get_available_buttons()
        if buttons == [
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD,
        ]:
            if no_backward_movement:
                feasible_actions = [
                    [True, False, False, False],  # Left
                    [False, True, False, False],  # Right
                    [False, False, True, False],  # Forward
                    [True, False, True, False],  # Left + Forward
                    [False, True, True, False],
                ]  # Right + forward
            else:
                feasible_actions = [
                    [True, False, False, False],  # Left
                    [False, True, False, False],  # Right
                    [False, False, True, False],  # Forward
                    [False, False, False, True],  # Backward
                    [True, False, True, False],  # Left + Forward
                    [True, False, False, True],  # Left + Backward
                    [False, True, True, False],  # Right + forward
                    [False, True, False, True],
                ]  # Right + backward

        else:
            feasible_actions = [
                list(l) for l in itertools.product([True, False], repeat=len(buttons))
            ]

        action_map = {i: act for i, act in enumerate(feasible_actions)}
        return action_map

    def seed(self, seed=None):
        if seed is not None:
            self.game.set_seed(seed)

    def reset(self, seed=None, **kwargs):
        info = {}
        self.game.new_episode()

        self.player_info = self.PlayerInfo(
            self.game.get_game_variable(GameVariable.POSITION_X),
            self.game.get_game_variable(GameVariable.POSITION_Y),
            math.radians(self.game.get_game_variable(GameVariable.ANGLE)),
        )

        if GameVariable.HEALTH in self.game.get_available_game_variables():
            self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)

        if self.use_shaping:
            self.shaping_reward = doom_fixed_to_double(
                self.game.get_game_variable(GameVariable.USER1)
            )

        return self.get_observation(), info

    def is_red_episode(self):
        return self.game.get_game_variable(GameVariable.USER2)

    def is_episode_finished(self):
        return self.game.is_episode_finished()

    def get_observation(self):
        obs = {}
        if self.is_episode_finished():
            self.game_state = None
            obs["image"] = self.empty_image
            obs["health"] = 0.0
            obs["shaping_reward"] = 0.0
        else:
            self.game_state = self.game.get_state()
            obs["image"] = self.get_image()
            obs["health"] = self.game.get_game_variable(GameVariable.HEALTH)
            obs["shaping_reward"] = doom_fixed_to_double(
                self.game.get_game_variable(GameVariable.USER1)
            )
            # obs['is_red'] = self.is_red_episode()
            # obs['frame_id'] = self.game_state.number

        # obs['prev_r'] = np.zeros(1, dtype=np.float32)
        # obs['prev_a'] = np.zeros(1, dtype=np.float32)
        return obs

    def get_image(self):
        if not self.game_state:
            return None

        observation = self.game_state.screen_buffer
        observation = resize(
            observation.transpose(1, 2, 0),
            (self.screen_width, self.screen_height),
            cv2.INTER_AREA,
        ).transpose(2, 0, 1)

        return observation

    def make_action(self, action, human_play=False):
        """
        perform an action, includes an option to skip frames but repeat
        the same action.

        """
        if human_play:
            self.game.advance_action(self.frame_skip)
            reward = self.game.get_last_reward()
        else:
            action = int(action)
            reward = self.game.make_action(self.action_map[action], self.frame_skip)
        # We shape rewards in health gathering to encourage collection of health packs
        if not self.use_shaping:
            reward += self._check_health()
        else:
            # alternatively ViZDoom offers a shaping reward in some scenarios
            current_shaping_reward = doom_fixed_to_double(
                self.game.get_game_variable(GameVariable.USER1)
            )
            diff = current_shaping_reward - self.shaping_reward
            reward += diff

            self.shaping_reward += diff

        return reward

    def get_info(self):
        info = dict(
            last_action=self.game.get_last_action(),
            is_red=self.is_red_episode(),
        )
        return info

    def step(self, action, human_play=False):
        reward = self.make_action(action, human_play)
        done = self.is_episode_finished()

        # if done:
        # print('vizdoom episode is finished!')

        obs = self.get_observation()
        if not done:
            new_x = self.game.get_game_variable(GameVariable.POSITION_X)
            new_y = self.game.get_game_variable(GameVariable.POSITION_Y)
            new_theta = self.game.get_game_variable(GameVariable.ANGLE)
            self.player_info.update(new_x, new_y, math.radians(new_theta))

        info = self.get_info() if self.use_info else {}
        return obs, reward * self.reward_scaling, done, False, info

    def _check_health(self):
        """
        Modification to reward function in order to reward the act of finding a health pack

        """
        health_reward = 0.0

        if GameVariable.HEALTH not in self.game.get_available_game_variables():
            self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
            return health_reward

        if self.game.get_game_variable(GameVariable.HEALTH) > self.previous_health:
            # print('found healthkit')
            health_reward = 1.0

        self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
        return health_reward

    def get_total_reward(self):
        return self.game.get_total_reward()

    def get_player_position(self):
        return self.player_info.x, self.player_info.y, self.player_info.theta

    def get_player_deltas(self):
        return self.player_info.dx, self.player_info.dy, self.player_info.dtheta

    def get_player_origins(self):
        return self.player_info.origin_x, self.player_info.origin_y

    def get_player_pos_delta_origin(self):
        return (
            self.player_info.x,
            self.player_info.y,
            self.player_info.theta,
            self.player_info.dx,
            self.player_info.dy,
            self.player_info.dtheta,
            self.player_info.origin_x,
            self.player_info.origin_y,
        )

    def close(self):
        self.game.close()

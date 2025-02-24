from gymnasium import register

__version__ = "0.0.1"
import os

#################################### Ballet ####################################


# ['2_delay16', '4_delay16', '8_delay16',
#  '2_delay48', '4_delay48', '8_delay48']


register(
    id=f"Ballet-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
)


register(
    id=f"BalletEasy-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
    kwargs={"num_dancers": 2, "dance_delay": 16, "easy_mode": False},
)

register(
    id=f"BalletMedium-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
    kwargs={"num_dancers": 4, "dance_delay": 16, "easy_mode": False},
)

register(
    id=f"BalletHard-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
    kwargs={"num_dancers": 8, "dance_delay": 48, "easy_mode": False},
)

# light moidfication - use red color for all dancers

register(
    id=f"BalletLightEasy-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
    kwargs={"num_dancers": 2, "dance_delay": 16, "easy_mode": True},
)

register(
    id=f"BalletLightMedium-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
    kwargs={"num_dancers": 4, "dance_delay": 16, "easy_mode": True},
)

register(
    id=f"BalletLightHard-v0",
    entry_point=f"mikasa_base.Ballet.env.ballet_env:BalletEnvironment",
    kwargs={"num_dancers": 8, "dance_delay": 48, "easy_mode": True},
)


#################################### Bsuite ####################################


register(
    id=f"MemoryLength-v0",
    entry_point=f"mikasa_base.Bsuite.env.bsuite_env:BsuiteGymWrapper",
    kwargs={
        "env_id": "MemoryLength",
    },
)

register(
    id=f"MemoryLengthEasy-v0",
    entry_point=f"mikasa_base.Bsuite.env.bsuite_env:BsuiteGymWrapper",
    kwargs={"env_id": "MemoryLength", "memory_length": 10, "num_bits": 1},
)

register(
    id=f"MemoryLengthMedium-v0",
    entry_point=f"mikasa_base.Bsuite.env.bsuite_env:BsuiteGymWrapper",
    kwargs={"env_id": "MemoryLength", "memory_length": 30, "num_bits": 1},
)

register(
    id=f"MemoryLengthHard-v0",
    entry_point=f"mikasa_base.Bsuite.env.bsuite_env:BsuiteGymWrapper",
    kwargs={"env_id": "MemoryLength", "memory_length": 100, "num_bits": 3},
)

# register(
#     id=f"DiscountingChain-v0",
#     entry_point=f"mikasa_base.Bsuite.env.bsuite_env:BsuiteGymWrapper",
#     kwargs={"env_id": "DiscountingChain",},
# )


#################################### MemoryCards ####################################

register(
    id=f"MemoryCards-v0",
    entry_point=f"mikasa_base.MemoryCards.env.memory_cards_env:Memory",
)

register(
    id=f"MemoryCardsEasy-v0",
    entry_point=f"mikasa_base.MemoryCards.env.memory_cards_env:Memory",
    kwargs={"num_pairs": 5},
)

register(
    id=f"MemoryCardsMedium-v0",
    entry_point=f"mikasa_base.MemoryCards.env.memory_cards_env:Memory",
    kwargs={"num_pairs": 10},
)

register(
    id=f"MemoryCardsHard-v0",
    entry_point=f"mikasa_base.MemoryCards.env.memory_cards_env:Memory",
    kwargs={"num_pairs": 30},
)


#################################### MemoryGym ####################################

# MysteryPath

register(
    id="MysteryPath-v0",
    entry_point=f"mikasa_base.MemoryGym.env.mystery_path:MysteryPathEnv",
)

register(
    id="MysteryPath-Grid-v0",
    entry_point=f"mikasa_base.MemoryGym.env.mystery_path_grid:GridMysteryPathEnv",
)

register(
    id="Endless-MysteryPath-v0",
    entry_point=f"mikasa_base.MemoryGym.env.endless_mystery_path:EndlessMysteryPathEnv",
)


# MortarMayhem

register(
    id="MortarMayhem-v0",
    entry_point="mikasa_base.MemoryGym.env.mortar_mayhem:MortarMayhemEnv",
)

register(
    id="Endless-MortarMayhem-v0",
    entry_point="mikasa_base.MemoryGym.env.endless_mortar_mayhem:EndlessMortarMayhemEnv",
)

register(
    id="MortarMayhem-Grid-v0",
    entry_point="mikasa_base.MemoryGym.env.mortar_mayhem_grid:GridMortarMayhemEnv",
)

register(
    id="MortarMayhemB-v0",
    entry_point="mikasa_base.MemoryGym.env.mortar_mayhem_b:MortarMayhemTaskBEnv",
)

register(
    id="MortarMayhemB-Grid-v0",
    entry_point="mikasa_base.MemoryGym.env.mortar_mayhem_b_grid:GridMortarMayhemTaskBEnv",
)


# SearingSpotlights

register(
    id="SearingSpotlights-v0",
    entry_point="mikasa_base.MemoryGym.env.searing_spotlights:SearingSpotlightsEnv",
)

register(
    id="Endless-SearingSpotlights-v0",
    entry_point="mikasa_base.MemoryGym.env.endless_searing_spotlights:EndlessSearingSpotlightsEnv",
)


#################################### MiniGrid-Memory ####################################

register(
    id="MiniGrid-Memory-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
)

register(
    id="MiniGrid-MemoryS7-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
    kwargs={"size": 7},
)

register(
    id="MiniGrid-MemoryS13-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
    kwargs={"size": 13},
)

register(
    id="MiniGrid-MemoryS19-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
    kwargs={"size": 19},
)

register(
    id="MiniGrid-MemoryS7Random-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
    kwargs={"size": 7, "random_length": True},
)

register(
    id="MiniGrid-MemoryS13Random-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
    kwargs={"size": 13, "random_length": True},
)

register(
    id="MiniGrid-MemoryS17Random-v0",
    entry_point="mikasa_base.MinigridMemory.env.memory:MemoryEnv",
    kwargs={"size": 19, "random_length": True},
)

#################################### Numpad ####################################

register(
    id="Numpad-Discrete-v0",
    entry_point=f"mikasa_base.Numpad.env.numpad_discrete:Numpad2DDiscrete",
)

register(
    id="Numpad-Continuous-v0",
    entry_point=f"mikasa_base.Numpad.env.numpad_continuous:Numpad2DContinuous",
)

register(
    id="Numpad-ContinuousDiscreteActions-v0",
    entry_point=f"mikasa_base.Numpad.env.numpad_continuous_discrete_actions:NumpadContinuousDiscreteActions",
)


#################################### Numpad ####################################


register(
    id="Passive-VisualMatch-v0",
    entry_point=f"mikasa_base.PassiveVisualMatch2D.env.passive_visual_match_env:VisualMatch",
)

# register(
#     id="KeyToDoor-v0",
#     entry_point=f"mikasa_base.PassiveVisualMatch2D.env.passive_visual_match_env:KeyToDoor"
# )


#################################### Numpad ####################################


register(
    id="ViZDoomTwoColorsBase-v0",
    entry_point=f"mikasa_base.ViZDoom_Two_Colors.env.env_vizdoom:DoomEnvironment",
    kwargs={"scenario": "scenarios/two_colors.cfg"},
)

register(
    id="ViZDoomTwoColorsDelay1000-v0",
    entry_point=f"mikasa_base.ViZDoom_Two_Colors.env.env_vizdoom:DoomEnvironment",
    kwargs={"scenario": "scenarios/two_colors_delay_1000.cfg"},
)

register(
    id="ViZDoomTwoColorsHard-v0",
    entry_point=f"mikasa_base.ViZDoom_Two_Colors.env.env_vizdoom:DoomEnvironment",
    kwargs={"scenario": "scenarios/two_colors_hard.cfg"},
)


# T-Maze


register(
    id=f"T-Maze-v0",
    entry_point="mikasa_base.Passive_T_Maze.env.env_passive_t_maze:TMazeClassicPassive",
)

######### T-Maze-dense #########

register(
    id=f"T-Maze-Dense-v0",
    entry_point="mikasa_base.Passive_T_Maze.env.env_passive_t_maze:TMazeClassicPassive",
    kwargs={"mode": "dense"},
)

######### T-Maze-Noise #########

# register(
#     id=f"T-Maze-Noise-v0",
#     entry_point="mikasa_base.Passive_T_Maze.env.env_passive_t_maze_flag:TMazeClassicPassive",
# )

#################################### POPGym ####################################

register(
    id=f"Autoencode-v0",
    entry_point=f"mikasa_base.POPGym.env.autoencode:Autoencode",
)

register(
    id=f"RepeatPrevious-v0",
    entry_point=f"mikasa_base.POPGym.env.repeat_previous:RepeatPrevious",
)

register(
    id=f"Concentration-v0",
    entry_point=f"mikasa_base.POPGym.env.concentration:Concentration",
)

register(
    id=f"CountRecall-v0",
    entry_point=f"mikasa_base.POPGym.env.count_recall:CountRecall",
)

register(
    id=f"Battleship-v0",
    entry_point=f"mikasa_base.POPGym.env.battleship:Battleship",
)

register(
    id=f"HigherLower-v0",
    entry_point=f"mikasa_base.POPGym.env.higher_lower:HigherLower",
)

register(
    id=f"LabyrinthEscape-v0",
    entry_point=f"mikasa_base.POPGym.env.labyrinth_escape:LabyrinthEscape",
)

register(
    id=f"LabyrinthExplore-v0",
    entry_point=f"mikasa_base.POPGym.env.labyrinth_explore:LabyrinthExplore",
)

register(
    id=f"MineSweeper-v0",
    entry_point=f"mikasa_base.POPGym.env.minesweeper:MineSweeper",
)

register(
    id=f"MultiarmedBandit-v0",
    entry_point=f"mikasa_base.POPGym.env.multiarmed_bandit:MultiarmedBandit",
)

register(
    id=f"RepeatFirst-v0",
    entry_point=f"mikasa_base.POPGym.env.repeat_first:RepeatFirst",
)

register(
    id=f"PositionOnlyCartPole-v0",
    entry_point=f"mikasa_base.POPGym.env.position_only_cartpole:PositionOnlyCartPole",
)

register(
    id=f"PositionOnlyPendulum-v0",
    entry_point=f"mikasa_base.POPGym.env.position_only_pendulum:PositionOnlyPendulum",
)

register(
    id=f"VelocityOnlyCartPole-v0",
    entry_point=f"mikasa_base.POPGym.env.velocity_only_cartpole:VelocityOnlyCartPole",
)

register(
    id=f"NoisyPositionOnlyCartPole-v0",
    entry_point=f"mikasa_base.POPGym.env.noisy_position_only_cartpole:NoisyPositionOnlyCartPole",
)

register(
    id=f"NoisyPositionOnlyPendulum-v0",
    entry_point=f"mikasa_base.POPGym.env.noisy_position_only_pendulum:NoisyPositionOnlyPendulum",
)


######### Autoencode, RepeatPrevious, Concentration #########

for level in ["Easy", "Medium", "Hard"]:

    register(
        id=f"Autoencode{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.autoencode:Autoencode{level}",
    )

    register(
        id=f"RepeatPrevious{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.repeat_previous:RepeatPrevious{level}",
    )

    register(
        id=f"Concentration{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.concentration:Concentration{level}",
    )

    register(
        id=f"CountRecall{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.count_recall:CountRecall{level}",
    )

    register(
        id=f"Battleship{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.battleship:Battleship{level}",
    )

    register(
        id=f"HigherLower{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.higher_lower:HigherLower{level}",
    )

    register(
        id=f"LabyrinthEscape{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.labyrinth_escape:LabyrinthEscape{level}",
    )

    register(
        id=f"LabyrinthExplore{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.labyrinth_explore:LabyrinthExplore{level}",
    )

    register(
        id=f"MineSweeper{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.minesweeper:MineSweeper{level}",
    )

    register(
        id=f"MultiarmedBandit{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.multiarmed_bandit:MultiarmedBandit{level}",
    )

    register(
        id=f"RepeatFirst{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.repeat_first:RepeatFirst{level}",
    )

    register(
        id=f"PositionOnlyCartPole{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.position_only_cartpole:PositionOnlyCartPole{level}",
    )

    register(
        id=f"PositionOnlyPendulum{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.position_only_pendulum:PositionOnlyPendulum{level}",
    )

    register(
        id=f"VelocityOnlyCartPole{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.velocity_only_cartpole:VelocityOnlyCartPole{level}",
    )

    register(
        id=f"NoisyPositionOnlyCartPole{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.noisy_position_only_cartpole:NoisyPositionOnlyCartPole{level}",
    )
    register(
        id=f"NoisyPositionOnlyPendulum{level}-v0",
        entry_point=f"mikasa_base.POPGym.env.noisy_position_only_pendulum:NoisyPositionOnlyPendulum{level}",
    )


#################################### MemoryMaze ####################################


# NOTE: Env MUJOCO_GL=egl is necessary for headless hardware rendering on GPU,
# but breaks when running on a CPU machine. Alternatively set MUJOCO_GL=osmesa.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from .MemoryMaze.env import tasks

try:
    # Register gym environments, if gym is available

    from typing import Callable
    from functools import partial as f

    import dm_env
    import gymnasium as gym

    # from gym.envs.registration import register

    from .MemoryMaze.env.gym_wrappers import GymWrapper

    def _make_gym_env(dm_task: Callable[[], dm_env.Environment], **kwargs):
        dmenv = dm_task(**kwargs)
        return GymWrapper(dmenv)

    sizes = {
        "9x9": tasks.memory_maze_9x9,
        "11x11": tasks.memory_maze_11x11,
        "13x13": tasks.memory_maze_13x13,
        "15x15": tasks.memory_maze_15x15,
    }

    for key, dm_task in sizes.items():
        # Image-only obs space
        register(
            id=f"MemoryMaze-{key}-v0",
            entry_point=f(_make_gym_env, dm_task, image_only_obs=True),
        )  # Standard
        register(
            id=f"MemoryMaze-{key}-Vis-v0",
            entry_point=f(
                _make_gym_env, dm_task, image_only_obs=True, good_visibility=True
            ),
        )  # Easily visible targets
        register(
            id=f"MemoryMaze-{key}-HD-v0",
            entry_point=f(
                _make_gym_env, dm_task, image_only_obs=True, camera_resolution=256
            ),
        )  # High-res camera
        register(
            id=f"MemoryMaze-{key}-Top-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                image_only_obs=True,
                camera_resolution=256,
                top_camera=True,
            ),
        )  # Top-down camera

        # Extra global observables (dict obs space)
        register(
            id=f"MemoryMaze-{key}-ExtraObs-v0",
            entry_point=f(_make_gym_env, dm_task, global_observables=True),
        )
        register(
            id=f"MemoryMaze-{key}-ExtraObs-Vis-v0",
            entry_point=f(
                _make_gym_env, dm_task, global_observables=True, good_visibility=True
            ),
        )
        register(
            id=f"MemoryMaze-{key}-ExtraObs-Top-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                global_observables=True,
                camera_resolution=256,
                top_camera=True,
            ),
        )

        # Oracle observables with shortest path shown
        register(
            id=f"MemoryMaze-{key}-Oracle-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                image_only_obs=True,
                global_observables=True,
                show_path=True,
            ),
        )
        register(
            id=f"MemoryMaze-{key}-Oracle-Top-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                image_only_obs=True,
                global_observables=True,
                show_path=True,
                camera_resolution=256,
                top_camera=True,
            ),
        )
        register(
            id=f"MemoryMaze-{key}-Oracle-ExtraObs-v0",
            entry_point=f(
                _make_gym_env, dm_task, global_observables=True, show_path=True
            ),
        )

        # High control frequency
        register(
            id=f"MemoryMaze-{key}-HiFreq-v0",
            entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40),
        )
        register(
            id=f"MemoryMaze-{key}-HiFreq-Vis-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                image_only_obs=True,
                control_freq=40,
                good_visibility=True,
            ),
        )
        register(
            id=f"MemoryMaze-{key}-HiFreq-HD-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                image_only_obs=True,
                control_freq=40,
                camera_resolution=256,
            ),
        )

        # Six colors even for smaller mazes
        register(
            id=f"MemoryMaze-{key}-6CL-v0",
            entry_point=f(
                _make_gym_env, dm_task, randomize_colors=True, image_only_obs=True
            ),
        )
        register(
            id=f"MemoryMaze-{key}-6CL-Top-v0",
            entry_point=f(
                _make_gym_env,
                dm_task,
                randomize_colors=True,
                image_only_obs=True,
                camera_resolution=256,
                top_camera=True,
            ),
        )
        register(
            id=f"MemoryMaze-{key}-6CL-ExtraObs-v0",
            entry_point=f(
                _make_gym_env, dm_task, randomize_colors=True, global_observables=True
            ),
        )


except ImportError:
    print("memory_maze: gym environments not registered.")
    raise

<h1 align="center">MIKASA-Base</h1>

<h3 align="center">Unified Benchmark for Memory-Intensive Tasks</h3>

<div align="center">
    <a href="https://arxiv.org/abs/2502.10550">
        <img src="https://img.shields.io/badge/arXiv-2502.10550-b31b1b.svg"/>
    </a>
    <a href="https://sites.google.com/view/memorybenchrobots/">
        <img src="https://img.shields.io/badge/Website-Project_Page-blue.svg"/>
    </a>
</div>

---
## Overview

MIKASA-Base is a unified benchmark for memory-intensive tasks in reinforcement learning. It standardizes various memory-demanding environments into a single platform to systematically evaluate agent memory.

## Key Features

- **Diverse Memory Testing**: Covers four fundamental memory types:
    - Object Memory
    - Spatial Memory
    - Sequential Memory
    - Memory Capacity

- **Built on the [Gymnasium](https://gymnasium.farama.org) API**, providing:
  - Consistent and standardized environment interfaces
  - Ease of integration with a variety of RL algorithms
  - Flexibility for future extensions and customizations


## List of Tasks

For a detailed description of the tasks, see [Tasks description](#mikasa-base-tasks-description)


## Quick Start

### Installation
```bash
git clone git@github.com:CognitiveAISystems/MIKASA-Base.git
cd MIKASA-Base
pip install .
```


## Basic Usage
```python
import mikasa_base
import gymnasium as gym

# custom task configuration
# env_id = 'MemoryLength-v0'
# env_kwargs = {'memory_length': 10, 'num_bits': 1}

# use predefined task
env_id = 'MemoryLengthHard-v0'
seed = 123

env = gym.make(env_id)

obs, _ = env.reset(seed)

for i in range(101):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Vectorize enviroments
```python
import mikasa_base
import gymnasium as gym

def make_env(env_id, idx, capture_video, run_name, env_kwargs):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

num_envs = 8
env_id = 'MemoryLengthHard-v0'
seed = 123
env_kwargs = {}

envs = gym.vector.AsyncVectorEnv(
    [make_env(env_id, i, False, 'test', env_kwargs) for i in range(num_envs)],
)

obs, _ = envs.reset(seed)

for i in range(101):
    actions = envs.action_space.sample()
    obs, reward, terminated, truncated, info = envs.step(actions)

envs.close()
```


## Example of Training
Code for PPO training is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) 

### PPO with MLP
```bash
python3 baselines/ppo/ppo_mlp.py \
    --env_id='MemoryLength-v0' \
    --env_kwargs memory_length 20 num_bits 1 \
    --num_envs 128 --total_timesteps 10_000_000 \
    --num_steps 21 \
    --num_eval_steps 21
```

### PPO with LSTM
```bash
python3 baselines/ppo/ppo_lstm.py \
    --env_id='MemoryLength-v0' \
    --env_kwargs memory_length 20 num_bits 1 \
    --num_envs 128 --total_timesteps 10_000_000 \
    --num_steps 21 \
    --num_eval_steps 21
```

## Citation
If you find our work useful, please cite our paper:
```
@misc{cherepanov2025mikasa,
      title={Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning}, 
      author={Egor Cherepanov and Nikita Kachaev and Alexey K. Kovalev and Aleksandr I. Panov},
      year={2025},
      eprint={2502.10550},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.10550}, 
}
```

## References

This repository's code is based on and inspired by the work available in the following projects:

- [DeepMind Research](https://github.com/google-deepmind/deepmind-research)
- [bsuite](https://github.com/google-deepmind/bsuite)
- [DTQN](https://github.com/kevslinger/DTQN)
- [Endless Memory Gym](https://github.com/MarcoMeter/endless-memory-gym)
- [Memory Maze](https://github.com/jurgisp/memory-maze)
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid)
- [Numpad Gym](https://github.com/Syrlander/numpad-gym)
- [Memory-RL](https://github.com/twni2016/Memory-RL)
- [PopGym](https://github.com/proroklab/popgym)
- [Memup](https://github.com/griver/memup)

We would like to express our gratitude to the developers of these projects for providing valuable resources and inspiration.



<h1 align="center">MIKASA-Base Tasks description</h1>

| Environment                    | Brief description | Memory Task          | Observation Space   | Action Space         |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|---------------------|----------------------|
| [`MemoryCards-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/MemoryCards)              | Memorize the positions of revealed cards and correctly match pairs while minimizing incorrect guesses.                                                                                                                                                                        | Capacity             | vector              | discrete             |
| [`Numpad-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/Numpad)                     | Memorize the sequence of movements and navigate the rolling ball on a 3×3 grid by following the correct order while avoiding mistakes.                                                                                                                                         | Sequential           | image, vector       | discrete, continuous |
| [`MemoryLength-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/Bsuite)         | Memorize the initial context signal and recall it after a given number of steps to take the correct action.                                                                                                                                                                      | Object               | vector              | discrete             |
| [`Minigrid-Memory-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/MinigridMemory)             | Memorize the object in the starting room and use this information to select the correct path at the junction.                                                                                                                                                                   | Object               | image               | discrete             |
| [`Ballet-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/Ballet)                        | Memorize the sequence of movements performed by each uniquely colored and shaped dancer, then identify and approach the dancer who executed the given pattern.                                                                                                              | Sequential, Object   | image               | discrete             |
| [`Passive-VisualMatch-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/PassiveVisualMatch2D)       | Memorize the target color displayed on the wall during the initial phase. After a brief distractor phase, identify and select the target color among the distractors by stepping on the corresponding ground pad.                                                         | Object               | image               | discrete             |
| [`Passive-T-Maze-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/Passive_T_Maze)              | Memorize the goal’s location upon initial observation, navigate through the maze with limited sensory input, and select the correct path at the junction.                                                                                                                    | Object               | vector              | discrete             |
| [`ViZDoom-two-colors-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/ViZDoom_Two_Colors)         | Memorize the color of the briefly appearing pillar (green or red) and collect items of the same color to survive in the acid-filled room.                                                                                                                                       | Object               | image               | discrete             |
| [`MemoryMaze-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/MemoryMaze)               | Memorize the locations of objects and the maze structure using visual clues, then navigate efficiently to find objects of a specific color and score points.                                                                                                               | Spatial              | image               | discrete             |
| [`MortarMayhem-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/MemoryGym)    | Memorize a sequence of movement commands and execute them in the correct order.                                                                                                                                                                                                | Capacity, Sequential | image               | discrete             |
| [`MysteryPath-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/MemoryGym)    | Memorize the invisible path and navigate it without stepping off.                                                                                                                                                                                                             | Capacity, Spatial    | image               | discrete             |
| [`RepeatFirst-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)       | Memorize the initial value presented at the first step and recall it correctly after receiving a sequence of random values.                                                                                                                                                     | Object               | vector              | discrete             |
| [`RepeatPrevious-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)    | Memorize the value observed at each step and recall the value from \( k \) steps earlier when required.                                                                                                                                                                         | Sequential, Object   | vector              | discrete             |
| [`Autoencode-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)          | Memorize the sequence of cards presented at the beginning and reproduce them in the same order when required.                                                                                                                                                                    | Sequential           | vector              | discrete             |
| [`CountRecall-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)        | Memorize unique values encountered and count how many times a specific value has appeared.                                                                                                                                                                                     | Object, Capacity     | vector              | discrete             |
| [`VelocityOnlyCartPole-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)| Memorize velocity data over time and integrate it to infer the position of the pole for balance control.                                                                                                                                                                         | Sequential           | vector              | continuous           |
| [`MultiarmedBandit-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)  | Memorize the reward probabilities of different slot machines by exploring them and identify the one with the highest expected reward.                                                                                                                                           | Object, Capacity     | vector              | discrete             |
| [`Concentration-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)      | Memorize the positions of revealed cards and match them with previously seen cards to find all matching pairs.                                                                                                                                                                  | Capacity             | vector              | discrete             |
| [`Battleship-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)         | Memorize the coordinates of previous shots and their HIT or MISS feedback to build an internal representation of the board, avoid repeat shots, and strategically target ships for maximum rewards.                                                                   | Spatial              | vector              | discrete             |
| [`MineSweeper-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)       | Memorize revealed grid information and use numerical clues to infer safe tiles while avoiding mines.                                                                                                                                                                           | Spatial              | vector              | discrete             |
| [`LabyrinthExplore-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)  | Memorize previously visited cells and navigate the maze efficiently to discover new, unexplored areas and maximize rewards.                                                                                                                                                      | Spatial              | vector              | discrete             |
| [`LabyrinthEscape-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)   | Memorize the maze layout while exploring and navigate efficiently to find the exit and receive a reward.                                                                                                                                                                         | Spatial              | vector              | discrete             |
| [`HigherLower-v0`](https://github.com/CognitiveAISystems/MIKASA-Base/mikasa_base/POPGym)       | Memorize previously revealed card ranks and predict whether the next card will be higher or lower, updating the reference card after each prediction to maximize rewards.                                                                                           | Object, Sequential   | vector              | discrete             |



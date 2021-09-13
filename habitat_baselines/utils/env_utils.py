#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import time
from collections import defaultdict
from typing import List, Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat.core.registry import registry
from tqdm import tqdm


def make_env_fn(
    config: Config, env_class: Union[Type[Env], Type[RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    # YK: There is lot of ugly code here, the dataset init is being called
    #  multiple times w/o particular reason

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            logging.log(level=logging.WARNING, msg=
                f"you might want to reduce the number of processes as there "
                f"aren't enough number of scenes "
                f"// processes: {num_processes} and scenes: {len(scenes)}"
            )

        random.shuffle(scenes)

    # hack to run overfit faster
    if len(scenes) < num_processes:
        repeat_scenes = num_processes // len(scenes)
        scenes = scenes * repeat_scenes
        assert len(scenes) >= num_processes

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID[i % len(config.SIMULATOR_GPU_ID)]
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)
    # task_config.DATASET.CONTENT_SCENES now contain all scenes splits

    if registry.mapping["debug"]:
        env = make_env_fn(configs[0], env_classes[0])
        env.reset()
        env.step(action={"action": 3, "action_args": {"iid": -1}})
        from cos_eor.utils.debug import debug_viewer
        debug_viewer(env)

    # observations = []
    # for i in range(60):
    #     debug_action = env.action_space.sample()
    #     obs = env.step(action=debug_action)[0]
    #     observations.append(obs)
    #     print(f"Debug Action: {debug_action}")
    #
    # from cos_eor.play_utils import make_video_cv2
    # make_video_cv2(observations, prefix="debug-rgb-", sensor="rgb")
    # make_video_cv2(observations, prefix="debug-rgb-third-", sensor="rgb_3rd_person")

    envs = habitat.ThreadedVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    # envs.reset()
    # inp1 = [{"action": {"action": 1, "action_args": {"iid": 10}}}]
    # envs.step(inp1)
    # envs = habitat.VectorEnv(
    #     make_env_fn=make_env_fn,
    #     env_fn_args=tuple(zip(configs, env_classes)),
    #     workers_ignore_signals=workers_ignore_signals,
    # )

    # envs = habitat.SequentialEnv(
    #     make_env_fn=make_env_fn,
    #     env_fn_args=tuple(zip(configs, env_classes)),
    # )

    # timing code

    # env = make_env_fn(configs[0], env_classes[0])
    # env.reset()
    # num_envs = envs.num_envs
    # envs.reset()

    # # try 400 random actions and report average time for each category
    # possible_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
    # num_actions = len(possible_actions)
    # actions = [2 for _ in range(1000)]
    # envs_actions = [[action] * num_envs for action in actions]
    # envs_time = defaultdict(list)
    #
    # for actions in tqdm(envs_actions, desc="Debug action timings"):
    #     start = time.time()
    #
    #     # individual_times = []
    #     # for i in range(num_envs):
    #     #     start_time = time.time()
    #     #     envs.step_at(i, actions[i])
    #     #     time_take = time.time() - start_time
    #     #     # print(f"Step at {i} w/ action {actions[i]}: {time_take}")
    #     #     individual_times.append(time_take)
    #
    #     mp_time = time.time()
    #     envs.step(actions)
    #     # print(
    #         # f"Average individual times: {sum(individual_times)/len(individual_times)}"
    #         #   f" // MP time: {time.time()-mp_time}")
    #     # env.step(action=actions[0])
    #     end = time.time()
    #     envs_time[actions[0]].append(end-start)
    #
    # for action, times in envs_time.items():
    #     print(f"Action: {possible_actions[action]} || "
    #           f"Avg. Time over {len(times)} "
    #           f"tries: {round(sum(times)/len(times), 4)} secs || "
    #           f"Num Processes: {num_envs}")
    #
    # import pdb
    # pdb.set_trace()

    return envs

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.utils.env_utils import construct_envs

import rlf.rl.utils as rutils
from rlf.exp_mgr.viz_utils import save_mp4
import torch.nn as nn

import sys
sys.path.insert(0, './')
from method.orp_policy_adapter import HabPolicy

@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space
        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        save_path = os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        print('Checkpointed to ', save_path)

        torch.save(
            checkpoint, save_path
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                use_v = {k:dv for k, dv in v.items() if isinstance(k, str)}
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            use_v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def is_simple_env(self):
        return self.config.ENV_NAME != 'Orp-v1'

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats,
        running_episode_counts
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        profiling_wrapper.range_pop()  # compute actions

        if self.is_simple_env():
            step_data = [a.item() for a in actions.to(device="cpu")]
            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
        else:
            step_data = actions.cpu().numpy()
            observations, rewards_l, dones, infos = self.envs.step(actions.cpu().numpy())

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward  # type: ignore
        running_episode_stats["count"] += 1 - masks  # type: ignore

        #for i in range(len(infos)):
        #    if masks[i] != 0:
        #        continue
        #    for k, v in self._extract_scalars_from_info(infos[i]).items():
        #        if k not in running_episode_stats:
        #            running_episode_stats[k] = torch.zeros_like(
        #                running_episode_stats["count"]
        #            )
        #        running_episode_stats[k][i] += v
        #        import ipdb; ipdb.set_trace()

        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v
            running_episode_counts[k] += 1

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        from orp_env_adapter import get_hab_envs
        from method.orp_log_adapter import CustomLogger
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),  # type: ignore
        )

        #with TensorboardWriter(
        #    self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        #) as writer:
        with CustomLogger(not self.config.no_wb, args) as writer:
            for update in range(self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                profiling_wrapper.range_push("rollouts loop")
                for _step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    fps = count_steps / (time.time() - t_start)
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, fps
                        )
                    )
                    writer.add_scalars('metrics', {
                        'fps': fps,
                        'pth_time': pth_time,
                        'env_time': env_time
                        }, count_steps)

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt_{count_steps}.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1
                #if update % self.config.EVAL_INTERVAL == 0:
                #    self._eval_cur(writer, count_steps)

                profiling_wrapper.range_pop()  # train update

            self.envs.close()


    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self.config.EVAL.EMPTY:
            ckpt_dict = {
                    'state_dict': {
                        'actor_critic.dummy_param': nn.Parameter(torch.tensor([0.0]))
                        }
                    }
        else:
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        # Always keep the video directory the same.
        config.defrost()
        config.VIDEO_DIR = self.config.VIDEO_DIR
        config.freeze()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        use_video_option = self.config.VIDEO_OPTION[:]
        if (checkpoint_index+1) % config.CHECKPOINT_RENDER_INTERVAL != 0:
            use_video_option = []
            config.defrost()
            config.hab_high_render = False
            config.freeze()
        else:
            print('Rendering')
            config.defrost()
            config.hab_high_render = True
            config.freeze()

        if len(use_video_option) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        from orp_env_adapter import get_hab_envs, get_hab_args

        if self.config.EVAL.EMPTY:
            tmp_policy = baseline_registry.get_policy(self.config.RL.POLICY.name)(config)
        else:
            tmp_policy = None

        if self.is_simple_env():
            args = get_hab_args(config, './config.yaml')
            self.envs = construct_envs(
                    config,
                    get_env_class(config.ENV_NAME),
                    workers_ignore_signals=True,
                    )
        else:
            self.envs, args = get_hab_envs(config, './config.yaml',
                    True, setup_policy=tmp_policy)

        self._setup_actor_critic_agent(ppo_cfg)
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        if self.actor_critic is not None and isinstance(self.agent.actor_critic, HabPolicy):
            self.agent.actor_critic.init(self.envs.observation_spaces[0], self.envs.action_spaces[0], args)
            self.agent.actor_critic.set_env_ref(self.envs)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        if self.is_simple_env():
            ac_shape = 1
        else:
            ac_shape = self.envs.action_spaces[0].shape[0]
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES,
            ac_shape, device=self.device,
            dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        stats_counts = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)]

        if len(use_video_option) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        use_video_dir = os.path.join(config.VIDEO_DIR,
                "ckpt_%i_%i" % (checkpoint_index, int(step_id)))
        if len(use_video_option) > 0 and not os.path.exists(use_video_dir):
            os.makedirs(use_video_dir)

        cur_render = 0
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            if self.is_simple_env():
                outputs = self.envs.step([a[0].item() for a in actions])
                observations, rewards_l, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
            else:
                observations, rewards_l, dones, infos = self.envs.step(actions.cpu().numpy())

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            if len(use_video_option) > 0:
                if self.is_simple_env():
                    frames = [observations_to_image(observations[i], infos[i])
                            for i in range(len(infos))]
                else:
                    frames = self.envs.render(mode='rgb_array')
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # WE WANT TO RENDER THE FINAL FRAME. But only for manip tasks
                # since we display statistics at the end.
                if len(use_video_option) > 0:
                    frame = frames[i]
                    if self.is_simple_env():
                        if not_done_masks[i].item() != 0:
                            rgb_frames[i].append(np.flip(frame,0))
                    else:
                        rgb_frames[i].append(frame)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    if hasattr(self.actor_critic, 'mod_policy'):
                        # Stats from the modular policy such as failed modules.
                        fsm_dat = self.actor_critic.mod_policy.get_fsm_data()
                        if infos[i]['ep_success'] == 1.0:
                            # Nothing could have been a failure
                            fsm_dat = {k: 0.0 if 'failure' in k else v
                                    for k, v in fsm_dat.items()}
                        episode_stats.update(fsm_dat)
                    episode_stats["reward"] = current_episode_reward[i].item()
                    extracted = self._extract_scalars_from_info(infos[i])
                    episode_stats.update(
                        extracted
                    )
                    for k in extracted:
                        stats_counts[k] += 1
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(use_video_option) > 0:
                        # only the important metrics can make it in the video
                        # filename
                        fname_metrics = {
                                k: v
                                for k, v in self._extract_scalars_from_info(infos[i]).items()
                                if k in ['ep_success', 'ep_constraint_violate',
                                    'spl', 'ep_accum_force_end']
                                }
                        fname_metrics['reward'] = episode_stats['reward']
                        generate_video(
                            video_option=use_video_option,
                            video_dir=use_video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=fname_metrics,
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []
                        if self.config.VIDEO_MAX_RENDER > 0 and cur_render > self.config.VIDEO_MAX_RENDER:
                            # Turn off rendering.
                            self.config.defrost()
                            use_video_option = []
                            self.config.freeze()
                        cur_render += 1

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key in ['reward', 'count']:
                use_count = num_episodes
            else:
                use_count = stats_counts[stat_key]

            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values() if stat_key in v])
                / use_count
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

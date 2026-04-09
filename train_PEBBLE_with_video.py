#!/usr/bin/env python3
"""
train_PEBBLE_with_video.py

Extended version of train_PEBBLE.py with video recording capability.
All training logic is identical to the original version.

VIDEO FEATURES:
- Records training episodes at configurable frequency
- Saves videos with reward overlay showing reward_hat per step
- Automatically manages video storage (keeps only recent N videos)
- Configuration:
  - video_record_frequency: Record every N episodes (default: 1)
  - max_videos_keep: Maximum number of videos to keep (default: 2000)
"""
import numpy as np
import torch
import os
import time
import pickle as pkl

from logger import Logger
from replay_buffer import ReplayBuffer, ProgressDiffReplayBuffer
from reward_model import RewardModel
from reward_model_score import RewardModelScore
from collections import deque
from prompt import clip_env_prompts

import utils
import hydra
from PIL import Image

from vlms.blip_infer_2 import blip2_image_text_matching
from vlms.clip_infer import clip_infer_score as clip_image_text_matching
from vlms.qwen_infer import qwen_image_text_matching
import cv2
import wandb

# VIDEO: Import video recording tools
from video_visualizer import RewardVideoVisualizer, EpisodeRecorder


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.cfg.prompt = clip_env_prompts[cfg.env]
        self.cfg.clip_prompt = clip_env_prompts[cfg.env]
        self.reward = self.cfg.reward  # which type of reward to use
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        # Only create episode_rng when metaworld_random_init is True
        self.metaworld_random_init = getattr(cfg, 'metaworld_random_init', False)
        if self.metaworld_random_init:
            self.episode_rng = np.random.RandomState(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        # Copy prompt file into the log directory for exact reproducibility
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system("cp {}/prompt.py {}/".format(current_file_path, self.logger._log_dir))

        # Build environment
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            self.env = utils.make_classic_control_env(cfg)
            self.log_success = True
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
            self.log_success = True  # Enable success tracking for softgym
        else:
            self.env = utils.make_env(cfg)

        # Override max episode steps if specified (0 = use env default)
        if getattr(cfg, 'max_episode_steps', 0) > 0:
            self.env._max_episode_steps = cfg.max_episode_steps

        # Agent I/O shapes
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # Image sizes / resize factor for image-based reward/progress heads
        image_height = image_width = cfg.image_size
        self.resize_factor = 1
        if "sweep" in cfg.env or 'drawer' in cfg.env or "soccer" in cfg.env:
            image_height = image_width = 300
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "CartPole" in cfg.env:
            image_height = image_width = 200
        if "Cloth" in cfg.env:
            image_height = image_width = 360

        self.image_height = image_height
        self.image_width = image_width

        # Check progress_diff mode early (needed for replay buffer selection)
        self.use_progress_diff_reward = bool(getattr(cfg, "use_progress_diff_reward", False))
        self.terminate_on_success = bool(getattr(cfg, "terminate_on_success", False))

        # Replay buffer capacity (smaller if storing images to control memory)
        _use_episode = bool(getattr(cfg, 'use_episode', False))
        if self.cfg.image_reward:
            ep_cap_episodes = int(getattr(cfg, 'image_replay_capacity_episodes', 0))
            if _use_episode and ep_cap_episodes > 0:
                _max_ep_steps = int(getattr(cfg, 'max_episode_steps', 0)) or getattr(self.env, '_max_episode_steps', 500)
                cap = ep_cap_episodes * _max_ep_steps
            else:
                img_capacity = getattr(cfg, "image_replay_capacity", None)
                cap = int(img_capacity) if img_capacity is not None else 200000
        else:
            cap = int(cfg.replay_buffer_capacity)

        # Select replay buffer based on mode
        if self.use_progress_diff_reward and self.cfg.image_reward:
            # progress_diff mode: online reward = P(s'), relabeling uses per-episode diff (+ optional SG smooth)
            use_smooth_relabel = bool(getattr(cfg, "use_smooth_relabel", False))
            smooth_window = int(getattr(cfg, "smooth_window", 21))
            self.progress_diff_reward_scale = float(getattr(cfg, "progress_diff_reward_scale", 1.0))
            self.progress_diff_discount = float(getattr(cfg, "progress_diff_discount", 1.0))
            self.progress_diff_scale_by_inv_one_minus_gamma = bool(
                getattr(cfg, "progress_diff_scale_by_inv_one_minus_gamma", False)
            )
            (
                self.progress_diff_effective_reward_scale,
                self.progress_diff_inv_one_minus_gamma_scale,
            ) = utils.get_progress_diff_reward_scale(
                reward_scale=self.progress_diff_reward_scale,
                discount=self.progress_diff_discount,
                scale_by_inv_one_minus_gamma=self.progress_diff_scale_by_inv_one_minus_gamma,
            )
            self.replay_buffer = ProgressDiffReplayBuffer(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                cap,
                self.device,
                image_size=image_height,
                smooth_window=smooth_window,
                smooth_relabel=use_smooth_relabel,
                reward_scale=self.progress_diff_reward_scale,
                discount=self.progress_diff_discount,
                scale_by_inv_one_minus_gamma=self.progress_diff_scale_by_inv_one_minus_gamma)
            print(
                "[ProgressDiff] Using ProgressDiffReplayBuffer with "
                f"capacity={cap}, smooth_window={smooth_window}, "
                f"smooth_relabel={use_smooth_relabel}, "
                f"reward_scale={self.progress_diff_reward_scale}, "
                f"discount={self.progress_diff_discount}, "
                f"scale_by_inv_one_minus_gamma="
                f"{self.progress_diff_scale_by_inv_one_minus_gamma}, "
                f"inv_one_minus_gamma_scale={self.progress_diff_inv_one_minus_gamma_scale}, "
                f"effective_reward_scale={self.progress_diff_effective_reward_scale}"
            )
        else:
            # baseline mode: use original ReplayBuffer
            use_smooth_relabel = bool(getattr(cfg, "use_smooth_relabel", False))
            smooth_window = int(getattr(cfg, "smooth_window", 21))
            self.replay_buffer = ReplayBuffer(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                cap,
                self.device,
                store_image=self.cfg.image_reward,
                image_size=image_height,
                smooth_relabel=use_smooth_relabel,
                smooth_window=smooth_window)
            if use_smooth_relabel:
                print(f"[Baseline] SG smooth relabel enabled: smooth_window={smooth_window}")

        # Basic logging counters
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # NEW: Count how many episodes succeeded across the entire training run
        # (episode-level success; incremented by 1 for each successful episode)
        self.total_success_episodes = 0

        # Moving average buffer for eval success rate (last 10 evals)
        self.eval_success_history = deque(maxlen=10)

        # Instantiate reward/progress model (same class; trained from preferences)
        reward_model_class = RewardModel
        if self.reward == 'learn_from_preference':
            reward_model_class = RewardModel
        elif self.reward == 'learn_from_score':
            reward_model_class = RewardModelScore

        self.reward_model = reward_model_class(
            # Original PEBBLE parameters
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            capacity=cfg.max_feedback * 2,

            # VLM parameters
            vlm_label=cfg.vlm_label,
            vlm=cfg.vlm,
            env_name=cfg.env,
            clip_prompt=clip_env_prompts[cfg.env],
            log_dir=self.logger._log_dir,
            flip_vlm_label=cfg.flip_vlm_label,
            cached_label_path=cfg.cached_label_path,
            save_query_interval=cfg.save_query_interval,

            # Image-based reward/progress model parameters
            image_reward=cfg.image_reward,
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
            debug=cfg.vlm_debug,
        )

        # Optional model loading
        if self.cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, 1000000)

        if self.cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, 1000000)

        self.progress_target = None

        # VIDEO: Initialize video recording system
        print("\n" + "="*60)
        print("VIDEO RECORDING ENABLED")
        print("="*60)

        video_save_dir = os.path.join(self.work_dir, 'training_videos')

        # Get max episode steps (check multiple attributes for different env types)
        max_ep_steps = 500  # default
        if hasattr(self.env, '_max_episode_steps'):
            max_ep_steps = self.env._max_episode_steps
        elif hasattr(self.env, 'max_path_length'):
            max_ep_steps = self.env.max_path_length
        elif hasattr(self.env, 'horizon'):
            max_ep_steps = self.env.horizon  # softgym uses horizon

        # Create visualizer with dynamic fps based on frame count
        self.video_visualizer = RewardVideoVisualizer(
            env_frame_size=(image_height, image_width),
            plot_size=(400, 300),
            fps=30,
            max_episode_length=max_ep_steps,
            save_dir=video_save_dir
        )

        # Create episode recorder
        record_freq = cfg.video_record_frequency
        max_videos = cfg.max_videos_keep

        # VIDEO: Get save-success-only setting
        self.save_env_reward_video_success_only = cfg.save_env_reward_video_success_only

        self.episode_recorder = EpisodeRecorder(
            visualizer=self.video_visualizer,
            record_frequency=record_freq,
            max_videos=max_videos
        )

        print(f"Video save directory: {video_save_dir}")
        print(f"Recording frequency: Every {record_freq} episodes")
        print(f"Max videos to keep: {max_videos}")
        print(f"Frame size: {image_height}x{image_width}")
        print(f"Save success only: {self.save_env_reward_video_success_only}")
        print("="*60 + "\n")

    def evaluate(self, save_additional=False, eval_cnt=None, episode=None):
        """Run evaluation episodes.
        Local GIFs: save for EVERY episode (episode 0..N-1) under eval_gifs/.
        W&B: upload ONLY episode 0 as a video artifact per evaluate() call.
        """
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for ep_idx in range(self.cfg.num_eval_episodes):
            print("evaluating episode {}".format(ep_idx))
            images = []
            if self.metaworld_random_init:
                # Random seed each episode when metaworld_random_init is True
                eval_seed = self.episode_rng.randint(400, 500)
                np.random.seed(eval_seed)
                try:
                    obs = self.env.reset(seed=eval_seed)
                except TypeError:
                    self.env.seed(eval_seed)
                    obs = self.env.reset()
            else:
                # Natural RNG progression when metaworld_random_init is False
                obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]

            self.agent.reset()
            done = False
            episode_reward = 0          # Accumulate reward_hat
            true_episode_reward = 0     # Accumulate true env reward
            if self.log_success:
                episode_success = 0

            ep_info = []
            rewards = []
            t_idx = 0
            curr_state_rgb = None  # for progress_diff online reward
            # For progress_diff: render s_0 so step-0 reward = P(s_1) - P(s_0), not 0
            if self.use_progress_diff_reward and self.cfg.image_reward:
                if "metaworld" in self.cfg.env:
                    _s0 = self.env.render()
                    _s0 = _s0[::-1, :, :]
                    if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                        _s0 = _s0[100:400, 100:400, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    _s0 = self.env.render(mode='rgb_array')
                elif 'softgym' in self.cfg.env:
                    _s0 = self.env.render(mode='rgb_array', hide_picker=True)
                else:
                    _s0 = self.env.render(mode='rgb_array')
                if 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                    _s0 = cv2.resize(_s0, (self.image_height, self.image_width))
                curr_state_rgb = _s0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                try:
                    obs, reward, done, extra = self.env.step(action)
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated
                ep_info.append(extra)

                rewards.append(reward)
                if "metaworld" in self.cfg.env:
                    rgb_image = self.env.render()
                    rgb_image = rgb_image[::-1, :, :]
                    if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                        rgb_image = rgb_image[100:400, 100:400, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    rgb_image = self.env.render(mode='rgb_array')
                elif 'softgym' in self.cfg.env:
                    rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
                else:
                    rgb_image = self.env.render(mode='rgb_array')

                if self.cfg.image_reward and 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                    rgb_image = cv2.resize(rgb_image, (self.image_height, self.image_width))

                if 'softgym' not in self.cfg.env:
                    images.append(rgb_image)

                # ===================== reward_hat computation for eval =====================
                if self.reward == 'learn_from_preference' or self.reward == 'learn_from_score':
                    if self.use_progress_diff_reward and self.cfg.image_reward:
                        # Online raw diff with optional 1 / (1 - gamma) scaling.
                        if rgb_image is None or curr_state_rgb is None:
                            reward_hat = 0.0
                        else:
                            curr_img = curr_state_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                            curr_img = curr_img[:, ::self.resize_factor, ::self.resize_factor]
                            curr_img = curr_img.reshape(1, 3, curr_img.shape[1], curr_img.shape[2])
                            next_img = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                            next_img = next_img[:, ::self.resize_factor, ::self.resize_factor]
                            next_img = next_img.reshape(1, 3, next_img.shape[1], next_img.shape[2])
                            self.reward_model.eval()
                            p_curr = float(self.reward_model.r_hat(curr_img))
                            p_next = float(self.reward_model.r_hat(next_img))
                            reward_hat = (
                                self.progress_diff_discount * p_next - p_curr
                            ) * self.progress_diff_effective_reward_scale
                            self.reward_model.train()
                        curr_state_rgb = rgb_image
                    elif not self.cfg.image_reward:
                        self.reward_model.eval()
                        reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                        self.reward_model.train()
                    else:
                        image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                        image = image[:, ::self.resize_factor, ::self.resize_factor]
                        image = image.reshape(1, 3, image.shape[1], image.shape[2])
                        self.reward_model.eval()
                        reward_hat = self.reward_model.r_hat(image)
                        self.reward_model.train()

                elif self.reward == 'blip2_image_text_matching':
                    query_image = rgb_image
                    query_prompt = clip_env_prompts[self.cfg.env]
                    reward_hat = blip2_image_text_matching(query_image, query_prompt) * 2 - 1
                    if self.cfg.flip_vlm_label:
                        reward_hat = -reward_hat

                elif self.reward == 'clip_image_text_matching':
                    query_image = rgb_image
                    query_prompt = clip_env_prompts[self.cfg.env]
                    reward_hat = clip_image_text_matching(query_image, query_prompt) * 2 - 1
                    if self.cfg.flip_vlm_label:
                        reward_hat = -reward_hat

                elif self.reward == 'qwen_image_text_matching':
                    query_image = rgb_image
                    query_prompt = clip_env_prompts[self.cfg.env]
                    reward_hat = qwen_image_text_matching(query_image, query_prompt) * 2 - 1
                    if self.cfg.flip_vlm_label:
                        reward_hat = -reward_hat

                elif self.reward == 'gt_task_reward':
                    reward_hat = reward

                elif self.reward == 'sparse_task_reward':
                    reward_hat = extra['success']

                else:
                    reward_hat = reward
                # ===========================================================================

                episode_reward += reward_hat
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

                # Terminate eval episode on success if enabled (mirrors training behavior)
                if self.terminate_on_success and self.log_success and extra.get('success', 0):
                    break

                t_idx += 1
                if self.cfg.mode == 'eval' and t_idx > 50:
                    break

            all_ep_infos.append(ep_info)
            if 'softgym' in self.cfg.env:
                images = self.env.video_frames

            video_frames = np.array(images)

            # --- NEW: Always save a local GIF for EVERY eval episode ---
            save_gif_path = os.path.join(
                save_gif_dir,
                'step{:07}_episode{:02}_{}.gif'.format(self.step, ep_idx, round(true_episode_reward, 2)))
            try:
                utils.save_numpy_as_gif(video_frames, save_gif_path)
            except Exception as e:
                print(f"Failed to save eval GIF for episode {ep_idx}: {e}")

            # --- W&B: Upload ONLY episode 0 per evaluate() call ---
            if ep_idx == 0:
                try:
                    video_tensor = video_frames.transpose(0, 3, 1, 2)
                    wandb.log(
                        {"eval_step/video": wandb.Video(video_tensor, fps=12, format="gif")},
                        step=self.step
                    )
                except Exception as e:
                    print(f"Failed to log eval video for episode {ep_idx}: {e}")

            if save_additional:
                save_image_dir = os.path.join(self.logger._log_dir, 'eval_images')
                if not os.path.exists(save_image_dir):
                    os.makedirs(save_image_dir)
                for i, image in enumerate(images):
                    save_image_path = os.path.join(
                        save_image_dir, 'step{:07}_episode{:02}_{}.png'.format(self.step, ep_idx, i))
                    image = Image.fromarray(image)
                    image.save(save_image_path)
                save_reward_path = os.path.join(self.logger._log_dir, "eval_reward")
                if not os.path.exists(save_reward_path):
                    os.makedirs(save_reward_path)
                with open(os.path.join(save_reward_path, "step{:07}_episode{:02}.pkl".format(self.step, ep_idx)), "wb") as f:
                    pkl.dump(rewards, f)

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        # Aggregate eval metrics
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        for key, value in extra.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.logger.log('eval/' + key, value, self.step)

        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)

        self.logger.dump(self.step)

        eval_metrics = {
            "eval_step/episode_reward": average_episode_reward,
            "eval_step/true_episode_reward": average_true_episode_reward,
        }
        if self.log_success:
            self.eval_success_history.append(success_rate)
            success_rate_ma10 = float(np.mean(self.eval_success_history))
            eval_metrics["eval_step/success_rate"] = success_rate
            eval_metrics["eval_step/success_rate_ma10"] = success_rate_ma10
        wandb.log(eval_metrics, step=self.step)
        if episode is not None:
            ep_eval_metrics = {f"eval_episode/{k[10:]}": v for k, v in eval_metrics.items()}
            ep_eval_metrics["episode"] = episode
            wandb.log(ep_eval_metrics, step=self.step)

    def learn_reward(self, first_flag=0):
        """Collect preference labels and train the reward/progress model."""
        labeled_queries = 0
        if first_flag == 1:
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        total_acc = 0
        if self.labeled_feedback > 0:
            # Preference training loop
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    self.reward_model.train()
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    self.reward_model.train()
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break

        if self.reward == 'learn_from_preference':
            print("Reward/Value model is updated!! ACC: " + str(total_acc))
        elif self.reward == 'learn_from_score':
            print("Reward/Value model is updated!! MSE: " + str(total_acc))
        return total_acc, self.reward_model.vlm_label_acc

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        # Keep recent 10 train returns
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        # Track success history for the last 100 episodes
        success_history = deque(maxlen=100)

        interact_count = 0
        reward_learning_acc = 0
        vlm_acc = 0
        eval_cnt = 0

        # Step-based video window state
        video_step_interval = int(getattr(self.cfg, 'video_step_interval', 10000))
        video_step_offset = getattr(self.cfg, 'video_step_offset', None)
        if video_step_offset is None:
            next_video_checkpoint = video_step_interval
        else:
            next_video_checkpoint = int(video_step_offset)
        video_window_episodes = int(getattr(self.cfg, 'video_window_episodes', 5))
        video_window_remaining = 0

        # Previous frame buffer for progress-difference online reward
        curr_state_rgb = None

        # ── Episode-based training config ──────────────────────────────────────
        _use_episode = bool(getattr(self.cfg, 'use_episode', False))
        if _use_episode:
            _num_train_ep   = int(self.cfg.num_train_episodes)
            _eval_ep_freq   = int(getattr(self.cfg, 'eval_episode_frequency', 20))
            _interact_ep    = int(getattr(self.cfg, 'num_interact_episodes', 8))
            _save_ep_int    = int(getattr(self.cfg, 'save_episode_interval', 50))
            _video_ep_int   = int(getattr(self.cfg, 'video_episode_interval', 50))
            _num_seed_ep    = int(getattr(self.cfg, 'num_seed_episodes', 2))
            _num_unsup_ep   = int(getattr(self.cfg, 'num_unsup_episodes', 10))
            _supervised_started = False   # one-time flag for first preference learning
        # ───────────────────────────────────────────────────────────────────────

        while (episode <= _num_train_ep) if _use_episode else (self.step < self.cfg.num_train_steps):
            if done:
                # VIDEO: End episode recording (for the episode that just finished)
                if self.step > 0:
                    # Determine whether to save this episode's video
                    should_save_video = True
                    if self.save_env_reward_video_success_only and self.log_success:
                        # Only save if episode was successful
                        should_save_video = (episode_success > 0)

                    # ClothFoldDiagonal: horizon=1 so only 1 frame per step.
                    # Replace with softgym's internal video_frames for a watchable video.
                    if 'ClothFoldDiagonal' in self.cfg.env and self.episode_recorder.recording:
                        sim_frames = getattr(self.env, 'video_frames', None)
                        if sim_frames is not None and len(sim_frames) > 1:
                            last_reward = self.episode_recorder.visualizer.rewards[-1] if self.episode_recorder.visualizer.rewards else 0.0
                            self.episode_recorder.visualizer.reset_episode()
                            for f in sim_frames:
                                self.episode_recorder.visualizer.add_frame(f, last_reward)

                    self.episode_recorder.end_episode(save_video=should_save_video, save_gif=False)

                # Episode boundary logging (for the episode that just finished)
                if self.step > 0:
                    duration = time.time() - start_time
                    self.logger.log('train/duration', duration, self.step)
                    self.logger.log('train/reward_learning_acc', reward_learning_acc, self.step)
                    self.logger.log('train/vlm_acc', vlm_acc, self.step)
                    for key, value in extra.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            self.logger.log('train/' + key, value, self.step)

                    # NEW: increment and log the cumulative count of successful episodes
                    if self.log_success:
                        self.total_success_episodes += int(episode_success)  # episode_success ∈ {0,1}
                        self.logger.log('train/total_success_episodes', self.total_success_episodes, self.step)

                        # Track success history
                        success_history.append(int(episode_success))

                        # Every 100 episodes, compute and log success rate
                        if episode > 0 and episode % 100 == 0:
                            # Calculate success rate for last 100 episodes
                            current_success_rate = sum(success_history) / len(success_history)

                            # Log to wandb (wandb will automatically create plots)
                            wandb.log({
                                "train_step/success_rate_per_100ep": current_success_rate,
                            }, step=self.step)
                            if _use_episode:
                                wandb.log({
                                    "train_episode/success_rate_per_100ep": current_success_rate,
                                    "episode": episode,
                                }, step=self.step)

                    start_time = time.time()

                # Periodic evaluation
                if _use_episode:
                    if episode > 0 and episode % _eval_ep_freq == 0:
                        self.logger.log('eval/episode', episode, self.step)
                        self.evaluate(eval_cnt=eval_cnt, episode=episode)
                        eval_cnt += 1
                elif self.step > 0 and self.step >= (eval_cnt + 1) * self.cfg.eval_frequency:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate(eval_cnt=eval_cnt)
                    eval_cnt += 1

                # Per-episode scalars
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                # Pack and send to W&B
                if self.step > 0:
                    train_metrics = {
                        "train_step/episode_reward": episode_reward,
                        "train_step/true_episode_reward": true_episode_reward,
                        "train_step/total_feedback": self.total_feedback,
                        "train_step/labeled_feedback": self.labeled_feedback,
                        "train_step/reward_learning_acc": reward_learning_acc,
                        "train_step/vlm_acc": vlm_acc,
                    }
                    if self.log_success:
                        train_metrics["train_step/episode_success"] = episode_success
                        # NEW: also send the cumulative success count (shows as a curve in W&B)
                        train_metrics["train_step/total_success_episodes"] = self.total_success_episodes
                    wandb.log(train_metrics, step=self.step)
                    if _use_episode:
                        ep_train_metrics = {f"train_episode/{k[11:]}": v for k, v in train_metrics.items()}
                        ep_train_metrics["episode"] = episode
                        wandb.log(ep_train_metrics, step=self.step)

                # In episode mode, the final episode has just finished its done-block
                # above. Exit before resetting and starting episode N+1.
                if _use_episode and self.step > 0 and episode >= _num_train_ep:
                    break

                if self.metaworld_random_init:
                    # Random seed each episode when metaworld_random_init is True
                    train_seed = self.episode_rng.randint(0, 400)
                    np.random.seed(train_seed)
                    try:
                        obs = self.env.reset(seed=train_seed)
                    except TypeError:
                        self.env.seed(train_seed)
                        obs = self.env.reset()
                else:
                    # Natural RNG progression when metaworld_random_init is False
                    obs = self.env.reset()
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

                traj_images = []
                ep_info = []

                # Debug variables for reward_hat anomaly detection
                prev_step = None
                prev_obj_to_target = None
                prev_reward_hat = None
                prev_members = None
                # Reset previous frame at episode boundary (progress_diff only)
                curr_state_rgb = None
                # For progress_diff: render s_0 so step-0 reward = P(s_1) - P(s_0), not 0
                if self.use_progress_diff_reward and self.cfg.image_reward:
                    if "metaworld" in self.cfg.env:
                        _s0 = self.env.render()
                        _s0 = _s0[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            _s0 = _s0[100:400, 100:400, :]
                    elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                        _s0 = self.env.render(mode='rgb_array')
                    elif 'softgym' in self.cfg.env:
                        _s0 = self.env.render(mode='rgb_array', hide_picker=True)
                    else:
                        _s0 = self.env.render(mode='rgb_array')
                    if 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                        _s0 = cv2.resize(_s0, (self.image_height, self.image_width))
                    curr_state_rgb = _s0

                # VIDEO: open a new recording window every N steps (step mode) or N episodes (episode mode)
                if _use_episode:
                    # Record the last `video_window_episodes` episodes in each interval.
                    # Episode counting here is 1-indexed because `episode` has already been
                    # incremented for the new episode before we enter this block.
                    _video_ep_trigger = max(0, _video_ep_int - video_window_episodes)
                    if episode > 0 and ((episode - 1) % _video_ep_int) == _video_ep_trigger:
                        video_window_remaining = video_window_episodes
                elif self.step >= next_video_checkpoint:
                    video_window_remaining = video_window_episodes
                    next_video_checkpoint += video_step_interval
                if video_window_remaining > 0:
                    self.episode_recorder.recording = True
                    self.video_visualizer.reset_episode()
                    video_window_remaining -= 1
                else:
                    self.episode_recorder.recording = False

                # ── Episode-mode triggers (VLM / relabel / save) ───────────────
                if _use_episode and self.step > 0:
                    _trans_ep = _num_seed_ep + _num_unsup_ep
                    # ① First transition into supervised phase (fires exactly once)
                    if episode == _trans_ep + 1 and not _supervised_started:
                        _supervised_started = True
                        print("finished unsupervised exploration!!")
                        if self.reward in ('learn_from_preference', 'learn_from_score'):
                            self.reward_model.change_batch(1)
                            new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                            self.reward_model.set_teacher_thres_skip(new_margin)
                            self.reward_model.set_teacher_thres_equal(new_margin)
                            reward_learning_acc, vlm_acc = self.learn_reward(first_flag=1)
                            self.reward_model.eval()
                            self.replay_buffer.relabel_with_predictor(self.reward_model)
                            self.reward_model.train()
                        self.agent.reset_critic()
                        self.agent.update_after_reset(
                            self.replay_buffer, self.logger, self.step,
                            gradient_update=self.cfg.reset_update,
                            policy_update=True)
                        interact_count = 0
                    # ② Periodic preference learning + relabeling (every _interact_ep episodes)
                    elif episode > _trans_ep + 1:
                        interact_count += 1
                        if self.total_feedback < self.cfg.max_feedback and (
                                self.reward in ('learn_from_preference', 'learn_from_score')):
                            if interact_count >= _interact_ep:
                                self.reward_model.change_batch(1)
                                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                                self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                                self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                                if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                                    self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                                reward_learning_acc, vlm_acc = self.learn_reward()
                                self.reward_model.eval()
                                self.replay_buffer.relabel_with_predictor(self.reward_model)
                                self.reward_model.train()
                                interact_count = 0
                    # ③ Model checkpoint
                    if episode > 0 and episode % _save_ep_int == 0:
                        self.agent.save(model_save_dir, self.step)
                        self.reward_model.save(model_save_dir, self.step)
                # ───────────────────────────────────────────────────────────────

            # Sample an action
            if _use_episode:
                if episode <= _num_seed_ep:
                    action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=True)
            elif self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # Phase-dependent per-step updates
            if _use_episode:
                # Episode mode: VLM/relabel triggers are handled in the if-done block above.
                # Here we only do the per-step agent update based on current phase.
                _trans_ep = _num_seed_ep + _num_unsup_ep
                if episode > _trans_ep:
                    self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                elif episode > _num_seed_ep:
                    if self.step % 1000 == 0:
                        print("unsupervised exploration!!")
                    self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                                gradient_update=1, K=self.cfg.topK)
                # seed phase: no update
            else:
                # ── Original step-based logic (unchanged) ────────────────────────
                # Switch from unsupervised to supervised preference learning
                if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                    print("finished unsupervised exploration!!")

                    if self.reward == 'learn_from_preference' or self.reward == 'learn_from_score':
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        # Optional teacher thresholds
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin)
                        self.reward_model.set_teacher_thres_equal(new_margin)

                        # First preference learning
                        reward_learning_acc, vlm_acc = self.learn_reward(first_flag=1)

                        # Relabel replay with the updated model
                        self.reward_model.eval()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        self.reward_model.train()

                    # Reset critic after unsupervised exploration
                    self.agent.reset_critic()

                    # Warmup updates
                    self.agent.update_after_reset(
                        self.replay_buffer, self.logger, self.step,
                        gradient_update=self.cfg.reset_update,
                        policy_update=True)

                    interact_count = 0

                elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                    # Periodic preference learning and relabeling
                    if self.total_feedback < self.cfg.max_feedback and (
                            self.reward == 'learn_from_preference' or self.reward == 'learn_from_score'):
                        if interact_count == self.cfg.num_interact:
                            if self.cfg.reward_schedule == 1:
                                frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                                if frac == 0:
                                    frac = 0.01
                            elif self.cfg.reward_schedule == 2:
                                frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                            else:
                                frac = 1
                            self.reward_model.change_batch(frac)

                            new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                            self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                            self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                            # Avoid exceeding max_feedback
                            if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                                self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                            reward_learning_acc, vlm_acc = self.learn_reward()
                            self.reward_model.eval()
                            self.replay_buffer.relabel_with_predictor(self.reward_model)
                            self.reward_model.train()
                            interact_count = 0

                    self.agent.update(self.replay_buffer, self.logger, self.step, 1)

                # Unsupervised exploration updates (state entropy) before preference learning kicks in
                elif self.step > self.cfg.num_seed_steps:
                    if self.step % 1000 == 0:
                        print("unsupervised exploration!!")
                    self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                                gradient_update=1, K=self.cfg.topK)
                # ─────────────────────────────────────────────────────────────────

            # Environment step
            try:  # Handle different gym APIs
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
            ep_info.append(extra)

            # Capture image if needed
            if self.cfg.vlm_label or self.reward in ['blip2_image_text_matching', 'clip_image_text_matching', 'qwen_image_text_matching'] or \
               (self.cfg.image_reward and self.reward not in ["gt_task_reward", "sparse_task_reward"]):
                if "metaworld" in self.cfg.env:
                    rgb_image = self.env.render()
                    rgb_image = rgb_image[::-1, :, :]
                    if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                        rgb_image = rgb_image[100:400, 100:400, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    rgb_image = self.env.render(mode='rgb_array')
                elif 'softgym' in self.cfg.env:
                    rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
                else:
                    rgb_image = self.env.render(mode='rgb_array')

                if self.cfg.image_reward and 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                    rgb_image = cv2.resize(rgb_image, (self.image_height, self.image_width))
                traj_images.append(rgb_image)
            else:
                rgb_image = None

            # ===================== reward computation (train) =====================
            p_curr = float('nan')
            p_next = float('nan')
            _curr_img_for_buf = None  # will hold s_t for buffer image (progress_diff only)
            if self.reward == 'learn_from_preference' or self.reward == 'learn_from_score':
                if self.use_progress_diff_reward and self.cfg.image_reward:
                    # Online raw diff with optional 1 / (1 - gamma) scaling.
                    if rgb_image is None or curr_state_rgb is None:
                        reward_hat = 0.0
                    else:
                        curr_img = curr_state_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                        curr_img = curr_img[:, ::self.resize_factor, ::self.resize_factor]
                        curr_img = curr_img.reshape(1, 3, curr_img.shape[1], curr_img.shape[2])
                        next_img = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                        next_img = next_img[:, ::self.resize_factor, ::self.resize_factor]
                        next_img = next_img.reshape(1, 3, next_img.shape[1], next_img.shape[2])
                        self.reward_model.eval()
                        p_curr = float(self.reward_model.r_hat(curr_img))
                        p_next = float(self.reward_model.r_hat(next_img))
                        reward_hat = (
                            self.progress_diff_discount * p_next - p_curr
                        ) * self.progress_diff_effective_reward_scale
                        self.reward_model.train()
                    # Capture s_t BEFORE overwriting curr_state_rgb with s_{t+1}
                    _curr_img_for_buf = curr_state_rgb[::self.resize_factor, ::self.resize_factor, :] if curr_state_rgb is not None else None
                    curr_state_rgb = rgb_image
                elif not self.cfg.image_reward:
                    self.reward_model.eval()
                    if getattr(self.cfg, 'reward_hat_debug', False):
                        reward_hat, r_hats_each = self.reward_model.r_hat(
                            np.concatenate([obs, action], axis=-1), return_members=True)
                    else:
                        reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                    self.reward_model.train()
                else:
                    image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                    image = image[:, ::self.resize_factor, ::self.resize_factor]
                    image = image.reshape(1, 3, image.shape[1], image.shape[2])
                    self.reward_model.eval()
                    if getattr(self.cfg, 'reward_hat_debug', False):
                        reward_hat, r_hats_each = self.reward_model.r_hat(image, return_members=True)
                    else:
                        reward_hat = self.reward_model.r_hat(image)
                    self.reward_model.train()

                # Debug: detect anomalies in reward_hat (baseline mode only)
                if not self.use_progress_diff_reward and getattr(self.cfg, 'reward_hat_debug', False) and 'obj_to_target' in extra:
                    curr_obj_to_target = extra['obj_to_target']
                    curr_members = r_hats_each.flatten()

                    # Always print episode step 0 info
                    if episode_step == 0:
                        print(f"[EPISODE START] Step {self.step}: obj_to_target={curr_obj_to_target:.4f}, reward_hat={reward_hat:.3f}, members={curr_members}")

                    # Detect anomalies (only after step 0)
                    if prev_obj_to_target is not None and prev_reward_hat is not None:
                        obj_threshold = 0.001  # minimum change in obj_to_target to be considered movement
                        reward_threshold = 0.05  # minimum change in reward_hat to be considered significant

                        # Case 1: opened more (obj_to_target decreased) but reward dropped
                        opened_more = curr_obj_to_target < prev_obj_to_target - obj_threshold
                        reward_dropped = reward_hat < prev_reward_hat - reward_threshold
                        # Case 2: opened less (obj_to_target increased) but reward increased
                        opened_less = curr_obj_to_target > prev_obj_to_target + obj_threshold
                        reward_increased = reward_hat > prev_reward_hat + reward_threshold

                        if (opened_more and reward_dropped) or (opened_less and reward_increased):
                            anomaly_type = "BETTER->WORSE" if opened_more else "WORSE->BETTER"
                            print(f"[ANOMALY {anomaly_type}]")
                            print(f"  Step {prev_step}: obj_to_target={prev_obj_to_target:.4f}, reward_hat={prev_reward_hat:.3f}, members={prev_members}")
                            print(f"  Step {self.step}: obj_to_target={curr_obj_to_target:.4f}, reward_hat={reward_hat:.3f}, members={curr_members}")

                    prev_step = self.step
                    prev_obj_to_target = curr_obj_to_target
                    prev_reward_hat = reward_hat
                    prev_members = curr_members

            elif self.reward == 'blip2_image_text_matching':
                query_image = rgb_image
                query_prompt = clip_env_prompts[self.cfg.env]
                # Scale to [-1, 1] since tanh is used in the reward/progress head
                reward_hat = blip2_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat

            elif self.reward == 'clip_image_text_matching':
                query_image = rgb_image
                query_prompt = clip_env_prompts[self.cfg.env]
                reward_hat = clip_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat

            elif self.reward == 'qwen_image_text_matching':
                query_image = rgb_image
                query_prompt = clip_env_prompts[self.cfg.env]
                reward_hat = qwen_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat

            elif self.reward == 'gt_task_reward':
                reward_hat = reward

            elif self.reward == 'sparse_task_reward':
                reward_hat = extra['success']

            else:
                reward_hat = reward
            # =====================================================================

            # VIDEO: Record frame with reward_hat for this step
            if rgb_image is not None:
                self.episode_recorder.add_step(rgb_image, float(reward_hat))

            # Per-step W&B logging (scalars)
            log_dict = {
                "train_step/reward_hat": float(reward_hat),
                "train_step/true_reward": float(reward),
            }
            if self.use_progress_diff_reward and self.cfg.image_reward:
                log_dict["train_step/p_curr"] = p_curr
                log_dict["train_step/p_next"] = p_next
            wandb.log(log_dict, step=self.step)

            # Allow infinite bootstrap
            done = float(done)
            if 'softgym' not in self.cfg.env:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            else:
                done_no_max = done

            # Terminate episode on success if enabled
            if self.terminate_on_success and self.log_success and extra.get('success', 0):
                done = 1.0
                done_no_max = 1.0  # success = natural termination, not timeout

            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # Add transition to preference/progress training buffer (Signal A)
            if self.reward == 'learn_from_preference' or self.reward == 'learn_from_score':
                self.reward_model.add_data(obs, action, reward, done, img=rgb_image)

            # Push transition into replay buffer (image/non-image paths)
            if self.cfg.image_reward and self.reward not in ["gt_task_reward", "sparse_task_reward"]:
                if self.use_progress_diff_reward:
                    # Store s_t (current state) in buffer; store s_{t+1} as terminal_image when done.
                    # _curr_img_for_buf was captured BEFORE curr_state_rgb was updated to s_{t+1}.
                    _buf_img = _curr_img_for_buf if _curr_img_for_buf is not None else rgb_image[::self.resize_factor, ::self.resize_factor, :]
                    _term_img = rgb_image[::self.resize_factor, ::self.resize_factor, :] if done > 0.5 else None
                    self.replay_buffer.add(
                        obs, action, reward_hat, next_obs, done, done_no_max,
                        image=_buf_img,
                        terminal_image=_term_img)
                else:
                    self.replay_buffer.add(
                        obs, action, reward_hat, next_obs, done, done_no_max,
                        image=rgb_image[::self.resize_factor, ::self.resize_factor, :])
            else:
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            if not _use_episode:
                interact_count += 1

            # Periodic checkpointing (step mode only; episode mode saves in the if-done block)
            if not _use_episode and self.step % self.cfg.save_interval == 0 and self.step > 0:
                self.agent.save(model_save_dir, self.step)
                self.reward_model.save(model_save_dir, self.step)

        # ── Finalize any in-progress recorded episode ──────────────────────────
        # The while loop can exit before the next done-block runs:
        #   - episode mode: after the last scheduled episode
        #   - step mode: after hitting num_train_steps mid-episode
        # In both cases, a currently recorded episode would otherwise never reach
        # end_episode(), so flush it here.
        if self.episode_recorder.recording:
            should_save_video = True
            if self.save_env_reward_video_success_only and self.log_success:
                should_save_video = (episode_success > 0)
            # ClothFoldDiagonal: replace single frame with softgym simulation frames
            if 'ClothFoldDiagonal' in self.cfg.env and self.episode_recorder.recording:
                sim_frames = getattr(self.env, 'video_frames', None)
                if sim_frames is not None and len(sim_frames) > 1:
                    last_reward = self.episode_recorder.visualizer.rewards[-1] if self.episode_recorder.visualizer.rewards else 0.0
                    self.episode_recorder.visualizer.reset_episode()
                    for f in sim_frames:
                        self.episode_recorder.visualizer.add_frame(f, last_reward)
            self.episode_recorder.end_episode(save_video=should_save_video, save_gif=False)

        # ── Episode-mode: finalize any trailing success-rate window ────────────
        if _use_episode:
            # Final eval / checkpoint / video flush already happen in the regular
            # done-block above. Only the trailing partial 100-episode success
            # window still needs an explicit log here.
            trailing_window = _num_train_ep % 100
            if self.log_success and len(success_history) > 0 and trailing_window != 0:
                trailing_successes = list(success_history)[-trailing_window:]
                final_rate = sum(trailing_successes) / len(trailing_successes)
                wandb.log({"train_step/success_rate_per_100ep": final_rate}, step=self.step)
                wandb.log({"train_episode/success_rate_per_100ep": final_rate,
                           "episode": _num_train_ep}, step=self.step)
        # ───────────────────────────────────────────────────────────────────────

        # Final checkpoint
        self.agent.save(model_save_dir, self.step)
        self.reward_model.save(model_save_dir, self.step)


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=False)
def main(cfg):

    exp_name = getattr(cfg, "exp_name", None)
    wandb.init(
        entity="haobaizhan2-usc",
        project="rlvlmf",
        group=exp_name,
        name=f"{exp_name}_s{cfg.seed}" if exp_name else None,
    )
    
    wandb.config.update({
        "env": cfg.env,
        "reward": cfg.reward,
        "vlm": cfg.vlm,
        "seed": cfg.seed,
        "num_train_steps": cfg.num_train_steps,
    })

    if getattr(cfg, 'use_episode', False):
        wandb.define_metric("episode")
        wandb.define_metric("train_episode/*", step_metric="episode")
        wandb.define_metric("eval_episode/*", step_metric="episode")

    workspace = Workspace(cfg)

    if cfg.mode == 'eval':
        workspace.evaluate(save_additional=cfg.save_images)
    else:
        workspace.run()

    wandb.finish()


if __name__ == '__main__':
    main()

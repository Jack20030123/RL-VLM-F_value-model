#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
import pickle as pkl
import copy  # Used to build a frozen target network for value inference

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from reward_model_score import RewardModelScore
from collections import deque
from prompt import clip_env_prompts

import utils
import hydra
from PIL import Image

from vlms.blip_infer_2 import blip2_image_text_matching
from vlms.clip_infer import clip_infer_score as clip_image_text_matching
import cv2
import wandb


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
        self.device = torch.device(cfg.device)
        self.log_success = False

        # copy prompt file into the log directory for reproducibility
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system("cp {}/prompt.py {}/".format(current_file_path, self.logger._log_dir))

        # build environment
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            self.env = utils.make_classic_control_env(cfg)
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # image sizes / resize factor for image-based reward/value
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

        # choose replay capacity based on whether we store images
        img_capacity = int(getattr(cfg, "image_replay_capacity", 5000))
        cap = int(cfg.replay_buffer_capacity) if not self.cfg.image_reward else img_capacity
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            cap,  # smaller capacity for image mode to control memory
            self.device,
            store_image=self.cfg.image_reward,
            image_size=image_height)

        # logging counters
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiate reward/value model (same class; trained from preferences)
        reward_model_class = RewardModel
        if self.reward == 'learn_from_preference':
            reward_model_class = RewardModel
        elif self.reward == 'learn_from_score':
            reward_model_class = RewardModelScore

        self.reward_model = reward_model_class(
            # original PEBBLE parameters
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

            # image-based reward/value model parameters
            image_reward=cfg.image_reward,
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
        )

        if self.cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, 1000000)

        if self.cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, 1000000)

        # switches and target network for value-difference reward
        self.use_value_diff_reward = getattr(cfg, "use_value_diff_reward", False)
        self.value_target_sync_every = int(getattr(cfg, "value_target_sync_every", 1000))
        if self.use_value_diff_reward:
            # a frozen copy to stabilize V(s) used for r_t = γ V(s_t) − V(s_{t-1})
            self.value_target = copy.deepcopy(self.reward_model)
            self.value_target.eval()
        else:
            self.value_target = None

    def evaluate(self, save_additional=False):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for episode in range(self.cfg.num_eval_episodes):
            print("evaluating episode {}".format(episode))
            images = []
            obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]

            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            ep_info = []
            rewards = []
            t_idx = 0
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
                    if self.cfg.mode != 'eval':
                        rgb_image = rgb_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            rgb_image = rgb_image[100:400, 100:400, :]
                    else:
                        rgb_image = rgb_image[::-1, :, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    rgb_image = self.env.render(mode='rgb_array')
                else:
                    rgb_image = self.env.render(mode='rgb_array')

                if 'softgym' not in self.cfg.env:
                    images.append(rgb_image)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

                t_idx += 1
                if self.cfg.mode == 'eval' and t_idx > 50:
                    break

            all_ep_infos.append(ep_info)
            if 'softgym' in self.cfg.env:
                images = self.env.video_frames

            # save gif locally
            video_frames = np.array(images)
            save_gif_path = os.path.join(
                save_gif_dir,
                'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2)))
            utils.save_numpy_as_gif(video_frames, save_gif_path)

            # log gif as wandb.Video (all eval gifs)
            try:
                # video_frames: (T, H, W, 3) -> (T, C, H, W)
                video_tensor = video_frames.transpose(0, 3, 1, 2)
                wandb.log(
                    {
                        f"eval/video_step{self.step:07d}_ep{episode:02d}":
                            wandb.Video(video_tensor, fps=12, format="gif")
                    },
                    step=self.step
                )
            except Exception as e:
                print(f"Failed to log eval video for episode {episode}: {e}")

            if save_additional:
                save_image_dir = os.path.join(self.logger._log_dir, 'eval_images')
                if not os.path.exists(save_image_dir):
                    os.makedirs(save_image_dir)
                for i, image in enumerate(images):
                    save_image_path = os.path.join(
                        save_image_dir, 'step{:07}_episode{:02}_{}.png'.format(self.step, episode, i))
                    image = Image.fromarray(image)
                    image.save(save_image_path)
                save_reward_path = os.path.join(self.logger._log_dir, "eval_reward")
                if not os.path.exists(save_reward_path):
                    os.makedirs(save_reward_path)
                with open(os.path.join(save_reward_path, "step{:07}_episode{:02}.pkl".format(self.step, episode)), "wb") as f:
                    pkl.dump(rewards, f)

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        for key, value in extra.items():
            self.logger.log('eval/' + key, value, self.step)

        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)

        self.logger.dump(self.step)

        eval_metrics = {
            "eval/episode_reward": average_episode_reward,
            "eval/true_episode_reward": average_true_episode_reward,
        }
        if self.log_success:
            eval_metrics["eval/success_rate"] = success_rate
        wandb.log(eval_metrics, step=self.step)

    def learn_reward(self, first_flag=0):
        # Collect preference labels and train the reward/value model (Signal A).
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
            # preference training loop
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

        # keep recent 10 train returns
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        reward_learning_acc = 0
        vlm_acc = 0
        eval_cnt = 0

        # previous frame buffer for value-difference reward
        prev_rgb_image = None

        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    duration = time.time() - start_time
                    self.logger.log('train/duration', duration, self.step)
                    self.logger.log('train/reward_learning_acc', reward_learning_acc, self.step)
                    self.logger.log('train/vlm_acc', vlm_acc, self.step)
                    for key, value in extra.items():
                        self.logger.log('train/' + key, value, self.step)
                    start_time = time.time()

                # periodic evaluation
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    eval_cnt += 1

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                if self.step > 0:
                    train_metrics = {
                        "train/episode": episode,
                        "train/episode_reward": episode_reward,
                        "train/true_episode_reward": true_episode_reward,
                        "train/total_feedback": self.total_feedback,
                        "train/labeled_feedback": self.labeled_feedback,
                        "train/reward_learning_acc": reward_learning_acc,
                        "train/vlm_acc": vlm_acc,
                    }
                    if self.log_success:
                        train_metrics["train/episode_success"] = episode_success
                    wandb.log(train_metrics, step=self.step)

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

                # capture an initial frame for value-difference reward
                if self.use_value_diff_reward and self.cfg.image_reward:
                    if "metaworld" in self.cfg.env:
                        prev_rgb_image = self.env.render()
                        prev_rgb_image = prev_rgb_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            prev_rgb_image = prev_rgb_image[100:400, 100:400, :]
                    elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                        prev_rgb_image = self.env.render(mode='rgb_array')
                    elif 'softgym' in self.cfg.env:
                        prev_rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
                    else:
                        prev_rgb_image = self.env.render(mode='rgb_array')

                    if 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                        prev_rgb_image = cv2.resize(prev_rgb_image, (self.image_height, self.image_width))
                else:
                    prev_rgb_image = None

            # sample an action
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # switch from unsupervised to supervised preference learning
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

                    # optional teacher thresholds
                    new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                    self.reward_model.set_teacher_thres_skip(new_margin)
                    self.reward_model.set_teacher_thres_equal(new_margin)

                    # first preference learning
                    reward_learning_acc, vlm_acc = self.learn_reward(first_flag=1)

                    # relabel replay with the updated model
                    self.reward_model.eval()
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    self.reward_model.train()

                # reset critic after unsupervised exploration
                self.agent.reset_critic()

                # warmup updates
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True)

                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # periodic preference learning and relabeling
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

                        # avoid exceeding max_feedback
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                        reward_learning_acc, vlm_acc = self.learn_reward()
                        self.reward_model.eval()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        self.reward_model.train()
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            # unsupervised exploration updates (state entropy) before preference learning kicks in
            elif self.step > self.cfg.num_seed_steps:
                if self.step % 1000 == 0:
                    print("unsupervised exploration!!")
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)

            # environment step
            try:  # handle different gym APIs
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
            ep_info.append(extra)

            # capture image if needed
            if self.cfg.vlm_label or self.reward in ['blip2_image_text_matching', 'clip_image_text_matching'] or \
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

            # ===================== reward computation =====================
            if self.reward == 'learn_from_preference' or self.reward == 'learn_from_score':
                if self.use_value_diff_reward and self.cfg.image_reward:
                    # Value-difference: r_t = gamma * V(s_t) - V(s_{t-1})
                    try:
                        gamma_v = float(self.cfg.agent.params.discount)
                    except Exception:
                        gamma_v = 0.99

                    if rgb_image is None:
                        reward_hat = 0.0
                    else:
                        curr_img = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                        curr_img = curr_img[:, ::self.resize_factor, ::self.resize_factor]
                        curr_img = curr_img.reshape(1, 3, curr_img.shape[1], curr_img.shape[2])

                        if prev_rgb_image is None:
                            reward_hat = 0.0
                        else:
                            prev_img = prev_rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                            prev_img = prev_img[:, ::self.resize_factor, ::self.resize_factor]
                            prev_img = prev_img.reshape(1, 3, prev_img.shape[1], prev_img.shape[2])

                            self.value_target.eval()
                            v_tm1 = float(self.value_target.r_hat(prev_img))
                            v_t = float(self.value_target.r_hat(curr_img))
                            reward_hat = gamma_v * v_t - v_tm1

                        # update previous-frame cache with current raw rgb
                        prev_rgb_image = rgb_image

                        # periodic target sync
                        if self.step > 0 and (self.step % self.value_target_sync_every == 0):
                            # rebuild frozen target from current reward model
                            if self.value_target is not None:
                                del self.value_target
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            self.value_target = copy.deepcopy(self.reward_model)
                            self.value_target.eval()
                else:
                    # Original path: treat reward_model output as immediate reward (for image or non-image)
                    if not self.cfg.image_reward:
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
                # scale to [-1, 1] since tanh is used in the reward/value head
                reward_hat = blip2_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat

            elif self.reward == 'clip_image_text_matching':
                query_image = rgb_image
                query_prompt = clip_env_prompts[self.cfg.env]
                reward_hat = clip_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat

            elif self.reward == 'gt_task_reward':
                reward_hat = reward

            elif self.reward == 'sparse_task_reward':
                reward_hat = extra['success']

            else:
                reward_hat = reward

            # allow infinite bootstrap
            done = float(done)
            if 'softgym' not in self.cfg.env:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            else:
                done_no_max = done

            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # add transition to preference/value training buffer (Signal A pairs are formed inside the model)
            if self.reward == 'learn_from_preference' or self.reward == 'learn_from_score':
                self.reward_model.add_data(obs, action, reward, done, img=rgb_image)

            # push transition into replay
            if self.cfg.image_reward and self.reward not in ["gt_task_reward", "sparse_task_reward"]:
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max,
                    image=rgb_image[::self.resize_factor, ::self.resize_factor, :])
            else:
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            # periodic checkpointing
            if self.step % self.cfg.save_interval == 0 and self.step > 0:
                self.agent.save(model_save_dir, self.step)
                self.reward_model.save(model_save_dir, self.step)

        self.agent.save(model_save_dir, self.step)
        self.reward_model.save(model_save_dir, self.step)


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=False)
def main(cfg):

    wandb.init(
        project="rlvlmf",
        name=getattr(cfg, "exp_name", None),
    )
    
    wandb.config.update({
        "env": cfg.env,
        "reward": cfg.reward,
        "vlm": cfg.vlm,
        "seed": cfg.seed,
        "num_train_steps": cfg.num_train_steps,
    })

    workspace = Workspace(cfg)

    if cfg.mode == 'eval':
        workspace.evaluate(save_additional=cfg.save_images)
    else:
        workspace.run()

    wandb.finish()


if __name__ == '__main__':
    main()


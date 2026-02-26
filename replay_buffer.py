import numpy as np
import torch
import utils
from scipy.signal import savgol_filter

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1, store_image=False, image_size=300,
                 smooth_relabel=False, smooth_window=21):
        self.capacity = capacity
        self.device = device
        self.smooth_relabel = smooth_relabel
        sw = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.smooth_window = max(3, sw)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window
        self.store_image = store_image
        if self.store_image:
            self.images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, image=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        if image is not None and self.store_image:
            np.copyto(self.images[self.idx], image)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 128
        total_samples = self.capacity if self.full else self.idx
        total_iter = int(total_samples / batch_size)
        if total_samples > batch_size * total_iter:
            total_iter += 1

        if self.smooth_relabel:
            # Collect all predictions first, then per-episode SG smooth
            all_preds = np.empty(total_samples, dtype=np.float32)
            for index in range(total_iter):
                start = index * batch_size
                end = min((index + 1) * batch_size, total_samples)
                if not self.store_image:
                    obses = self.obses[start:end]
                    actions = self.actions[start:end]
                    inputs = np.concatenate([obses, actions], axis=-1)
                else:
                    inputs = self.images[start:end]
                    inputs = np.transpose(inputs, (0, 3, 1, 2))
                    inputs = inputs.astype(np.float32) / 255.0
                all_preds[start:end] = predictor.r_hat_batch(inputs).flatten()

            # Per-episode SG smooth (no diff — rewards stay as absolute values)
            new_rewards = np.empty(total_samples, dtype=np.float32)
            episode_start = 0
            for t in range(total_samples):
                is_end = (self.not_dones[t, 0] < 0.5) or (t == total_samples - 1)
                if is_end:
                    ep_p = all_preds[episode_start:t + 1]
                    ep_len = len(ep_p)
                    if ep_len < 3:
                        smooth_ep_p = ep_p.copy()
                    else:
                        sw = min(self.smooth_window, ep_len if ep_len % 2 == 1 else ep_len - 1)
                        sw = max(3, sw)
                        smooth_ep_p = savgol_filter(ep_p, window_length=sw, polyorder=2)
                    new_rewards[episode_start:t + 1] = smooth_ep_p
                    episode_start = t + 1
            self.rewards[:total_samples] = new_rewards.reshape(-1, 1)
        else:
            for index in range(total_iter):
                last_index = min((index + 1) * batch_size, total_samples)
                if not self.store_image:
                    obses = self.obses[index * batch_size:last_index]
                    actions = self.actions[index * batch_size:last_index]
                    inputs = np.concatenate([obses, actions], axis=-1)
                else:
                    inputs = self.images[index * batch_size:last_index]
                    inputs = np.transpose(inputs, (0, 3, 1, 2))
                    inputs = inputs.astype(np.float32) / 255.0
                pred_reward = predictor.r_hat_batch(inputs)
                self.rewards[index * batch_size:last_index] = pred_reward

        torch.cuda.empty_cache()
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max


class ProgressDiffReplayBuffer(object):
    """
    Replay buffer for progress_diff mode.

    Online reward_hat = P(s') (same as baseline, no diff online).
    relabel_with_predictor: computes P(s') for all stored images in batches,
    then per-episode applies Savitzky-Golay smooth + np.diff.
    Only stores next_image (self.images), no curr_image needed.
    """

    def __init__(self, obs_shape, action_shape, capacity, device, window=1,
                 image_size=300, smooth_window=21, smooth_relabel=True, reward_scale=1.0):
        self.capacity = capacity
        self.device = device
        self.image_size = image_size
        self.smooth_window = smooth_window
        self.smooth_relabel = smooth_relabel
        self.reward_scale = reward_scale

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        # Only next_image needed; named `images` to match ReplayBuffer interface
        self.images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, image=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if image is not None:
            np.copyto(self.images[self.idx], image)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def relabel_with_predictor(self, predictor):
        """
        Relabel rewards using per-episode SG smooth + diff.

        Steps:
        1. Batch-compute P(s') for all stored images.
        2. Walk buffer in chronological order; detect episode ends via not_dones.
        3. Per episode: Savitzky-Golay smooth the P values, then np.diff.
           First step of each episode gets reward=0 (no prev state).
        4. Write results back to self.rewards.
        Returns None (no statistics needed by caller).
        """
        batch_size = 128
        total_samples = self.capacity if self.full else self.idx

        # Step 1: compute P(s') for all samples
        p_values = np.zeros(total_samples, dtype=np.float32)
        total_iter = int(np.ceil(total_samples / batch_size))
        for i in range(total_iter):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            imgs = self.images[start:end]
            imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
            p_values[start:end] = predictor.r_hat_batch(imgs).flatten()

        # Step 2 & 3: per-episode smooth + diff
        new_rewards = np.zeros(total_samples, dtype=np.float32)
        sw_base = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        sw_base = max(3, sw_base)

        episode_start = 0
        for t in range(total_samples):
            is_end = (self.not_dones[t, 0] < 0.5) or (t == total_samples - 1)
            if is_end:
                ep_p = p_values[episode_start:t + 1]
                ep_len = len(ep_p)
                if self.smooth_relabel:
                    if ep_len < 3:
                        smooth_ep_p = ep_p.copy()
                    else:
                        sw = min(sw_base, ep_len if ep_len % 2 == 1 else ep_len - 1)
                        sw = max(3, sw)
                        smooth_ep_p = savgol_filter(ep_p, window_length=sw, polyorder=2)
                    ep_rewards = np.concatenate([[0.0], np.diff(smooth_ep_p)])
                else:
                    # Raw diff: no smooth, just diff of raw model output
                    ep_rewards = np.concatenate([[0.0], np.diff(ep_p)])
                new_rewards[episode_start:t + 1] = ep_rewards
                episode_start = t + 1

        self.rewards[:total_samples] = (new_rewards * self.reward_scale).reshape(-1, 1)
        torch.cuda.empty_cache()
        return None

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_state_ent(self, batch_size):
        """Sample for state entropy computation."""
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[:self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)

        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max
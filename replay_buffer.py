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
        total_iter = int(np.ceil(total_samples / batch_size))

        # Iterate in chronological order to avoid splice bug at circular buffer wrap point.
        # When full: oldest data starts at self.idx; physical order 0..N-1 cuts across that boundary.
        if self.full:
            chron_idxs = np.concatenate([
                np.arange(self.idx, self.capacity),
                np.arange(0, self.idx),
            ])
        else:
            chron_idxs = np.arange(0, self.idx)

        if self.smooth_relabel:
            # Collect all predictions in chronological order, then per-episode SG smooth
            all_preds = np.empty(total_samples, dtype=np.float32)
            for index in range(total_iter):
                start = index * batch_size
                end = min((index + 1) * batch_size, total_samples)
                batch_phys = chron_idxs[start:end]
                if not self.store_image:
                    obses = self.obses[batch_phys]
                    actions = self.actions[batch_phys]
                    inputs = np.concatenate([obses, actions], axis=-1)
                else:
                    inputs = self.images[batch_phys]
                    inputs = np.transpose(inputs, (0, 3, 1, 2))
                    inputs = inputs.astype(np.float32) / 255.0
                all_preds[start:end] = predictor.r_hat_batch(inputs).flatten()

            # Per-episode SG smooth (no diff — rewards stay as absolute values)
            new_rewards = np.empty(total_samples, dtype=np.float32)
            episode_start = 0
            for t in range(total_samples):
                phys_t = chron_idxs[t]
                is_end = (self.not_dones[phys_t, 0] < 0.5) or (t == total_samples - 1)
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
            # Write back to physical positions
            self.rewards[chron_idxs] = new_rewards.reshape(-1, 1)
        else:
            for index in range(total_iter):
                start = index * batch_size
                end = min((index + 1) * batch_size, total_samples)
                batch_phys = chron_idxs[start:end]
                if not self.store_image:
                    obses = self.obses[batch_phys]
                    actions = self.actions[batch_phys]
                    inputs = np.concatenate([obses, actions], axis=-1)
                else:
                    inputs = self.images[batch_phys]
                    inputs = np.transpose(inputs, (0, 3, 1, 2))
                    inputs = inputs.astype(np.float32) / 255.0
                pred_reward = predictor.r_hat_batch(inputs)
                self.rewards[batch_phys] = pred_reward

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

    Stores current-state images s_t in self.images (one per slot).
    Stores terminal images s_L (state after last action of each episode) in
    self._terminal_images, keyed by the physical slot of that episode's last step.

    relabel_with_predictor:
      1. Batch-compute P(s_t) for all stored current-state images.
      2. Batch-compute P(s_L) for each episode's terminal image.
      3. Per episode: full_p = [P(s_0),...,P(s_{L-1}), P(s_L)],
         optionally SG-smooth, then diff → r_t = γ·P(s_{t+1}) - P(s_t).
         All steps (including first and last) are fully correct for both
         complete episodes and fragments after buffer wrap-around.
    """

    def __init__(self, obs_shape, action_shape, capacity, device, window=1,
                 image_size=300, smooth_window=21, smooth_relabel=True, reward_scale=1.0, discount=1.0):
        self.capacity = capacity
        self.device = device
        self.image_size = image_size
        self.smooth_window = smooth_window
        self.smooth_relabel = smooth_relabel
        self.reward_scale = reward_scale
        self.discount = discount

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        # Stores current-state images s_t (not next-state); named `images` to match ReplayBuffer interface
        self.images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        # _terminal_images[buf_idx] = s_L image (state after last action of the episode
        # whose last step occupies buf_idx).  Keyed by the physical slot of the done step.
        self._terminal_images = {}

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, image=None, terminal_image=None):
        # Clean up stale terminal_image if this slot is being overwritten
        if self.idx in self._terminal_images:
            del self._terminal_images[self.idx]

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if image is not None:
            np.copyto(self.images[self.idx], image)

        # Store terminal image s_L at the done step (only passed when done=True)
        if terminal_image is not None:
            self._terminal_images[self.idx] = terminal_image.copy()

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def relabel_with_predictor(self, predictor):
        """
        Relabel rewards using per-episode SG smooth + diff.

        Steps:
        1. Batch-compute P(s_t) for all stored current-state images.
        2. Batch-compute P(s_L) for each episode's terminal image from _terminal_images.
        3. Per episode: full_p = [P(s_0),...,P(s_{L-1}), P(s_L)],
           optionally SG-smooth, then diff → r_t = γ·P(s_{t+1}) - P(s_t).
        4. Write results back to self.rewards.
        Returns None.
        """
        batch_size = 128
        total_samples = self.capacity if self.full else self.idx

        # Iterate in chronological order to avoid splice bug at circular buffer wrap point.
        if self.full:
            chron_idxs = np.concatenate([
                np.arange(self.idx, self.capacity),
                np.arange(0, self.idx),
            ])
        else:
            chron_idxs = np.arange(0, self.idx)

        # Step 1: compute P(s_t) for all stored current-state images in chronological order
        p_values = np.zeros(total_samples, dtype=np.float32)
        total_iter = int(np.ceil(total_samples / batch_size))
        for i in range(total_iter):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            batch_phys = chron_idxs[start:end]
            imgs = self.images[batch_phys]
            imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
            p_values[start:end] = predictor.r_hat_batch(imgs).flatten()

        # Step 2: batch-compute P(s_L) for all episode terminal images (keyed by physical slot)
        terminal_p_values = {}
        if self._terminal_images:
            term_items = list(self._terminal_images.items())
            for i in range(0, len(term_items), batch_size):
                batch = term_items[i:i + batch_size]
                phys_idxs = [item[0] for item in batch]
                imgs_batch = np.array([item[1] for item in batch])
                imgs_batch = np.transpose(imgs_batch, (0, 3, 1, 2)).astype(np.float32) / 255.0
                p_vals = predictor.r_hat_batch(imgs_batch).flatten()
                for idx, p in zip(phys_idxs, p_vals):
                    terminal_p_values[idx] = p

        # Step 3: per-episode smooth + diff in chronological order
        new_rewards = np.zeros(total_samples, dtype=np.float32)
        sw_base = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        sw_base = max(3, sw_base)

        episode_start = 0
        for t in range(total_samples):
            phys_t = chron_idxs[t]
            is_end = (self.not_dones[phys_t, 0] < 0.5) or (t == total_samples - 1)
            if is_end:
                ep_end_phys = chron_idxs[t]
                ep_p = p_values[episode_start:t + 1]        # [P(s_0), ..., P(s_{L-1})]
                # Get P(s_L) from terminal image; fall back to P(s_{L-1}) only for partial
                # episodes forced-ended by t==total_samples-1 (step-mode mid-episode relabel)
                terminal_p = terminal_p_values.get(ep_end_phys, ep_p[-1] if len(ep_p) > 0 else 0.0)
                full_p = np.append(ep_p, terminal_p)        # [P(s_0), ..., P(s_{L-1}), P(s_L)]
                full_len = len(full_p)
                if self.smooth_relabel:
                    if full_len < 3:
                        smooth_full_p = full_p.copy()
                    else:
                        sw = min(sw_base, full_len if full_len % 2 == 1 else full_len - 1)
                        sw = max(3, sw)
                        smooth_full_p = savgol_filter(full_p, window_length=sw, polyorder=2)
                else:
                    smooth_full_p = full_p
                # diff over L+1 values → L rewards: r_t = γ·P(s_{t+1}) - P(s_t)
                ep_rewards = self.discount * smooth_full_p[1:] - smooth_full_p[:-1]
                new_rewards[episode_start:t + 1] = ep_rewards
                episode_start = t + 1

        # Write back to physical positions
        self.rewards[chron_idxs] = (new_rewards * self.reward_scale).reshape(-1, 1)
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
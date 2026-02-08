import numpy as np
import torch
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1, store_image=False, image_size=300):
        self.capacity = capacity
        self.device = device

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
        if not self.store_image:
            batch_size = 128
        else:
            batch_size = 128
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
            
            if not self.store_image:
                obses = self.obses[index*batch_size:last_index]
                actions = self.actions[index*batch_size:last_index]
                inputs = np.concatenate([obses, actions], axis=-1)
            else:
                inputs = self.images[index*batch_size:last_index]
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                inputs = inputs.astype(np.float32) / 255.0

            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward
            
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

    When use_baseline_relabel=False (default):
    - Stores both curr_image and next_image (~100GB)
    - relabel: reward = P(s_{t+1}) - P(s_t)

    When use_baseline_relabel=True:
    - Only stores next_image (~50GB, saves memory)
    - relabel: reward = P(s_{t+1})
    - Online reward_hat still uses P(s_{t+1}) - P(s_t)
    """

    def __init__(self, obs_shape, action_shape, capacity, device, window=1, image_size=300,
                 use_baseline_relabel=False):
        self.capacity = capacity
        self.device = device
        self.image_size = image_size
        self.use_baseline_relabel = use_baseline_relabel

        # Proprioceptive observations
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        # Only allocate curr_images when needed for pure progress_diff relabel
        if not use_baseline_relabel:
            self.curr_images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)
        self.next_images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max,
            curr_image=None, next_image=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if curr_image is not None and not self.use_baseline_relabel:
            np.copyto(self.curr_images[self.idx], curr_image)
        if next_image is not None:
            np.copyto(self.next_images[self.idx], next_image)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def relabel_with_predictor(self, predictor):
        """
        Relabel rewards.

        If use_baseline_relabel=False (default):
            reward = P(s_{t+1}) - P(s_t)  (progress difference)
        If use_baseline_relabel=True:
            reward = P(s_{t+1})  (baseline style, using next_image)
        """
        batch_size = 128
        total_samples = self.capacity if self.full else self.idx
        total_iter = int(total_samples / batch_size)

        if total_samples > batch_size * total_iter:
            total_iter += 1

        for index in range(total_iter):
            start_idx = index * batch_size
            last_index = min((index + 1) * batch_size, total_samples)

            # Prepare next_images (s_{t+1}) - always needed
            next_imgs = self.next_images[start_idx:last_index]
            next_imgs = np.transpose(next_imgs, (0, 3, 1, 2))  # HWC -> CHW
            next_imgs = next_imgs.astype(np.float32) / 255.0

            if self.use_baseline_relabel:
                # Baseline relabel: reward = P(s_{t+1})
                pred_reward = predictor.r_hat_batch(next_imgs)
            else:
                # Progress diff relabel: reward = P(s_{t+1}) - P(s_t)
                # Prepare curr_images (s_t)
                curr_imgs = self.curr_images[start_idx:last_index]
                curr_imgs = np.transpose(curr_imgs, (0, 3, 1, 2))  # HWC -> CHW
                curr_imgs = curr_imgs.astype(np.float32) / 255.0

                # Compute P(s_t) and P(s_{t+1})
                p_curr = predictor.r_hat_batch(curr_imgs)  # shape: (batch, 1)
                p_next = predictor.r_hat_batch(next_imgs)  # shape: (batch, 1)

                # reward = P(s_{t+1}) - P(s_t)
                pred_reward = p_next - p_curr

            self.rewards[start_idx:last_index] = pred_reward

        torch.cuda.empty_cache()

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
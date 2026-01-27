"""
Generate expert demonstrations using trained actor from models_gt directory,
and score each timestep of the expert trajectory using reward model from models directory.
Analyze whether the learned reward model behaves more like a reward model, value model, or progress model.

Usage:
    python score_expert_trajectory.py \
        --model_dir models_gt \
        --step 500000 \
        --reward_model_dir models \
        --reward_model_step 1000000 \
        --save_images \
        --output_dir expert_trajectory_output
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_model import gen_image_net
from agent.actor import DiagGaussianActor
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import metaworld.envs.mujoco.env_dict as _env_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_metaworld_env(env_name="drawer-open-v2", seed=0, random_init=True):
    """Create Metaworld environment."""
    if env_name.startswith("metaworld_"):
        env_name = env_name[len("metaworld_"):]

    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls(render_mode='rgb_array', random_init=random_init)
    env.camera_name = env_name
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)

    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def load_actor(model_dir, step, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2):
    """Load trained actor model."""
    actor = DiagGaussianActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        log_std_bounds=[-5, 2]
    ).to(device)

    actor_path = os.path.join(model_dir, f'actor_{step}.pt')
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor file not found: {actor_path}")

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print(f"Loaded actor from {actor_path}")

    return actor


def load_reward_model(model_dir, step,
                      ensemble_size=3,
                      image_height=300,
                      image_width=300,
                      conv_kernel_sizes=[5, 3, 3, 3],
                      conv_n_channels=[16, 32, 64, 128],
                      conv_strides=[3, 2, 2, 2],
                      resnet=False):
    """Load trained reward model ensemble."""
    ensemble = []

    for member in range(ensemble_size):
        if not resnet:
            model = gen_image_net(image_height, image_width,
                                  conv_kernel_sizes, conv_n_channels,
                                  conv_strides).float().to(device)
        else:
            model = gen_image_net2().float().to(device)

        model_path = os.path.join(model_dir, f'reward_model_{step}_{member}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        ensemble.append(model)
        print(f"Loaded reward model {member} from {model_path}")

    return ensemble


def r_hat(ensemble, image):
    """
    Score a single image using the reward model ensemble.

    Args:
        ensemble: list of models
        image: numpy array, shape (H, W, 3), uint8 [0-255]

    Returns:
        mean_score: float, ensemble mean score
        se_score: float, ensemble standard error (std / sqrt(n))
        member_scores: list, score from each ensemble member
    """
    # Preprocess: HWC -> CHW, normalize to [0, 1]
    img = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = img.reshape(1, 3, img.shape[1], img.shape[2])
    img_tensor = torch.from_numpy(img).float().to(device)

    member_scores = []
    with torch.no_grad():
        for model in ensemble:
            score = model(img_tensor).detach().cpu().numpy().item()
            member_scores.append(score)

    n = len(member_scores)
    std = np.std(member_scores)
    se = std / np.sqrt(n)  # Standard error

    return np.mean(member_scores), se, member_scores


def act(actor, obs, sample=False):
    """Select action using actor."""
    obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
    with torch.no_grad():
        dist = actor(obs_tensor)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(-1.0, 1.0)
    return action.cpu().numpy()[0]


def normalize_to_01(arr):
    """Normalize array to [0, 1] range."""
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def compute_discounted_return(rewards, gamma):
    """
    Compute discounted cumulative return from each timestep to the end.

    Args:
        rewards: list or array, immediate reward at each timestep
        gamma: float, discount factor

    Returns:
        returns: array, discounted cumulative return at each timestep
                 returns[t] = r[t] + gamma*r[t+1] + gamma^2*r[t+2] + ...
    """
    rewards = np.array(rewards)
    T = len(rewards)
    returns = np.zeros(T)

    # Compute backwards
    returns[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns


def generate_expert_trajectory(env, actor, max_steps=200, image_size=300, seed=0):
    """
    Generate expert trajectory using trained actor.

    Returns:
        images: list of numpy arrays, RGB image at each frame
        task_progress_list: list of float, task progress (in_place_reward) at each frame
        gt_reward_list: list of float, ground truth reward at each frame
        success: bool, whether episode succeeded
        success_step: int or None, step at which success occurred (0-indexed), None if not successful
    """
    images = []
    task_progress_list = []
    gt_reward_list = []
    success = False
    success_step = None

    # Set random seed
    np.random.seed(seed)
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        reset_result = env.reset()
    # Compatible with old and new Gym API
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    for step in range(max_steps):
        # Get RGB image - consistent with training code
        rgb_image = env.render()
        rgb_image = rgb_image[::-1, :, :]  # Flip
        # Crop and resize
        rgb_image = rgb_image[100:400, 100:400, :]
        rgb_image = cv2.resize(rgb_image, (image_size, image_size))

        images.append(rgb_image.copy())

        # Select action
        action = act(actor, obs, sample=False)

        # Execute action
        next_obs, reward, done, info = env.step(action)

        # Get task progress (in_place_reward is opening_reward, range 0-1)
        task_progress = info.get('in_place_reward', 0.0)
        task_progress_list.append(task_progress)
        gt_reward_list.append(reward)

        obs = next_obs

        # Check success (but don't exit early, continue to max_steps)
        if info.get('success', False) and not success:
            print(f"Episode succeeded at step {step + 1}")
            success = True
            success_step = step

        if done:
            break

    if not success:
        success = info.get('success', False)
    print(f"Episode ended after {len(images)} steps (success={success})")
    return images, task_progress_list, gt_reward_list, success, success_step


def main():
    parser = argparse.ArgumentParser(description='Score expert trajectory with trained reward model')

    # Model parameters
    parser.add_argument('--model_dir', type=str, default='models_gt',
                        help='Path to the directory containing actor model files')
    parser.add_argument('--reward_model_dir', type=str, default='models',
                        help='Path to the directory containing reward model files (default: same as model_dir)')
    parser.add_argument('--step', type=int, default=500000,
                        help='Training step of the actor to load')
    parser.add_argument('--reward_model_step', type=int, default=1000000,
                        help='Training step of the reward model to load')
    parser.add_argument('--actor_step', type=int, default=None,
                        help='Training step of the actor (if different from --step)')
    parser.add_argument('--ensemble_size', type=int, default=3,
                        help='Number of ensemble members')

    # Environment parameters
    parser.add_argument('--env', type=str, default='metaworld_drawer-open-v2',
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=300,
                        help='Image size (height and width)')

    # Output parameters
    parser.add_argument('--save_images', action='store_true',
                        help='Save trajectory images (first 30 frames)')
    parser.add_argument('--output_dir', type=str, default='expert_trajectory_output',
                        help='Output directory for saved images and plots')

    args = parser.parse_args()

    # If actor_step not specified, use step
    actor_step = args.actor_step if args.actor_step is not None else args.step
    # Reward model directory
    reward_model_dir = args.reward_model_dir if args.reward_model_dir else args.model_dir

    print(f"\n{'='*50}")
    print(f"Configuration:")
    print(f"  Actor model dir:    {args.model_dir}")
    print(f"  Actor step:         {actor_step}")
    print(f"  Reward model dir:   {reward_model_dir}")
    print(f"  Reward model step:  {args.reward_model_step}")
    print(f"  Environment:        {args.env}")
    print(f"{'='*50}")

    # Create environment
    print(f"\n=== Creating Environment: {args.env} ===")
    env = make_metaworld_env(args.env, seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Load actor
    print(f"\n=== Loading Actor (step={actor_step}) ===")
    actor = load_actor(args.model_dir, actor_step, obs_dim, action_dim)

    # Load reward model
    print(f"\n=== Loading Reward Model (step={args.reward_model_step}) ===")
    reward_ensemble = load_reward_model(
        model_dir=reward_model_dir,
        step=args.reward_model_step,
        ensemble_size=args.ensemble_size,
        image_height=args.image_size,
        image_width=args.image_size
    )

    # Generate expert trajectory
    print(f"\n=== Generating Expert Trajectory ===")
    images, task_progress_list, gt_reward_list, success, success_step = generate_expert_trajectory(
        env, actor, max_steps=args.max_steps, image_size=args.image_size, seed=args.seed
    )
    print(f"Generated {len(images)} frames, success={success}, success_step={success_step}")

    # Score each frame
    print(f"\n=== Scoring Each Frame ===")
    reward_hats = []
    reward_ses = []  # Standard errors
    all_member_scores = []  # Scores from each member
    for i, img in enumerate(images):
        mean_score, se_score, member_scores = r_hat(reward_ensemble, img)
        reward_hats.append(mean_score)
        reward_ses.append(se_score)
        all_member_scores.append(member_scores)

    reward_hats = np.array(reward_hats)
    reward_ses = np.array(reward_ses)
    all_member_scores = np.array(all_member_scores)  # shape: (num_frames, 3)
    gt_rewards = np.array(gt_reward_list)
    task_progress = np.array(task_progress_list)

    # Compute discounted returns with different discount factors (Value Model ground truth)
    gammas = [0.75, 0.90, 0.99]
    discounted_returns = {}
    for gamma in gammas:
        discounted_returns[gamma] = compute_discounted_return(gt_rewards, gamma)

    # Normalized versions
    reward_hats_norm = normalize_to_01(reward_hats)
    gt_rewards_norm = normalize_to_01(gt_rewards)
    task_progress_norm = normalize_to_01(task_progress)
    discounted_returns_norm = {}
    for gamma in gammas:
        discounted_returns_norm[gamma] = normalize_to_01(discounted_returns[gamma])

    # Print results
    print(f"\n{'Step':<6} {'R_hat':<10} {'GT_Rew':<10} {'Progress':<10} {'Ret_0.99':<12}")
    print("-" * 60)

    for i in range(len(reward_hats)):
        print(f"{i:<6} {reward_hats[i]:<10.4f} {gt_rewards[i]:<10.4f} {task_progress[i]:<10.4f} {discounted_returns[0.99][i]:<12.4f}")

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Num frames:             {len(reward_hats)}")
    print(f"Reward_hat range:       [{np.min(reward_hats):.6f}, {np.max(reward_hats):.6f}]")
    print(f"GT reward range:        [{np.min(gt_rewards):.6f}, {np.max(gt_rewards):.6f}]")
    print(f"Task progress range:    [{np.min(task_progress):.6f}, {np.max(task_progress):.6f}]")
    for gamma in gammas:
        print(f"Return (gamma={gamma}) range:  [{np.min(discounted_returns[gamma]):.6f}, {np.max(discounted_returns[gamma]):.6f}]")
    print(f"Episode success:        {success}")
    print(f"Success step:           {success_step}")

    # Pre-success data (for correlation analysis)
    if success_step is not None:
        # Include success_step (the frame where success occurred), so 0 to success_step (inclusive)
        pre_success_end = success_step + 1
        reward_hats_pre = reward_hats[:pre_success_end]
        gt_rewards_pre = gt_rewards[:pre_success_end]
        task_progress_pre = task_progress[:pre_success_end]
        discounted_returns_pre = {gamma: discounted_returns[gamma][:pre_success_end] for gamma in gammas}
    else:
        # Not successful, use all data
        reward_hats_pre = reward_hats
        gt_rewards_pre = gt_rewards
        task_progress_pre = task_progress
        discounted_returns_pre = discounted_returns

    # ==================== Pre-success correlation analysis (main focus) ====================
    print(f"\n=== Correlation Analysis: PRE-SUCCESS (step 0-{success_step}, n={len(reward_hats_pre)}) ===")
    print(f"Correlation between reward_hat and ground truth (pre-success):\n")

    # Reward Model: reward_hat vs gt_reward
    pearson_r, pearson_p = pearsonr(reward_hats_pre, gt_rewards_pre)
    spearman_r, spearman_p = spearmanr(reward_hats_pre, gt_rewards_pre)
    print(f"  vs GT Reward (Reward Model):    Pearson={pearson_r:.4f} (p={pearson_p:.2e}), Spearman={spearman_r:.4f}")

    # Progress Model: reward_hat vs task_progress
    pearson_r, pearson_p = pearsonr(reward_hats_pre, task_progress_pre)
    spearman_r, spearman_p = spearmanr(reward_hats_pre, task_progress_pre)
    print(f"  vs Task Progress (Progress):    Pearson={pearson_r:.4f} (p={pearson_p:.2e}), Spearman={spearman_r:.4f}")

    # Value Model: reward_hat vs discounted_return (for each gamma)
    for gamma in gammas:
        pearson_r, pearson_p = pearsonr(reward_hats_pre, discounted_returns_pre[gamma])
        spearman_r, spearman_p = spearmanr(reward_hats_pre, discounted_returns_pre[gamma])
        print(f"  vs Return gamma={gamma} (Value):      Pearson={pearson_r:.4f} (p={pearson_p:.2e}), Spearman={spearman_r:.4f}")

    # ==================== All steps correlation analysis (reference) ====================
    print(f"\n=== Correlation Analysis: ALL STEPS (step 0-{len(reward_hats)-1}, n={len(reward_hats)}) ===")
    print(f"Correlation between reward_hat and ground truth (all steps):\n")

    # Reward Model: reward_hat vs gt_reward
    pearson_r, pearson_p = pearsonr(reward_hats, gt_rewards)
    spearman_r, spearman_p = spearmanr(reward_hats, gt_rewards)
    print(f"  vs GT Reward (Reward Model):    Pearson={pearson_r:.4f} (p={pearson_p:.2e}), Spearman={spearman_r:.4f}")

    # Progress Model: reward_hat vs task_progress
    pearson_r, pearson_p = pearsonr(reward_hats, task_progress)
    spearman_r, spearman_p = spearmanr(reward_hats, task_progress)
    print(f"  vs Task Progress (Progress):    Pearson={pearson_r:.4f} (p={pearson_p:.2e}), Spearman={spearman_r:.4f}")

    # Value Model: reward_hat vs discounted_return (for each gamma)
    for gamma in gammas:
        pearson_r, pearson_p = pearsonr(reward_hats, discounted_returns[gamma])
        spearman_r, spearman_p = spearmanr(reward_hats, discounted_returns[gamma])
        print(f"  vs Return gamma={gamma} (Value):      Pearson={pearson_r:.4f} (p={pearson_p:.2e}), Spearman={spearman_r:.4f}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save first 30 frames
    if args.save_images:
        print(f"\n=== Saving First 30 Images to {args.output_dir} ===")
        num_to_save = min(30, len(images))
        for i in range(num_to_save):
            img_path = os.path.join(args.output_dir, f'frame_{i:04d}.png')
            Image.fromarray(images[i]).save(img_path)
        print(f"Saved {num_to_save} images")

    # Save scores to file (including all raw and normalized values)
    scores_path = os.path.join(args.output_dir, 'scores.csv')
    with open(scores_path, 'w') as f:
        # Write header
        header = ["step",
                  "reward_hat", "reward_hat_norm",
                  "gt_reward", "gt_reward_norm",
                  "task_progress", "task_progress_norm"]
        for gamma in gammas:
            header.append(f"return_{gamma}")
            header.append(f"return_{gamma}_norm")
        # Add ensemble member scores
        for m in range(args.ensemble_size):
            header.append(f"model_{m}")
        header.append("reward_hat_se")
        f.write(",".join(header) + "\n")

        # Write data
        for i in range(len(reward_hats)):
            row = [
                str(i),
                f"{reward_hats[i]:.6f}", f"{reward_hats_norm[i]:.6f}",
                f"{gt_rewards[i]:.6f}", f"{gt_rewards_norm[i]:.6f}",
                f"{task_progress[i]:.6f}", f"{task_progress_norm[i]:.6f}"
            ]
            for gamma in gammas:
                row.append(f"{discounted_returns[gamma][i]:.6f}")
                row.append(f"{discounted_returns_norm[gamma][i]:.6f}")
            for m in range(args.ensemble_size):
                row.append(f"{all_member_scores[i][m]:.6f}")
            row.append(f"{reward_ses[i]:.6f}")
            f.write(",".join(row) + "\n")
    print(f"Saved scores to {scores_path}")

    # Save correlation analysis results to file
    corr_path = os.path.join(args.output_dir, 'correlation_analysis.txt')
    with open(corr_path, 'w') as f:
        f.write(f"Correlation Analysis for {args.env}\n")
        f.write(f"Actor step: {actor_step}, Reward model step: {args.reward_model_step}\n")
        f.write(f"Episode success: {success}\n")
        f.write(f"Success step: {success_step}\n")
        f.write(f"Num frames: {len(reward_hats)}\n\n")

        # ==================== PRE-SUCCESS correlation analysis ====================
        f.write("=" * 60 + "\n")
        f.write(f"PRE-SUCCESS Correlations (step 0-{success_step}, n={len(reward_hats_pre)}):\n")
        f.write("=" * 60 + "\n\n")

        pearson_r, pearson_p = pearsonr(reward_hats_pre, gt_rewards_pre)
        spearman_r, spearman_p = spearmanr(reward_hats_pre, gt_rewards_pre)
        f.write(f"vs GT Reward (Reward Model):\n")
        f.write(f"  Pearson:  {pearson_r:.6f} (p={pearson_p:.2e})\n")
        f.write(f"  Spearman: {spearman_r:.6f} (p={spearman_p:.2e})\n\n")

        pearson_r, pearson_p = pearsonr(reward_hats_pre, task_progress_pre)
        spearman_r, spearman_p = spearmanr(reward_hats_pre, task_progress_pre)
        f.write(f"vs Task Progress (Progress Model):\n")
        f.write(f"  Pearson:  {pearson_r:.6f} (p={pearson_p:.2e})\n")
        f.write(f"  Spearman: {spearman_r:.6f} (p={spearman_p:.2e})\n\n")

        for gamma in gammas:
            pearson_r, pearson_p = pearsonr(reward_hats_pre, discounted_returns_pre[gamma])
            spearman_r, spearman_p = spearmanr(reward_hats_pre, discounted_returns_pre[gamma])
            f.write(f"vs Return gamma={gamma} (Value Model):\n")
            f.write(f"  Pearson:  {pearson_r:.6f} (p={pearson_p:.2e})\n")
            f.write(f"  Spearman: {spearman_r:.6f} (p={spearman_p:.2e})\n\n")

        # ==================== ALL STEPS correlation analysis ====================
        f.write("=" * 60 + "\n")
        f.write(f"ALL STEPS Correlations (step 0-{len(reward_hats)-1}, n={len(reward_hats)}):\n")
        f.write("=" * 60 + "\n\n")

        pearson_r, pearson_p = pearsonr(reward_hats, gt_rewards)
        spearman_r, spearman_p = spearmanr(reward_hats, gt_rewards)
        f.write(f"vs GT Reward (Reward Model):\n")
        f.write(f"  Pearson:  {pearson_r:.6f} (p={pearson_p:.2e})\n")
        f.write(f"  Spearman: {spearman_r:.6f} (p={spearman_p:.2e})\n\n")

        pearson_r, pearson_p = pearsonr(reward_hats, task_progress)
        spearman_r, spearman_p = spearmanr(reward_hats, task_progress)
        f.write(f"vs Task Progress (Progress Model):\n")
        f.write(f"  Pearson:  {pearson_r:.6f} (p={pearson_p:.2e})\n")
        f.write(f"  Spearman: {spearman_r:.6f} (p={spearman_p:.2e})\n\n")

        for gamma in gammas:
            pearson_r, pearson_p = pearsonr(reward_hats, discounted_returns[gamma])
            spearman_r, spearman_p = spearmanr(reward_hats, discounted_returns[gamma])
            f.write(f"vs Return gamma={gamma} (Value Model):\n")
            f.write(f"  Pearson:  {pearson_r:.6f} (p={pearson_p:.2e})\n")
            f.write(f"  Spearman: {spearman_r:.6f} (p={spearman_p:.2e})\n\n")

    print(f"Saved correlation analysis to {corr_path}")

    # ==================== Generate video ====================
    print(f"\n=== Generating Animation Video ===")

    # Create 2x3 figure
    fig_anim, axes_anim = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) Top-left: Environment image
    ax_img = axes_anim[0, 0]
    im = ax_img.imshow(images[0])
    ax_img.set_title('Drawer Open Task', fontsize=12)
    ax_img.axis('off')
    step_text = ax_img.text(0.02, 0.98, '', transform=ax_img.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (0,1) Top-middle: reward_hat vs GT Reward
    ax_reward = axes_anim[0, 1]
    line_rhat1, = ax_reward.plot([], [], 'b-', linewidth=2, label='reward_hat (norm)')
    line_gt, = ax_reward.plot([], [], 'r--', linewidth=2, label='GT Reward (norm)')
    ax_reward.set_xlim(0, len(reward_hats))
    ax_reward.set_ylim(-0.05, 1.05)
    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel('Normalized [0, 1]')
    ax_reward.set_title('reward_hat vs GT Reward', fontsize=12)
    ax_reward.legend(loc='lower right', fontsize=9)
    ax_reward.grid(True, alpha=0.3)
    dot_rhat1, = ax_reward.plot([], [], 'bo', markersize=6)
    dot_gt, = ax_reward.plot([], [], 'ro', markersize=6)

    # (0,2) Top-right: reward_hat vs Progress
    ax_progress = axes_anim[0, 2]
    line_rhat2, = ax_progress.plot([], [], 'b-', linewidth=2, label='reward_hat (norm)')
    line_prog, = ax_progress.plot([], [], 'g--', linewidth=2, label='Progress (norm)')
    ax_progress.set_xlim(0, len(reward_hats))
    ax_progress.set_ylim(-0.05, 1.05)
    ax_progress.set_xlabel('Step')
    ax_progress.set_ylabel('Normalized [0, 1]')
    ax_progress.set_title('reward_hat vs Task Progress', fontsize=12)
    ax_progress.legend(loc='lower right', fontsize=9)
    ax_progress.grid(True, alpha=0.3)
    dot_rhat2, = ax_progress.plot([], [], 'bo', markersize=6)
    dot_prog, = ax_progress.plot([], [], 'go', markersize=6)

    # (1,0) Bottom-left: reward_hat vs Value gamma=0.75
    ax_val75 = axes_anim[1, 0]
    line_rhat3, = ax_val75.plot([], [], 'b-', linewidth=2, label='reward_hat (norm)')
    line_val75, = ax_val75.plot([], [], 'm--', linewidth=2, label='Value gamma=0.75 (norm)')
    ax_val75.set_xlim(0, len(reward_hats))
    ax_val75.set_ylim(-0.05, 1.05)
    ax_val75.set_xlabel('Step')
    ax_val75.set_ylabel('Normalized [0, 1]')
    ax_val75.set_title('reward_hat vs Value (gamma=0.75)', fontsize=12)
    ax_val75.legend(loc='lower right', fontsize=9)
    ax_val75.grid(True, alpha=0.3)
    dot_rhat3, = ax_val75.plot([], [], 'bo', markersize=6)
    dot_val75, = ax_val75.plot([], [], 'mo', markersize=6)

    # (1,1) Bottom-middle: reward_hat vs Value gamma=0.90
    ax_val90 = axes_anim[1, 1]
    line_rhat4, = ax_val90.plot([], [], 'b-', linewidth=2, label='reward_hat (norm)')
    line_val90, = ax_val90.plot([], [], 'c--', linewidth=2, label='Value gamma=0.90 (norm)')
    ax_val90.set_xlim(0, len(reward_hats))
    ax_val90.set_ylim(-0.05, 1.05)
    ax_val90.set_xlabel('Step')
    ax_val90.set_ylabel('Normalized [0, 1]')
    ax_val90.set_title('reward_hat vs Value (gamma=0.90)', fontsize=12)
    ax_val90.legend(loc='lower right', fontsize=9)
    ax_val90.grid(True, alpha=0.3)
    dot_rhat4, = ax_val90.plot([], [], 'bo', markersize=6)
    dot_val90, = ax_val90.plot([], [], 'co', markersize=6)

    # (1,2) Bottom-right: reward_hat vs Value gamma=0.99
    ax_val99 = axes_anim[1, 2]
    line_rhat5, = ax_val99.plot([], [], 'b-', linewidth=2, label='reward_hat (norm)')
    line_val99, = ax_val99.plot([], [], 'y-', linewidth=2, label='Value gamma=0.99 (norm)')
    ax_val99.set_xlim(0, len(reward_hats))
    ax_val99.set_ylim(-0.05, 1.05)
    ax_val99.set_xlabel('Step')
    ax_val99.set_ylabel('Normalized [0, 1]')
    ax_val99.set_title('reward_hat vs Value (gamma=0.99)', fontsize=12)
    ax_val99.legend(loc='upper right', fontsize=9)
    ax_val99.grid(True, alpha=0.3)
    dot_rhat5, = ax_val99.plot([], [], 'bo', markersize=6)
    dot_val99, = ax_val99.plot([], [], 'yo', markersize=6)

    plt.suptitle(f'Expert Trajectory Analysis - {args.env}', fontsize=14)
    plt.tight_layout()

    def init():
        """Initialize animation."""
        line_rhat1.set_data([], [])
        line_gt.set_data([], [])
        line_rhat2.set_data([], [])
        line_prog.set_data([], [])
        line_rhat3.set_data([], [])
        line_val75.set_data([], [])
        line_rhat4.set_data([], [])
        line_val90.set_data([], [])
        line_rhat5.set_data([], [])
        line_val99.set_data([], [])
        dot_rhat1.set_data([], [])
        dot_gt.set_data([], [])
        dot_rhat2.set_data([], [])
        dot_prog.set_data([], [])
        dot_rhat3.set_data([], [])
        dot_val75.set_data([], [])
        dot_rhat4.set_data([], [])
        dot_val90.set_data([], [])
        dot_rhat5.set_data([], [])
        dot_val99.set_data([], [])
        step_text.set_text('')
        return (line_rhat1, line_gt, line_rhat2, line_prog,
                line_rhat3, line_val75, line_rhat4, line_val90, line_rhat5, line_val99,
                dot_rhat1, dot_gt, dot_rhat2, dot_prog,
                dot_rhat3, dot_val75, dot_rhat4, dot_val90, dot_rhat5, dot_val99,
                step_text, im)

    def animate(frame):
        """Update each frame."""
        # Update image
        im.set_array(images[frame])

        # Update step text
        status = "SUCCESS!" if success_step is not None and frame >= success_step else ""
        step_text.set_text(f'Step: {frame}/{len(images)-1} {status}')

        # Update curves (show up to current frame)
        x_data = np.arange(frame + 1)

        # GT Reward
        line_rhat1.set_data(x_data, reward_hats_norm[:frame + 1])
        line_gt.set_data(x_data, gt_rewards_norm[:frame + 1])

        # Progress
        line_rhat2.set_data(x_data, reward_hats_norm[:frame + 1])
        line_prog.set_data(x_data, task_progress_norm[:frame + 1])

        # Value gamma=0.75
        line_rhat3.set_data(x_data, reward_hats_norm[:frame + 1])
        line_val75.set_data(x_data, discounted_returns_norm[0.75][:frame + 1])

        # Value gamma=0.90
        line_rhat4.set_data(x_data, reward_hats_norm[:frame + 1])
        line_val90.set_data(x_data, discounted_returns_norm[0.90][:frame + 1])

        # Value gamma=0.99
        line_rhat5.set_data(x_data, reward_hats_norm[:frame + 1])
        line_val99.set_data(x_data, discounted_returns_norm[0.99][:frame + 1])

        # Update current point markers
        dot_rhat1.set_data([frame], [reward_hats_norm[frame]])
        dot_gt.set_data([frame], [gt_rewards_norm[frame]])
        dot_rhat2.set_data([frame], [reward_hats_norm[frame]])
        dot_prog.set_data([frame], [task_progress_norm[frame]])
        dot_rhat3.set_data([frame], [reward_hats_norm[frame]])
        dot_val75.set_data([frame], [discounted_returns_norm[0.75][frame]])
        dot_rhat4.set_data([frame], [reward_hats_norm[frame]])
        dot_val90.set_data([frame], [discounted_returns_norm[0.90][frame]])
        dot_rhat5.set_data([frame], [reward_hats_norm[frame]])
        dot_val99.set_data([frame], [discounted_returns_norm[0.99][frame]])

        return (line_rhat1, line_gt, line_rhat2, line_prog,
                line_rhat3, line_val75, line_rhat4, line_val90, line_rhat5, line_val99,
                dot_rhat1, dot_gt, dot_rhat2, dot_prog,
                dot_rhat3, dot_val75, dot_rhat4, dot_val90, dot_rhat5, dot_val99,
                step_text, im)

    # Create animation (using all frames)
    anim = FuncAnimation(fig_anim, animate, init_func=init,
                         frames=len(images), interval=50, blit=True)

    # Save as MP4 video
    video_path = os.path.join(args.output_dir, 'trajectory_analysis.mp4')
    print(f"Saving video to {video_path} ({len(images)} frames)...")
    writer = FFMpegWriter(fps=20, metadata=dict(artist='RL-VLM-F'), bitrate=2400)
    anim.save(video_path, writer=writer)
    print(f"Saved video to {video_path}")

    plt.close(fig_anim)

    return reward_hats, task_progress


if __name__ == '__main__':
    main()

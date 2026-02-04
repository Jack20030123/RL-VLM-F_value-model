"""
Generate expert demonstrations and analyze reward model correlation with GT reward and task progress.

Usage:
    python score_expert_trajectory.py \
        --model_dir models_gt \
        --step 500000 \
        --reward_model_dir models \
        --reward_model_step 1000000 \
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
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import pearsonr, spearmanr

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
                      conv_strides=[3, 2, 2, 2]):
    """Load trained reward model ensemble."""
    ensemble = []

    for member in range(ensemble_size):
        model = gen_image_net(image_height, image_width,
                              conv_kernel_sizes, conv_n_channels,
                              conv_strides).float().to(device)

        model_path = os.path.join(model_dir, f'reward_model_{step}_{member}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        ensemble.append(model)
        print(f"Loaded reward model {member} from {model_path}")

    return ensemble


def r_hat(ensemble, image):
    """Score a single image using the reward model ensemble."""
    img = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = img.reshape(1, 3, img.shape[1], img.shape[2])
    img_tensor = torch.from_numpy(img).float().to(device)

    member_scores = []
    with torch.no_grad():
        for model in ensemble:
            score = model(img_tensor).detach().cpu().numpy().item()
            member_scores.append(score)

    return np.mean(member_scores)


def act(actor, obs, sample=False):
    """Select action using actor."""
    obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
    with torch.no_grad():
        dist = actor(obs_tensor)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(-1.0, 1.0)
    return action.cpu().numpy()[0]


def generate_expert_trajectory(env, actor, max_steps=200, image_size=300, seed=0):
    """Generate expert trajectory using trained actor."""
    images = []
    task_progress_list = []
    gt_reward_list = []
    success = False
    success_step = None

    np.random.seed(seed)
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        reset_result = env.reset()

    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    for step in range(max_steps):
        rgb_image = env.render()
        rgb_image = rgb_image[::-1, :, :]
        rgb_image = rgb_image[100:400, 100:400, :]
        rgb_image = cv2.resize(rgb_image, (image_size, image_size))

        images.append(rgb_image.copy())

        action = act(actor, obs, sample=False)
        next_obs, reward, done, info = env.step(action)

        task_progress = info.get('in_place_reward', 0.0)
        task_progress_list.append(task_progress)
        gt_reward_list.append(reward)

        obs = next_obs

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

    parser.add_argument('--model_dir', type=str, default='models_gt')
    parser.add_argument('--reward_model_dir', type=str, default='models')
    parser.add_argument('--step', type=int, default=500000)
    parser.add_argument('--reward_model_step', type=int, default=1000000)
    parser.add_argument('--actor_step', type=int, default=None)
    parser.add_argument('--ensemble_size', type=int, default=3)
    parser.add_argument('--env', type=str, default='metaworld_drawer-open-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default='expert_trajectory_output')

    args = parser.parse_args()

    actor_step = args.actor_step if args.actor_step is not None else args.step
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
    for i, img in enumerate(images):
        score = r_hat(reward_ensemble, img)
        reward_hats.append(score)

    reward_hats = np.array(reward_hats)
    gt_rewards = np.array(gt_reward_list)
    task_progress = np.array(task_progress_list)

    # Compute diff form: reward_hat_diff[i] = reward_hat[i+1] - reward_hat[i]
    reward_hat_diffs = np.diff(reward_hats)  # length = len(reward_hats) - 1
    # Align gt_rewards with diffs (use [:-1] to match diff indices)
    gt_rewards_for_diff = gt_rewards[:-1]
    # Compute progress diff for comparison
    progress_diffs = np.diff(task_progress)

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Num frames:             {len(reward_hats)}")
    print(f"Reward_hat range:       [{np.min(reward_hats):.6f}, {np.max(reward_hats):.6f}]")
    print(f"Reward_hat_diff range:  [{np.min(reward_hat_diffs):.6f}, {np.max(reward_hat_diffs):.6f}]")
    print(f"GT reward range:        [{np.min(gt_rewards):.6f}, {np.max(gt_rewards):.6f}]")
    print(f"Task progress range:    [{np.min(task_progress):.6f}, {np.max(task_progress):.6f}]")
    print(f"Progress diff range:    [{np.min(progress_diffs):.6f}, {np.max(progress_diffs):.6f}]")
    print(f"Episode success:        {success}")
    print(f"Success step:           {success_step}")

    # Correlation analysis - All steps (Original form)
    print(f"\n=== Correlation Analysis: ORIGINAL reward_hat (All Steps, n={len(reward_hats)}) ===")

    pearson_gt_all, p_gt_all = pearsonr(reward_hats, gt_rewards)
    spearman_gt_all, _ = spearmanr(reward_hats, gt_rewards)
    print(f"reward_hat vs GT Reward:      Pearson={pearson_gt_all:.4f} (p={p_gt_all:.2e}), Spearman={spearman_gt_all:.4f}")

    pearson_prog_all, p_prog_all = pearsonr(reward_hats, task_progress)
    spearman_prog_all, _ = spearmanr(reward_hats, task_progress)
    print(f"reward_hat vs Task Progress:  Pearson={pearson_prog_all:.4f} (p={p_prog_all:.2e}), Spearman={spearman_prog_all:.4f}")

    # Correlation analysis - First 30 steps (Original form)
    first_n = min(30, len(reward_hats))
    print(f"\n=== Correlation Analysis: ORIGINAL reward_hat (First {first_n} Steps) ===")

    pearson_gt_first30, p_gt_first30 = pearsonr(reward_hats[:first_n], gt_rewards[:first_n])
    spearman_gt_first30, _ = spearmanr(reward_hats[:first_n], gt_rewards[:first_n])
    print(f"reward_hat vs GT Reward:      Pearson={pearson_gt_first30:.4f} (p={p_gt_first30:.2e}), Spearman={spearman_gt_first30:.4f}")

    pearson_prog_first30, p_prog_first30 = pearsonr(reward_hats[:first_n], task_progress[:first_n])
    spearman_prog_first30, _ = spearmanr(reward_hats[:first_n], task_progress[:first_n])
    print(f"reward_hat vs Task Progress:  Pearson={pearson_prog_first30:.4f} (p={p_prog_first30:.2e}), Spearman={spearman_prog_first30:.4f}")

    # Correlation analysis - All steps (Diff form)
    print(f"\n=== Correlation Analysis: DIFF reward_hat (All Steps, n={len(reward_hat_diffs)}) ===")

    pearson_gt_diff_all, p_gt_diff_all = pearsonr(reward_hat_diffs, gt_rewards_for_diff)
    spearman_gt_diff_all, _ = spearmanr(reward_hat_diffs, gt_rewards_for_diff)
    print(f"reward_hat_diff vs GT Reward:      Pearson={pearson_gt_diff_all:.4f} (p={p_gt_diff_all:.2e}), Spearman={spearman_gt_diff_all:.4f}")

    pearson_progdiff_diff_all, p_progdiff_diff_all = pearsonr(reward_hat_diffs, progress_diffs)
    spearman_progdiff_diff_all, _ = spearmanr(reward_hat_diffs, progress_diffs)
    print(f"reward_hat_diff vs Progress Diff:  Pearson={pearson_progdiff_diff_all:.4f} (p={p_progdiff_diff_all:.2e}), Spearman={spearman_progdiff_diff_all:.4f}")

    # Correlation analysis - First 30 steps (Diff form)
    first_n_diff = min(30, len(reward_hat_diffs))
    print(f"\n=== Correlation Analysis: DIFF reward_hat (First {first_n_diff} Steps) ===")

    pearson_gt_diff_first30, p_gt_diff_first30 = pearsonr(reward_hat_diffs[:first_n_diff], gt_rewards_for_diff[:first_n_diff])
    spearman_gt_diff_first30, _ = spearmanr(reward_hat_diffs[:first_n_diff], gt_rewards_for_diff[:first_n_diff])
    print(f"reward_hat_diff vs GT Reward:      Pearson={pearson_gt_diff_first30:.4f} (p={p_gt_diff_first30:.2e}), Spearman={spearman_gt_diff_first30:.4f}")

    pearson_progdiff_diff_first30, p_progdiff_diff_first30 = pearsonr(reward_hat_diffs[:first_n_diff], progress_diffs[:first_n_diff])
    spearman_progdiff_diff_first30, _ = spearmanr(reward_hat_diffs[:first_n_diff], progress_diffs[:first_n_diff])
    print(f"reward_hat_diff vs Progress Diff:  Pearson={pearson_progdiff_diff_first30:.4f} (p={p_progdiff_diff_first30:.2e}), Spearman={spearman_progdiff_diff_first30:.4f}")

    # Correlation analysis - Pre-success
    if success_step is not None:
        pre_success_end = success_step + 1
        reward_hats_pre = reward_hats[:pre_success_end]
        gt_rewards_pre = gt_rewards[:pre_success_end]
        task_progress_pre = task_progress[:pre_success_end]

        print(f"\n=== Correlation Analysis: ORIGINAL reward_hat (Pre-Success, steps 0-{success_step}, n={len(reward_hats_pre)}) ===")

        pearson_gt_pre, p_gt_pre = pearsonr(reward_hats_pre, gt_rewards_pre)
        spearman_gt_pre, _ = spearmanr(reward_hats_pre, gt_rewards_pre)
        print(f"reward_hat vs GT Reward:      Pearson={pearson_gt_pre:.4f} (p={p_gt_pre:.2e}), Spearman={spearman_gt_pre:.4f}")

        pearson_prog_pre, p_prog_pre = pearsonr(reward_hats_pre, task_progress_pre)
        spearman_prog_pre, _ = spearmanr(reward_hats_pre, task_progress_pre)
        print(f"reward_hat vs Task Progress:  Pearson={pearson_prog_pre:.4f} (p={p_prog_pre:.2e}), Spearman={spearman_prog_pre:.4f}")

        # Diff form - Pre-success
        if pre_success_end > 1:
            reward_hat_diffs_pre = reward_hat_diffs[:pre_success_end - 1]
            gt_rewards_diff_pre = gt_rewards_for_diff[:pre_success_end - 1]
            progress_diffs_pre = progress_diffs[:pre_success_end - 1]

            print(f"\n=== Correlation Analysis: DIFF reward_hat (Pre-Success, steps 0-{success_step-1}, n={len(reward_hat_diffs_pre)}) ===")

            pearson_gt_diff_pre, p_gt_diff_pre = pearsonr(reward_hat_diffs_pre, gt_rewards_diff_pre)
            spearman_gt_diff_pre, _ = spearmanr(reward_hat_diffs_pre, gt_rewards_diff_pre)
            print(f"reward_hat_diff vs GT Reward:      Pearson={pearson_gt_diff_pre:.4f} (p={p_gt_diff_pre:.2e}), Spearman={spearman_gt_diff_pre:.4f}")

            pearson_progdiff_diff_pre, p_progdiff_diff_pre = pearsonr(reward_hat_diffs_pre, progress_diffs_pre)
            spearman_progdiff_diff_pre, _ = spearmanr(reward_hat_diffs_pre, progress_diffs_pre)
            print(f"reward_hat_diff vs Progress Diff:  Pearson={pearson_progdiff_diff_pre:.4f} (p={p_progdiff_diff_pre:.2e}), Spearman={spearman_progdiff_diff_pre:.4f}")
        else:
            pearson_gt_diff_pre, p_gt_diff_pre, spearman_gt_diff_pre = None, None, None
            pearson_progdiff_diff_pre, p_progdiff_diff_pre, spearman_progdiff_diff_pre = None, None, None
    else:
        pearson_gt_pre, p_gt_pre, spearman_gt_pre = None, None, None
        pearson_prog_pre, p_prog_pre, spearman_prog_pre = None, None, None
        pearson_gt_diff_pre, p_gt_diff_pre, spearman_gt_diff_pre = None, None, None
        pearson_progdiff_diff_pre, p_progdiff_diff_pre, spearman_progdiff_diff_pre = None, None, None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save correlation results
    corr_path = os.path.join(args.output_dir, 'correlation_analysis.txt')
    with open(corr_path, 'w') as f:
        f.write(f"Correlation Analysis for {args.env}\n")
        f.write(f"Actor step: {actor_step}, Reward model step: {args.reward_model_step}\n")
        f.write(f"Episode success: {success}, Success step: {success_step}\n")
        f.write(f"Num frames: {len(reward_hats)}\n\n")

        f.write(f"{'='*60}\n")
        f.write(f"ORIGINAL reward_hat = model(s)\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"=== All Steps (n={len(reward_hats)}) ===\n")
        f.write(f"reward_hat vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_all:.6f} (p={p_gt_all:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_all:.6f}\n\n")
        f.write(f"reward_hat vs Task Progress:\n")
        f.write(f"  Pearson:  {pearson_prog_all:.6f} (p={p_prog_all:.2e})\n")
        f.write(f"  Spearman: {spearman_prog_all:.6f}\n\n")

        f.write(f"=== First {first_n} Steps ===\n")
        f.write(f"reward_hat vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_first30:.6f} (p={p_gt_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_first30:.6f}\n\n")
        f.write(f"reward_hat vs Task Progress:\n")
        f.write(f"  Pearson:  {pearson_prog_first30:.6f} (p={p_prog_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_prog_first30:.6f}\n\n")

        if success_step is not None:
            f.write(f"=== Pre-Success (steps 0-{success_step}, n={pre_success_end}) ===\n")
            f.write(f"reward_hat vs GT Reward:\n")
            f.write(f"  Pearson:  {pearson_gt_pre:.6f} (p={p_gt_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_gt_pre:.6f}\n\n")
            f.write(f"reward_hat vs Task Progress:\n")
            f.write(f"  Pearson:  {pearson_prog_pre:.6f} (p={p_prog_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_prog_pre:.6f}\n\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"DIFF reward_hat_diff = model(s') - model(s)\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"=== All Steps (n={len(reward_hat_diffs)}) ===\n")
        f.write(f"reward_hat_diff vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_diff_all:.6f} (p={p_gt_diff_all:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_diff_all:.6f}\n\n")
        f.write(f"reward_hat_diff vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_progdiff_diff_all:.6f} (p={p_progdiff_diff_all:.2e})\n")
        f.write(f"  Spearman: {spearman_progdiff_diff_all:.6f}\n\n")

        f.write(f"=== First {first_n_diff} Steps ===\n")
        f.write(f"reward_hat_diff vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_diff_first30:.6f} (p={p_gt_diff_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_diff_first30:.6f}\n\n")
        f.write(f"reward_hat_diff vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_progdiff_diff_first30:.6f} (p={p_progdiff_diff_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_progdiff_diff_first30:.6f}\n\n")

        if success_step is not None and pre_success_end > 1:
            f.write(f"=== Pre-Success (steps 0-{success_step-1}, n={len(reward_hat_diffs_pre)}) ===\n")
            f.write(f"reward_hat_diff vs GT Reward:\n")
            f.write(f"  Pearson:  {pearson_gt_diff_pre:.6f} (p={p_gt_diff_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_gt_diff_pre:.6f}\n\n")
            f.write(f"reward_hat_diff vs Progress Diff:\n")
            f.write(f"  Pearson:  {pearson_progdiff_diff_pre:.6f} (p={p_progdiff_diff_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_progdiff_diff_pre:.6f}\n")

    print(f"Saved correlation analysis to {corr_path}")

    # Function to generate video
    def generate_video(images_subset, reward_hats_subset, gt_rewards_subset, task_progress_subset,
                       video_path, env_name, success_step_local, fps=20,
                       reward_label="reward_hat", progress_label="Task Progress"):
        num_frames = len(images_subset)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: Environment image
        ax_img = axes[0, 0]
        im = ax_img.imshow(images_subset[0])
        ax_img.set_title('Environment', fontsize=12)
        ax_img.axis('off')
        step_text = ax_img.text(0.02, 0.98, '', transform=ax_img.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Top-right: reward_hat (raw values)
        ax_rhat = axes[0, 1]
        line_rhat, = ax_rhat.plot([], [], 'b-', linewidth=2)
        ax_rhat.set_xlim(0, num_frames)
        rhat_margin = max(0.1, (np.max(reward_hats_subset) - np.min(reward_hats_subset)) * 0.1)
        ax_rhat.set_ylim(np.min(reward_hats_subset) - rhat_margin, np.max(reward_hats_subset) + rhat_margin)
        ax_rhat.set_xlabel('Step')
        ax_rhat.set_ylabel(reward_label)
        ax_rhat.set_title(reward_label, fontsize=12)
        ax_rhat.grid(True, alpha=0.3)
        dot_rhat, = ax_rhat.plot([], [], 'bo', markersize=6)

        # Bottom-left: GT Reward (raw values)
        ax_gt = axes[1, 0]
        line_gt, = ax_gt.plot([], [], 'r-', linewidth=2)
        ax_gt.set_xlim(0, num_frames)
        gt_margin = max(0.1, (np.max(gt_rewards_subset) - np.min(gt_rewards_subset)) * 0.1)
        ax_gt.set_ylim(np.min(gt_rewards_subset) - gt_margin, np.max(gt_rewards_subset) + gt_margin)
        ax_gt.set_xlabel('Step')
        ax_gt.set_ylabel('GT Reward')
        ax_gt.set_title('GT Reward', fontsize=12)
        ax_gt.grid(True, alpha=0.3)
        dot_gt, = ax_gt.plot([], [], 'ro', markersize=6)

        # Bottom-right: Task Progress (raw values)
        ax_prog = axes[1, 1]
        line_prog, = ax_prog.plot([], [], 'g-', linewidth=2)
        ax_prog.set_xlim(0, num_frames)
        prog_margin = max(0.05, (np.max(task_progress_subset) - np.min(task_progress_subset)) * 0.1)
        ax_prog.set_ylim(np.min(task_progress_subset) - prog_margin, np.max(task_progress_subset) + prog_margin)
        ax_prog.set_xlabel('Step')
        ax_prog.set_ylabel(progress_label)
        ax_prog.set_title(progress_label, fontsize=12)
        ax_prog.grid(True, alpha=0.3)
        dot_prog, = ax_prog.plot([], [], 'go', markersize=6)

        plt.suptitle(f'Expert Trajectory Analysis - {env_name}', fontsize=14)
        plt.tight_layout()

        def init():
            line_rhat.set_data([], [])
            line_gt.set_data([], [])
            line_prog.set_data([], [])
            dot_rhat.set_data([], [])
            dot_gt.set_data([], [])
            dot_prog.set_data([], [])
            step_text.set_text('')
            return line_rhat, line_gt, line_prog, dot_rhat, dot_gt, dot_prog, step_text, im

        def animate(frame):
            im.set_array(images_subset[frame])

            status = "SUCCESS!" if success_step_local is not None and frame >= success_step_local else ""
            step_text.set_text(f'Step: {frame}/{num_frames-1} {status}')

            x_data = np.arange(frame + 1)

            line_rhat.set_data(x_data, reward_hats_subset[:frame + 1])
            line_gt.set_data(x_data, gt_rewards_subset[:frame + 1])
            line_prog.set_data(x_data, task_progress_subset[:frame + 1])

            dot_rhat.set_data([frame], [reward_hats_subset[frame]])
            dot_gt.set_data([frame], [gt_rewards_subset[frame]])
            dot_prog.set_data([frame], [task_progress_subset[frame]])

            return line_rhat, line_gt, line_prog, dot_rhat, dot_gt, dot_prog, step_text, im

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=num_frames, interval=50, blit=True)

        print(f"Saving video to {video_path} ({num_frames} frames, fps={fps})...")
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='RL-VLM-F'), bitrate=2400)
        anim.save(video_path, writer=writer)
        print(f"Saved video to {video_path}")

        plt.close(fig)

    # Generate full video (Original)
    print(f"\n=== Generating Full Animation Video (Original) ===")
    video_path_full = os.path.join(args.output_dir, 'trajectory_analysis.mp4')
    generate_video(images, reward_hats, gt_rewards, task_progress,
                   video_path_full, args.env, success_step)

    # Generate first 30 steps video (10x slower, Original)
    print(f"\n=== Generating First 30 Steps Video (10x slower, Original) ===")
    num_frames_short = min(30, len(images))
    success_step_short = success_step if success_step is not None and success_step < num_frames_short else None
    video_path_short = os.path.join(args.output_dir, 'trajectory_analysis_first30.mp4')
    generate_video(images[:num_frames_short], reward_hats[:num_frames_short],
                   gt_rewards[:num_frames_short], task_progress[:num_frames_short],
                   video_path_short, args.env, success_step_short, fps=2)

    # Generate pre-success video (Original, 10x slower)
    if success_step is not None and success_step > 0:
        print(f"\n=== Generating Pre-Success Video (10x slower, Original) ===")
        video_path_presuccess = os.path.join(args.output_dir, 'trajectory_analysis_presuccess.mp4')
        generate_video(images[:pre_success_end], reward_hats[:pre_success_end],
                       gt_rewards[:pre_success_end], task_progress[:pre_success_end],
                       video_path_presuccess, args.env + " (Pre-Success)", success_step, fps=2)

    # Generate full video (Diff form)
    # For diff, we use images[:-1] since diff has one less element
    print(f"\n=== Generating Full Animation Video (Diff) ===")
    video_path_full_diff = os.path.join(args.output_dir, 'trajectory_analysis_diff.mp4')
    success_step_diff = success_step - 1 if success_step is not None and success_step > 0 else None
    generate_video(images[:-1], reward_hat_diffs, gt_rewards_for_diff, progress_diffs,
                   video_path_full_diff, args.env + " (DIFF)", success_step_diff,
                   reward_label="reward_hat_diff", progress_label="Progress Diff")

    # Generate first 30 steps video (10x slower, Diff)
    print(f"\n=== Generating First 30 Steps Video (10x slower, Diff) ===")
    num_frames_short_diff = min(30, len(reward_hat_diffs))
    success_step_short_diff = success_step_diff if success_step_diff is not None and success_step_diff < num_frames_short_diff else None
    video_path_short_diff = os.path.join(args.output_dir, 'trajectory_analysis_first30_diff.mp4')
    generate_video(images[:num_frames_short_diff], reward_hat_diffs[:num_frames_short_diff],
                   gt_rewards_for_diff[:num_frames_short_diff], progress_diffs[:num_frames_short_diff],
                   video_path_short_diff, args.env + " (DIFF)", success_step_short_diff, fps=2,
                   reward_label="reward_hat_diff", progress_label="Progress Diff")

    # Generate pre-success video (Diff form, 10x slower)
    if success_step is not None and success_step > 1:
        print(f"\n=== Generating Pre-Success Video (10x slower, Diff) ===")
        pre_success_end_diff = success_step  # For diff, one less than original
        video_path_presuccess_diff = os.path.join(args.output_dir, 'trajectory_analysis_presuccess_diff.mp4')
        generate_video(images[:pre_success_end_diff], reward_hat_diffs[:pre_success_end_diff],
                       gt_rewards_for_diff[:pre_success_end_diff], progress_diffs[:pre_success_end_diff],
                       video_path_presuccess_diff, args.env + " (DIFF, Pre-Success)", success_step_diff, fps=2,
                       reward_label="reward_hat_diff", progress_label="Progress Diff")

    return reward_hats, reward_hat_diffs, gt_rewards, task_progress, progress_diffs


if __name__ == '__main__':
    main()

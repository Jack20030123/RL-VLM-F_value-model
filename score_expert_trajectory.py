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
import importlib.metadata as importlib_metadata
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if not hasattr(importlib_metadata, "packages_distributions"):
    try:
        import importlib_metadata as backport_importlib_metadata
    except ImportError:
        backport_importlib_metadata = None

    if backport_importlib_metadata and hasattr(
        backport_importlib_metadata, "packages_distributions"
    ):
        importlib_metadata.packages_distributions = (
            backport_importlib_metadata.packages_distributions
        )
    else:
        importlib_metadata.packages_distributions = lambda: {}

from progress_diff_utils import compute_progress_diff_rewards, get_progress_diff_reward_scale
from reward_model import gen_image_net
from agent.actor import DiagGaussianActor
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import metaworld.envs.mujoco.env_dict as _env_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_metaworld_env(env_name="drawer-open-v2", seed=0, random_init=False):
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


def _infer_actor_architecture(actor_state_dict, obs_dim, action_dim):
    """Infer actor MLP width/depth from a saved state dict."""
    linear_layers = []
    for key, value in actor_state_dict.items():
        if not (key.startswith("trunk.") and key.endswith(".weight")):
            continue
        parts = key.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        linear_layers.append((int(parts[1]), tuple(value.shape)))

    if not linear_layers:
        raise RuntimeError("Could not infer actor architecture from checkpoint.")

    linear_layers.sort()
    hidden_dim = linear_layers[0][1][0]
    hidden_depth = len(linear_layers) - 1

    input_dim = linear_layers[0][1][1]
    output_dim = linear_layers[-1][1][0]
    expected_output_dim = 2 * action_dim
    if input_dim != obs_dim or output_dim != expected_output_dim:
        raise RuntimeError(
            "Actor checkpoint shape does not match environment dims: "
            f"expected input={obs_dim}, output={expected_output_dim}, "
            f"got input={input_dim}, output={output_dim}."
        )

    return hidden_dim, hidden_depth


def load_actor(model_dir, step, obs_dim, action_dim):
    """Load trained actor model."""
    actor_path = os.path.join(model_dir, f'actor_{step}.pt')
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor file not found: {actor_path}")

    actor_state_dict = torch.load(actor_path, map_location=device)
    hidden_dim, hidden_depth = _infer_actor_architecture(
        actor_state_dict, obs_dim, action_dim
    )

    actor = DiagGaussianActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        log_std_bounds=[-5, 2]
    ).to(device)

    actor.load_state_dict(actor_state_dict)
    actor.eval()
    print(
        f"Loaded actor from {actor_path} "
        f"(hidden_dim={hidden_dim}, hidden_depth={hidden_depth})"
    )

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
    parser.add_argument('--metaworld_random_init', action='store_true',
                        help='Use randomized MetaWorld initialization; default matches training config metaworld_random_init=false.')
    parser.add_argument('--smooth_window', type=int, default=21,
                        help='Window length for Savitzky-Golay smoothing (must be odd, >= 3)')
    parser.add_argument('--progress_diff_discount', type=float, default=1.0,
                        help='Discount factor used in diff reward: gamma * P(s_{t+1}) - P(s_t)')
    parser.add_argument('--progress_diff_reward_scale', type=float, default=1.0,
                        help='Base multiplier applied to diff rewards before optional 1 / (1 - gamma) scaling')
    parser.add_argument('--progress_diff_scale_by_inv_one_minus_gamma', action='store_true',
                        help='Multiply diff rewards by 1 / (1 - progress_diff_discount)')

    args = parser.parse_args()

    actor_step = args.actor_step if args.actor_step is not None else args.step
    reward_model_dir = args.reward_model_dir if args.reward_model_dir else args.model_dir
    progress_diff_effective_reward_scale, progress_diff_inv_one_minus_gamma_scale = get_progress_diff_reward_scale(
        reward_scale=args.progress_diff_reward_scale,
        discount=args.progress_diff_discount,
        scale_by_inv_one_minus_gamma=args.progress_diff_scale_by_inv_one_minus_gamma,
    )

    print(f"\n{'='*50}")
    print(f"Configuration:")
    print(f"  Actor model dir:    {args.model_dir}")
    print(f"  Actor step:         {actor_step}")
    print(f"  Reward model dir:   {reward_model_dir}")
    print(f"  Reward model step:  {args.reward_model_step}")
    print(f"  Environment:        {args.env}")
    print(f"  Diff gamma:         {args.progress_diff_discount}")
    print(f"  Diff base scale:    {args.progress_diff_reward_scale}")
    print(f"  Diff inv(1-gamma):  {progress_diff_inv_one_minus_gamma_scale}")
    print(f"  Diff eff. scale:    {progress_diff_effective_reward_scale}")
    print(f"{'='*50}")

    # Create environment
    print(f"\n=== Creating Environment: {args.env} ===")
    env = make_metaworld_env(args.env, seed=args.seed, random_init=args.metaworld_random_init)

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

    # Pure one-step reward model difference: P(s_{t+1}) - P(s_t)
    reward_hat_diffs = np.diff(reward_hats)
    # Align gt_rewards with diffs (use [:-1] to match diff indices)
    gt_rewards_for_diff = gt_rewards[:-1]
    # Compute progress diff for comparison
    progress_diffs = np.diff(task_progress)

    # Smoothed reward_hat via Savitzky-Golay filter
    sw = args.smooth_window
    if sw % 2 == 0:
        sw += 1  # must be odd
    sw = max(3, min(sw, len(reward_hats) if len(reward_hats) % 2 == 1 else len(reward_hats) - 1))
    smooth_reward_hats = savgol_filter(reward_hats, window_length=sw, polyorder=2)
    print(f"Applied Savitzky-Golay smoothing: window={sw}, polyorder=2")
    # Pure one-step difference of smoothed values
    smooth_reward_hat_diffs = np.diff(smooth_reward_hats)
    # Padded version (prepend 0) for video display: frame t shows smooth_P(t)-smooth_P(t-1)
    smooth_reward_hat_diffs_padded = np.concatenate([[0.0], smooth_reward_hat_diffs])

    # 100*(0.99*P(s')-P(s))
    reward_hat_diffs_099 = compute_progress_diff_rewards(
        reward_hats, discount=0.99, reward_scale=100, scale_by_inv_one_minus_gamma=False)
    reward_hat_diffs_099_padded = np.concatenate([[0.0], reward_hat_diffs_099])

    # 1000*(0.999*P(s')-P(s))
    reward_hat_diffs_0999 = compute_progress_diff_rewards(
        reward_hats, discount=0.999, reward_scale=1000, scale_by_inv_one_minus_gamma=False)
    reward_hat_diffs_0999_padded = np.concatenate([[0.0], reward_hat_diffs_0999])

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Num frames:                  {len(reward_hats)}")
    print(f"Reward_hat range:            [{np.min(reward_hats):.6f}, {np.max(reward_hats):.6f}]")
    print(f"Smooth_reward_hat range:     [{np.min(smooth_reward_hats):.6f}, {np.max(smooth_reward_hats):.6f}]")
    print(f"Reward_hat_diff range:       [{np.min(reward_hat_diffs):.6f}, {np.max(reward_hat_diffs):.6f}]")
    print(f"Smooth_reward_hat_diff range:[{np.min(smooth_reward_hat_diffs):.6f}, {np.max(smooth_reward_hat_diffs):.6f}]")
    print(f"GT reward range:             [{np.min(gt_rewards):.6f}, {np.max(gt_rewards):.6f}]")
    print(f"Task progress range:         [{np.min(task_progress):.6f}, {np.max(task_progress):.6f}]")
    print(f"Progress diff range:         [{np.min(progress_diffs):.6f}, {np.max(progress_diffs):.6f}]")
    print(f"Episode success:             {success}")
    print(f"Success step:                {success_step}")

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

    # Correlation analysis - All steps (Smooth form)
    print(f"\n=== Correlation Analysis: SMOOTH reward_hat (All Steps, n={len(smooth_reward_hats)}) ===")

    pearson_gt_smooth_all, p_gt_smooth_all = pearsonr(smooth_reward_hats, gt_rewards)
    spearman_gt_smooth_all, _ = spearmanr(smooth_reward_hats, gt_rewards)
    print(f"smooth_reward_hat vs GT Reward:      Pearson={pearson_gt_smooth_all:.4f} (p={p_gt_smooth_all:.2e}), Spearman={spearman_gt_smooth_all:.4f}")

    pearson_prog_smooth_all, p_prog_smooth_all = pearsonr(smooth_reward_hats, task_progress)
    spearman_prog_smooth_all, _ = spearmanr(smooth_reward_hats, task_progress)
    print(f"smooth_reward_hat vs Task Progress:  Pearson={pearson_prog_smooth_all:.4f} (p={p_prog_smooth_all:.2e}), Spearman={spearman_prog_smooth_all:.4f}")

    print(f"\n=== Correlation Analysis: SMOOTH DIFF (All Steps, n={len(smooth_reward_hat_diffs)}) ===")

    pearson_gt_sdiff_all, p_gt_sdiff_all = pearsonr(smooth_reward_hat_diffs, gt_rewards_for_diff)
    spearman_gt_sdiff_all, _ = spearmanr(smooth_reward_hat_diffs, gt_rewards_for_diff)
    print(f"smooth_diff vs GT Reward:      Pearson={pearson_gt_sdiff_all:.4f} (p={p_gt_sdiff_all:.2e}), Spearman={spearman_gt_sdiff_all:.4f}")

    pearson_pdiff_sdiff_all, p_pdiff_sdiff_all = pearsonr(smooth_reward_hat_diffs, progress_diffs)
    spearman_pdiff_sdiff_all, _ = spearmanr(smooth_reward_hat_diffs, progress_diffs)
    print(f"smooth_diff vs Progress Diff:  Pearson={pearson_pdiff_sdiff_all:.4f} (p={p_pdiff_sdiff_all:.2e}), Spearman={spearman_pdiff_sdiff_all:.4f}")

    # Correlation analysis - First 30 steps (Smooth form)
    print(f"\n=== Correlation Analysis: SMOOTH reward_hat (First {first_n} Steps) ===")

    pearson_gt_smooth_first30, p_gt_smooth_first30 = pearsonr(smooth_reward_hats[:first_n], gt_rewards[:first_n])
    spearman_gt_smooth_first30, _ = spearmanr(smooth_reward_hats[:first_n], gt_rewards[:first_n])
    print(f"smooth_reward_hat vs GT Reward:      Pearson={pearson_gt_smooth_first30:.4f} (p={p_gt_smooth_first30:.2e}), Spearman={spearman_gt_smooth_first30:.4f}")

    pearson_prog_smooth_first30, p_prog_smooth_first30 = pearsonr(smooth_reward_hats[:first_n], task_progress[:first_n])
    spearman_prog_smooth_first30, _ = spearmanr(smooth_reward_hats[:first_n], task_progress[:first_n])
    print(f"smooth_reward_hat vs Task Progress:  Pearson={pearson_prog_smooth_first30:.4f} (p={p_prog_smooth_first30:.2e}), Spearman={spearman_prog_smooth_first30:.4f}")

    first_n_sdiff = min(30, len(smooth_reward_hat_diffs))
    print(f"\n=== Correlation Analysis: SMOOTH DIFF (First {first_n_sdiff} Steps) ===")

    pearson_gt_sdiff_first30, p_gt_sdiff_first30 = pearsonr(smooth_reward_hat_diffs[:first_n_sdiff], gt_rewards_for_diff[:first_n_sdiff])
    spearman_gt_sdiff_first30, _ = spearmanr(smooth_reward_hat_diffs[:first_n_sdiff], gt_rewards_for_diff[:first_n_sdiff])
    print(f"smooth_diff vs GT Reward:      Pearson={pearson_gt_sdiff_first30:.4f} (p={p_gt_sdiff_first30:.2e}), Spearman={spearman_gt_sdiff_first30:.4f}")

    pearson_pdiff_sdiff_first30, p_pdiff_sdiff_first30 = pearsonr(smooth_reward_hat_diffs[:first_n_sdiff], progress_diffs[:first_n_sdiff])
    spearman_pdiff_sdiff_first30, _ = spearmanr(smooth_reward_hat_diffs[:first_n_sdiff], progress_diffs[:first_n_sdiff])
    print(f"smooth_diff vs Progress Diff:  Pearson={pearson_pdiff_sdiff_first30:.4f} (p={p_pdiff_sdiff_first30:.2e}), Spearman={spearman_pdiff_sdiff_first30:.4f}")

    # Correlation analysis - All steps (DIFF099 form)
    print(f"\n=== Correlation Analysis: DIFF099 reward_hat (All Steps, n={len(reward_hat_diffs_099)}) ===")

    pearson_gt_d099_all, p_gt_d099_all = pearsonr(reward_hat_diffs_099, gt_rewards_for_diff)
    spearman_gt_d099_all, _ = spearmanr(reward_hat_diffs_099, gt_rewards_for_diff)
    print(f"diff099 vs GT Reward:      Pearson={pearson_gt_d099_all:.4f} (p={p_gt_d099_all:.2e}), Spearman={spearman_gt_d099_all:.4f}")

    pearson_pdiff_d099_all, p_pdiff_d099_all = pearsonr(reward_hat_diffs_099, progress_diffs)
    spearman_pdiff_d099_all, _ = spearmanr(reward_hat_diffs_099, progress_diffs)
    print(f"diff099 vs Progress Diff:  Pearson={pearson_pdiff_d099_all:.4f} (p={p_pdiff_d099_all:.2e}), Spearman={spearman_pdiff_d099_all:.4f}")

    # Correlation analysis - First 30 steps (DIFF099 form)
    first_n_d099 = min(30, len(reward_hat_diffs_099))
    print(f"\n=== Correlation Analysis: DIFF099 reward_hat (First {first_n_d099} Steps) ===")

    pearson_gt_d099_first30, p_gt_d099_first30 = pearsonr(reward_hat_diffs_099[:first_n_d099], gt_rewards_for_diff[:first_n_d099])
    spearman_gt_d099_first30, _ = spearmanr(reward_hat_diffs_099[:first_n_d099], gt_rewards_for_diff[:first_n_d099])
    print(f"diff099 vs GT Reward:      Pearson={pearson_gt_d099_first30:.4f} (p={p_gt_d099_first30:.2e}), Spearman={spearman_gt_d099_first30:.4f}")

    pearson_pdiff_d099_first30, p_pdiff_d099_first30 = pearsonr(reward_hat_diffs_099[:first_n_d099], progress_diffs[:first_n_d099])
    spearman_pdiff_d099_first30, _ = spearmanr(reward_hat_diffs_099[:first_n_d099], progress_diffs[:first_n_d099])
    print(f"diff099 vs Progress Diff:  Pearson={pearson_pdiff_d099_first30:.4f} (p={p_pdiff_d099_first30:.2e}), Spearman={spearman_pdiff_d099_first30:.4f}")

    # Correlation analysis - All steps (DIFF0999 form)
    print(f"\n=== Correlation Analysis: DIFF0999 reward_hat (All Steps, n={len(reward_hat_diffs_0999)}) ===")

    pearson_gt_d0999_all, p_gt_d0999_all = pearsonr(reward_hat_diffs_0999, gt_rewards_for_diff)
    spearman_gt_d0999_all, _ = spearmanr(reward_hat_diffs_0999, gt_rewards_for_diff)
    print(f"diff0999 vs GT Reward:      Pearson={pearson_gt_d0999_all:.4f} (p={p_gt_d0999_all:.2e}), Spearman={spearman_gt_d0999_all:.4f}")

    pearson_pdiff_d0999_all, p_pdiff_d0999_all = pearsonr(reward_hat_diffs_0999, progress_diffs)
    spearman_pdiff_d0999_all, _ = spearmanr(reward_hat_diffs_0999, progress_diffs)
    print(f"diff0999 vs Progress Diff:  Pearson={pearson_pdiff_d0999_all:.4f} (p={p_pdiff_d0999_all:.2e}), Spearman={spearman_pdiff_d0999_all:.4f}")

    # Correlation analysis - First 30 steps (DIFF0999 form)
    first_n_d0999 = min(30, len(reward_hat_diffs_0999))
    print(f"\n=== Correlation Analysis: DIFF0999 reward_hat (First {first_n_d0999} Steps) ===")

    pearson_gt_d0999_first30, p_gt_d0999_first30 = pearsonr(reward_hat_diffs_0999[:first_n_d0999], gt_rewards_for_diff[:first_n_d0999])
    spearman_gt_d0999_first30, _ = spearmanr(reward_hat_diffs_0999[:first_n_d0999], gt_rewards_for_diff[:first_n_d0999])
    print(f"diff0999 vs GT Reward:      Pearson={pearson_gt_d0999_first30:.4f} (p={p_gt_d0999_first30:.2e}), Spearman={spearman_gt_d0999_first30:.4f}")

    pearson_pdiff_d0999_first30, p_pdiff_d0999_first30 = pearsonr(reward_hat_diffs_0999[:first_n_d0999], progress_diffs[:first_n_d0999])
    spearman_pdiff_d0999_first30, _ = spearmanr(reward_hat_diffs_0999[:first_n_d0999], progress_diffs[:first_n_d0999])
    print(f"diff0999 vs Progress Diff:  Pearson={pearson_pdiff_d0999_first30:.4f} (p={p_pdiff_d0999_first30:.2e}), Spearman={spearman_pdiff_d0999_first30:.4f}")

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

            # Smooth diff form - Pre-success
            smooth_reward_hats_pre = smooth_reward_hats[:pre_success_end]
            smooth_reward_hat_diffs_pre = smooth_reward_hat_diffs[:pre_success_end - 1]

            print(f"\n=== Correlation Analysis: SMOOTH reward_hat (Pre-Success, steps 0-{success_step}, n={pre_success_end}) ===")

            pearson_gt_smooth_pre, p_gt_smooth_pre = pearsonr(smooth_reward_hats_pre, gt_rewards_pre)
            spearman_gt_smooth_pre, _ = spearmanr(smooth_reward_hats_pre, gt_rewards_pre)
            print(f"smooth_reward_hat vs GT Reward:      Pearson={pearson_gt_smooth_pre:.4f} (p={p_gt_smooth_pre:.2e}), Spearman={spearman_gt_smooth_pre:.4f}")

            pearson_prog_smooth_pre, p_prog_smooth_pre = pearsonr(smooth_reward_hats_pre, task_progress_pre)
            spearman_prog_smooth_pre, _ = spearmanr(smooth_reward_hats_pre, task_progress_pre)
            print(f"smooth_reward_hat vs Task Progress:  Pearson={pearson_prog_smooth_pre:.4f} (p={p_prog_smooth_pre:.2e}), Spearman={spearman_prog_smooth_pre:.4f}")

            print(f"\n=== Correlation Analysis: SMOOTH DIFF (Pre-Success, steps 0-{success_step-1}, n={len(smooth_reward_hat_diffs_pre)}) ===")

            pearson_gt_sdiff_pre, p_gt_sdiff_pre = pearsonr(smooth_reward_hat_diffs_pre, gt_rewards_diff_pre)
            spearman_gt_sdiff_pre, _ = spearmanr(smooth_reward_hat_diffs_pre, gt_rewards_diff_pre)
            print(f"smooth_diff vs GT Reward:      Pearson={pearson_gt_sdiff_pre:.4f} (p={p_gt_sdiff_pre:.2e}), Spearman={spearman_gt_sdiff_pre:.4f}")

            pearson_pdiff_sdiff_pre, p_pdiff_sdiff_pre = pearsonr(smooth_reward_hat_diffs_pre, progress_diffs_pre)
            spearman_pdiff_sdiff_pre, _ = spearmanr(smooth_reward_hat_diffs_pre, progress_diffs_pre)
            print(f"smooth_diff vs Progress Diff:  Pearson={pearson_pdiff_sdiff_pre:.4f} (p={p_pdiff_sdiff_pre:.2e}), Spearman={spearman_pdiff_sdiff_pre:.4f}")

            # DIFF099 pre-success
            reward_hat_diffs_099_pre = reward_hat_diffs_099[:pre_success_end - 1]

            print(f"\n=== Correlation Analysis: DIFF099 reward_hat (Pre-Success, steps 0-{success_step-1}, n={len(reward_hat_diffs_099_pre)}) ===")

            pearson_gt_d099_pre, p_gt_d099_pre = pearsonr(reward_hat_diffs_099_pre, gt_rewards_diff_pre)
            spearman_gt_d099_pre, _ = spearmanr(reward_hat_diffs_099_pre, gt_rewards_diff_pre)
            print(f"diff099 vs GT Reward:      Pearson={pearson_gt_d099_pre:.4f} (p={p_gt_d099_pre:.2e}), Spearman={spearman_gt_d099_pre:.4f}")

            pearson_pdiff_d099_pre, p_pdiff_d099_pre = pearsonr(reward_hat_diffs_099_pre, progress_diffs_pre)
            spearman_pdiff_d099_pre, _ = spearmanr(reward_hat_diffs_099_pre, progress_diffs_pre)
            print(f"diff099 vs Progress Diff:  Pearson={pearson_pdiff_d099_pre:.4f} (p={p_pdiff_d099_pre:.2e}), Spearman={spearman_pdiff_d099_pre:.4f}")

            # DIFF0999 pre-success
            reward_hat_diffs_0999_pre = reward_hat_diffs_0999[:pre_success_end - 1]

            print(f"\n=== Correlation Analysis: DIFF0999 reward_hat (Pre-Success, steps 0-{success_step-1}, n={len(reward_hat_diffs_0999_pre)}) ===")

            pearson_gt_d0999_pre, p_gt_d0999_pre = pearsonr(reward_hat_diffs_0999_pre, gt_rewards_diff_pre)
            spearman_gt_d0999_pre, _ = spearmanr(reward_hat_diffs_0999_pre, gt_rewards_diff_pre)
            print(f"diff0999 vs GT Reward:      Pearson={pearson_gt_d0999_pre:.4f} (p={p_gt_d0999_pre:.2e}), Spearman={spearman_gt_d0999_pre:.4f}")

            pearson_pdiff_d0999_pre, p_pdiff_d0999_pre = pearsonr(reward_hat_diffs_0999_pre, progress_diffs_pre)
            spearman_pdiff_d0999_pre, _ = spearmanr(reward_hat_diffs_0999_pre, progress_diffs_pre)
            print(f"diff0999 vs Progress Diff:  Pearson={pearson_pdiff_d0999_pre:.4f} (p={p_pdiff_d0999_pre:.2e}), Spearman={spearman_pdiff_d0999_pre:.4f}")

        else:
            pearson_gt_diff_pre, p_gt_diff_pre, spearman_gt_diff_pre = None, None, None
            pearson_progdiff_diff_pre, p_progdiff_diff_pre, spearman_progdiff_diff_pre = None, None, None
            pearson_gt_smooth_pre = pearson_prog_smooth_pre = pearson_gt_sdiff_pre = pearson_pdiff_sdiff_pre = None
            p_gt_smooth_pre = p_prog_smooth_pre = p_gt_sdiff_pre = p_pdiff_sdiff_pre = None
            spearman_gt_smooth_pre = spearman_prog_smooth_pre = spearman_gt_sdiff_pre = spearman_pdiff_sdiff_pre = None
            pearson_gt_d099_pre = p_gt_d099_pre = spearman_gt_d099_pre = None
            pearson_pdiff_d099_pre = p_pdiff_d099_pre = spearman_pdiff_d099_pre = None
            pearson_gt_d0999_pre = p_gt_d0999_pre = spearman_gt_d0999_pre = None
            pearson_pdiff_d0999_pre = p_pdiff_d0999_pre = spearman_pdiff_d0999_pre = None
    else:
        pearson_gt_pre, p_gt_pre, spearman_gt_pre = None, None, None
        pearson_prog_pre, p_prog_pre, spearman_prog_pre = None, None, None
        pearson_gt_diff_pre, p_gt_diff_pre, spearman_gt_diff_pre = None, None, None
        pearson_progdiff_diff_pre, p_progdiff_diff_pre, spearman_progdiff_diff_pre = None, None, None
        pearson_gt_smooth_pre = pearson_prog_smooth_pre = pearson_gt_sdiff_pre = pearson_pdiff_sdiff_pre = None
        p_gt_smooth_pre = p_prog_smooth_pre = p_gt_sdiff_pre = p_pdiff_sdiff_pre = None
        spearman_gt_smooth_pre = spearman_prog_smooth_pre = spearman_gt_sdiff_pre = spearman_pdiff_sdiff_pre = None
        pearson_gt_d099_pre = p_gt_d099_pre = spearman_gt_d099_pre = None
        pearson_pdiff_d099_pre = p_pdiff_d099_pre = spearman_pdiff_d099_pre = None
        pearson_gt_d0999_pre = p_gt_d0999_pre = spearman_gt_d0999_pre = None
        pearson_pdiff_d0999_pre = p_pdiff_d0999_pre = spearman_pdiff_d0999_pre = None

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
        f.write(f"DIFF reward_hat_diff = model(s_(t+1)) - model(s_t)\n")
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

        f.write(f"\n\n{'='*60}\n")
        f.write(f"SMOOTH reward_hat = savgol(model(s), window={sw})\n")
        f.write(f"SMOOTH DIFF       = smooth_reward_hat[t+1] - smooth_reward_hat[t]\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"=== All Steps (n={len(smooth_reward_hats)}) ===\n")
        f.write(f"smooth_reward_hat vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_smooth_all:.6f} (p={p_gt_smooth_all:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_smooth_all:.6f}\n\n")
        f.write(f"smooth_reward_hat vs Task Progress:\n")
        f.write(f"  Pearson:  {pearson_prog_smooth_all:.6f} (p={p_prog_smooth_all:.2e})\n")
        f.write(f"  Spearman: {spearman_prog_smooth_all:.6f}\n\n")
        f.write(f"smooth_diff vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_sdiff_all:.6f} (p={p_gt_sdiff_all:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_sdiff_all:.6f}\n\n")
        f.write(f"smooth_diff vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_pdiff_sdiff_all:.6f} (p={p_pdiff_sdiff_all:.2e})\n")
        f.write(f"  Spearman: {spearman_pdiff_sdiff_all:.6f}\n\n")

        f.write(f"=== First {first_n} Steps ===\n")
        f.write(f"smooth_reward_hat vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_smooth_first30:.6f} (p={p_gt_smooth_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_smooth_first30:.6f}\n\n")
        f.write(f"smooth_reward_hat vs Task Progress:\n")
        f.write(f"  Pearson:  {pearson_prog_smooth_first30:.6f} (p={p_prog_smooth_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_prog_smooth_first30:.6f}\n\n")
        f.write(f"smooth_diff vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_sdiff_first30:.6f} (p={p_gt_sdiff_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_sdiff_first30:.6f}\n\n")
        f.write(f"smooth_diff vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_pdiff_sdiff_first30:.6f} (p={p_pdiff_sdiff_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_pdiff_sdiff_first30:.6f}\n\n")

        if success_step is not None and pre_success_end > 1:
            f.write(f"=== Pre-Success (steps 0-{success_step}, n={pre_success_end}) ===\n")
            f.write(f"smooth_reward_hat vs GT Reward:\n")
            f.write(f"  Pearson:  {pearson_gt_smooth_pre:.6f} (p={p_gt_smooth_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_gt_smooth_pre:.6f}\n\n")
            f.write(f"smooth_reward_hat vs Task Progress:\n")
            f.write(f"  Pearson:  {pearson_prog_smooth_pre:.6f} (p={p_prog_smooth_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_prog_smooth_pre:.6f}\n\n")
            f.write(f"smooth_diff vs GT Reward:\n")
            f.write(f"  Pearson:  {pearson_gt_sdiff_pre:.6f} (p={p_gt_sdiff_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_gt_sdiff_pre:.6f}\n\n")
            f.write(f"smooth_diff vs Progress Diff:\n")
            f.write(f"  Pearson:  {pearson_pdiff_sdiff_pre:.6f} (p={p_pdiff_sdiff_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_pdiff_sdiff_pre:.6f}\n")

        f.write(f"\n\n{'='*60}\n")
        f.write(f"DIFF099 = 100 * (0.99 * model(s') - model(s))\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"=== All Steps (n={len(reward_hat_diffs_099)}) ===\n")
        f.write(f"diff099 vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_d099_all:.6f} (p={p_gt_d099_all:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_d099_all:.6f}\n\n")
        f.write(f"diff099 vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_pdiff_d099_all:.6f} (p={p_pdiff_d099_all:.2e})\n")
        f.write(f"  Spearman: {spearman_pdiff_d099_all:.6f}\n\n")

        f.write(f"=== First {first_n_d099} Steps ===\n")
        f.write(f"diff099 vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_d099_first30:.6f} (p={p_gt_d099_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_d099_first30:.6f}\n\n")
        f.write(f"diff099 vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_pdiff_d099_first30:.6f} (p={p_pdiff_d099_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_pdiff_d099_first30:.6f}\n\n")

        if success_step is not None and pre_success_end > 1:
            f.write(f"=== Pre-Success (steps 0-{success_step-1}, n={len(reward_hat_diffs_099_pre)}) ===\n")
            f.write(f"diff099 vs GT Reward:\n")
            f.write(f"  Pearson:  {pearson_gt_d099_pre:.6f} (p={p_gt_d099_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_gt_d099_pre:.6f}\n\n")
            f.write(f"diff099 vs Progress Diff:\n")
            f.write(f"  Pearson:  {pearson_pdiff_d099_pre:.6f} (p={p_pdiff_d099_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_pdiff_d099_pre:.6f}\n")

        f.write(f"\n\n{'='*60}\n")
        f.write(f"DIFF0999 = 1000 * (0.999 * model(s') - model(s))\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"=== All Steps (n={len(reward_hat_diffs_0999)}) ===\n")
        f.write(f"diff0999 vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_d0999_all:.6f} (p={p_gt_d0999_all:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_d0999_all:.6f}\n\n")
        f.write(f"diff0999 vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_pdiff_d0999_all:.6f} (p={p_pdiff_d0999_all:.2e})\n")
        f.write(f"  Spearman: {spearman_pdiff_d0999_all:.6f}\n\n")

        f.write(f"=== First {first_n_d0999} Steps ===\n")
        f.write(f"diff0999 vs GT Reward:\n")
        f.write(f"  Pearson:  {pearson_gt_d0999_first30:.6f} (p={p_gt_d0999_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_gt_d0999_first30:.6f}\n\n")
        f.write(f"diff0999 vs Progress Diff:\n")
        f.write(f"  Pearson:  {pearson_pdiff_d0999_first30:.6f} (p={p_pdiff_d0999_first30:.2e})\n")
        f.write(f"  Spearman: {spearman_pdiff_d0999_first30:.6f}\n\n")

        if success_step is not None and pre_success_end > 1:
            f.write(f"=== Pre-Success (steps 0-{success_step-1}, n={len(reward_hat_diffs_0999_pre)}) ===\n")
            f.write(f"diff0999 vs GT Reward:\n")
            f.write(f"  Pearson:  {pearson_gt_d0999_pre:.6f} (p={p_gt_d0999_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_gt_d0999_pre:.6f}\n\n")
            f.write(f"diff0999 vs Progress Diff:\n")
            f.write(f"  Pearson:  {pearson_pdiff_d0999_pre:.6f} (p={p_pdiff_d0999_pre:.2e})\n")
            f.write(f"  Spearman: {spearman_pdiff_d0999_pre:.6f}\n")

    print(f"Saved correlation analysis to {corr_path}")

    # Save per-step raw data to CSV
    csv_path = os.path.join(args.output_dir, 'step_data.csv')
    with open(csv_path, 'w') as f:
        reward_hat_diffs_padded = np.concatenate([[0.0], reward_hat_diffs])
        f.write('step,reward_hat,reward_hat_diff_padded,gt_reward,task_progress,smooth_rhat,smooth_reward_hat_diff_padded,diff099_padded,diff0999_padded\n')
        for i in range(len(reward_hats)):
            f.write(f'{i},{reward_hats[i]:.6f},{reward_hat_diffs_padded[i]:.6f},{gt_rewards[i]:.6f},{task_progress[i]:.6f},'
                    f'{smooth_reward_hats[i]:.6f},{smooth_reward_hat_diffs_padded[i]:.6f},'
                    f'{reward_hat_diffs_099_padded[i]:.6f},{reward_hat_diffs_0999_padded[i]:.6f}\n')
    print(f"Saved per-step data to {csv_path}")

    # Function to generate video with raw reward_hat and reward_hat diff panels
    def generate_video(images_subset, reward_hats_subset, gt_rewards_subset,
                       video_path, env_name, success_step_local, fps=20,
                       diff099_subset=None, diff0999_subset=None):
        num_frames = len(images_subset)
        rhat_diffs_padded = np.concatenate([[0.0], np.diff(reward_hats_subset)])
        if diff099_subset is None:
            diff099_subset = reward_hat_diffs_099_padded[:num_frames]
        if diff0999_subset is None:
            diff0999_subset = reward_hat_diffs_0999_padded[:num_frames]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Top-left: Environment image
        ax_img = axes[0, 0]
        im = ax_img.imshow(images_subset[0])
        ax_img.set_title('Environment', fontsize=12)
        ax_img.axis('off')
        step_text = ax_img.text(0.02, 0.98, '', transform=ax_img.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Top-middle: reward model output (raw values)
        ax_rhat = axes[0, 1]
        line_rhat, = ax_rhat.plot([], [], 'b-', linewidth=2)
        ax_rhat.set_xlim(0, num_frames)
        rhat_margin = max(0.1, (np.max(reward_hats_subset) - np.min(reward_hats_subset)) * 0.1)
        ax_rhat.set_ylim(np.min(reward_hats_subset) - rhat_margin, np.max(reward_hats_subset) + rhat_margin)
        ax_rhat.set_xlabel('Step')
        ax_rhat.set_ylabel('reward_hat')
        ax_rhat.set_title('Reward Model Output (raw)', fontsize=12)
        ax_rhat.grid(True, alpha=0.3)
        dot_rhat, = ax_rhat.plot([], [], 'bo', markersize=6)

        # Top-right: 100*(0.99*P(s')-P(s))
        ax_d099 = axes[0, 2]
        line_d099, = ax_d099.plot([], [], color='orange', linewidth=2)
        ax_d099.set_xlim(0, num_frames)
        d099_margin = max(0.1, (np.max(diff099_subset) - np.min(diff099_subset)) * 0.1)
        ax_d099.set_ylim(np.min(diff099_subset) - d099_margin, np.max(diff099_subset) + d099_margin)
        ax_d099.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax_d099.set_xlabel('Step')
        ax_d099.set_ylabel('100*(0.99*P(s\')-P(s))')
        ax_d099.set_title('100*(0.99*P(s\')-P(s))', fontsize=12)
        ax_d099.grid(True, alpha=0.3)
        dot_d099, = ax_d099.plot([], [], 'o', color='orange', markersize=6)

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

        # Bottom-middle: reward_hat diff
        ax_diff = axes[1, 1]
        line_diff, = ax_diff.plot([], [], 'm-', linewidth=2)
        ax_diff.set_xlim(0, num_frames)
        diff_margin = max(0.1, (np.max(rhat_diffs_padded) - np.min(rhat_diffs_padded)) * 0.1)
        ax_diff.set_ylim(np.min(rhat_diffs_padded) - diff_margin, np.max(rhat_diffs_padded) + diff_margin)
        ax_diff.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_diff.set_xlabel('Step')
        ax_diff.set_ylabel('reward_hat diff')
        ax_diff.set_title('reward_hat diff', fontsize=12)
        ax_diff.grid(True, alpha=0.3)
        dot_diff, = ax_diff.plot([], [], 'mo', markersize=6)

        # Bottom-right: 1000*(0.999*P(s')-P(s))
        ax_d0999 = axes[1, 2]
        line_d0999, = ax_d0999.plot([], [], color='purple', linewidth=2)
        ax_d0999.set_xlim(0, num_frames)
        d0999_margin = max(0.1, (np.max(diff0999_subset) - np.min(diff0999_subset)) * 0.1)
        ax_d0999.set_ylim(np.min(diff0999_subset) - d0999_margin, np.max(diff0999_subset) + d0999_margin)
        ax_d0999.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax_d0999.set_xlabel('Step')
        ax_d0999.set_ylabel('1000*(0.999*P(s\')-P(s))')
        ax_d0999.set_title('1000*(0.999*P(s\')-P(s))', fontsize=12)
        ax_d0999.grid(True, alpha=0.3)
        dot_d0999, = ax_d0999.plot([], [], 'o', color='purple', markersize=6)

        plt.suptitle(f'Expert Trajectory Analysis - {env_name}', fontsize=14)
        plt.tight_layout()

        def init():
            line_rhat.set_data([], [])
            line_gt.set_data([], [])
            line_diff.set_data([], [])
            line_d099.set_data([], [])
            line_d0999.set_data([], [])
            dot_rhat.set_data([], [])
            dot_gt.set_data([], [])
            dot_diff.set_data([], [])
            dot_d099.set_data([], [])
            dot_d0999.set_data([], [])
            step_text.set_text('')
            return line_rhat, line_gt, line_diff, line_d099, line_d0999, dot_rhat, dot_gt, dot_diff, dot_d099, dot_d0999, step_text, im

        def animate(frame):
            im.set_array(images_subset[frame])

            status = "SUCCESS!" if success_step_local is not None and frame >= success_step_local else ""
            step_text.set_text(f'Step: {frame}/{num_frames-1} {status}')

            x_data = np.arange(frame + 1)

            line_rhat.set_data(x_data, reward_hats_subset[:frame + 1])
            line_gt.set_data(x_data, gt_rewards_subset[:frame + 1])
            line_diff.set_data(x_data, rhat_diffs_padded[:frame + 1])
            line_d099.set_data(x_data, diff099_subset[:frame + 1])
            line_d0999.set_data(x_data, diff0999_subset[:frame + 1])

            dot_rhat.set_data([frame], [reward_hats_subset[frame]])
            dot_gt.set_data([frame], [gt_rewards_subset[frame]])
            dot_diff.set_data([frame], [rhat_diffs_padded[frame]])
            dot_d099.set_data([frame], [diff099_subset[frame]])
            dot_d0999.set_data([frame], [diff0999_subset[frame]])

            return line_rhat, line_gt, line_diff, line_d099, line_d0999, dot_rhat, dot_gt, dot_diff, dot_d099, dot_d0999, step_text, im

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
    generate_video(images, reward_hats, gt_rewards,
                   video_path_full, args.env, success_step)

    # Generate first 30 steps video (10x slower, Original)
    print(f"\n=== Generating First 30 Steps Video (10x slower, Original) ===")
    num_frames_short = min(30, len(images))
    success_step_short = success_step if success_step is not None and success_step < num_frames_short else None
    video_path_short = os.path.join(args.output_dir, 'trajectory_analysis_first30.mp4')
    generate_video(images[:num_frames_short], reward_hats[:num_frames_short],
                   gt_rewards[:num_frames_short],
                   video_path_short, args.env, success_step_short, fps=2)

    # Generate pre-success video (Original, 10x slower)
    if success_step is not None and success_step > 0:
        print(f"\n=== Generating Pre-Success Video (10x slower, Original) ===")
        video_path_presuccess = os.path.join(args.output_dir, 'trajectory_analysis_presuccess.mp4')
        generate_video(images[:pre_success_end], reward_hats[:pre_success_end],
                       gt_rewards[:pre_success_end],
                       video_path_presuccess, args.env + " (Pre-Success)", success_step, fps=2)

    # Generate full video (Diff form)
    print(f"\n=== Generating Full Animation Video (Diff) ===")
    video_path_full_diff = os.path.join(args.output_dir, 'trajectory_analysis_diff.mp4')
    success_step_diff = success_step if success_step is not None else None
    generate_video(images, reward_hats, gt_rewards,
                   video_path_full_diff, args.env + " (DIFF)", success_step_diff)

    # Generate first 30 steps video (10x slower, Diff)
    print(f"\n=== Generating First 30 Steps Video (10x slower, Diff) ===")
    num_frames_short_diff = min(30, len(images))
    success_step_short_diff = success_step_diff if success_step_diff is not None and success_step_diff < num_frames_short_diff else None
    video_path_short_diff = os.path.join(args.output_dir, 'trajectory_analysis_first30_diff.mp4')
    generate_video(images[:num_frames_short_diff], reward_hats[:num_frames_short_diff],
                   gt_rewards[:num_frames_short_diff],
                   video_path_short_diff, args.env + " (DIFF)", success_step_short_diff, fps=2)

    # Generate pre-success video (Diff form, 10x slower)
    if success_step is not None and success_step > 0:
        print(f"\n=== Generating Pre-Success Video (10x slower, Diff) ===")
        pre_success_end_diff = pre_success_end
        video_path_presuccess_diff = os.path.join(args.output_dir, 'trajectory_analysis_presuccess_diff.mp4')
        generate_video(images[:pre_success_end_diff], reward_hats[:pre_success_end_diff],
                       gt_rewards[:pre_success_end_diff],
                       video_path_presuccess_diff, args.env + " (DIFF, Pre-Success)", success_step_diff, fps=2)

    # -----------------------------------------------------------------------
    # Smooth videos: top-right = smooth_reward_hat, bottom-right = smooth diff
    # -----------------------------------------------------------------------
    def generate_video_smooth(images_subset, smooth_rhat_subset,
                              gt_rewards_subset, video_path, env_name, success_step_local, fps=20,
                              diff099_subset=None, diff0999_subset=None):
        """2x3 video: env | smooth_reward_hat | 100*(0.99*P(s')-P(s)) / GT Reward | smooth_diff | 1000*(0.999*P(s')-P(s))."""
        num_frames = len(images_subset)
        if diff099_subset is None:
            diff099_subset = reward_hat_diffs_099_padded[:num_frames]
        if diff0999_subset is None:
            diff0999_subset = reward_hat_diffs_0999_padded[:num_frames]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Top-left: Environment image
        ax_img = axes[0, 0]
        im = ax_img.imshow(images_subset[0])
        ax_img.set_title('Environment', fontsize=12)
        ax_img.axis('off')
        step_text = ax_img.text(0.02, 0.98, '', transform=ax_img.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Top-middle: smooth reward_hat
        ax_smooth = axes[0, 1]
        line_smooth, = ax_smooth.plot([], [], 'b-', linewidth=2)
        ax_smooth.set_xlim(0, num_frames)
        sm = max(0.05, (np.max(smooth_rhat_subset) - np.min(smooth_rhat_subset)) * 0.1)
        ax_smooth.set_ylim(np.min(smooth_rhat_subset) - sm, np.max(smooth_rhat_subset) + sm)
        ax_smooth.set_xlabel('Step')
        ax_smooth.set_ylabel('smooth_reward_hat')
        ax_smooth.set_title(f'Smoothed reward_hat (window={sw})', fontsize=12)
        ax_smooth.grid(True, alpha=0.3)
        dot_smooth, = ax_smooth.plot([], [], 'bo', markersize=6)

        # Top-right: 100*(0.99*P(s')-P(s))
        ax_d099 = axes[0, 2]
        line_d099, = ax_d099.plot([], [], color='orange', linewidth=2)
        ax_d099.set_xlim(0, num_frames)
        d099_margin = max(0.1, (np.max(diff099_subset) - np.min(diff099_subset)) * 0.1)
        ax_d099.set_ylim(np.min(diff099_subset) - d099_margin, np.max(diff099_subset) + d099_margin)
        ax_d099.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax_d099.set_xlabel('Step')
        ax_d099.set_ylabel('100*(0.99*P(s\')-P(s))')
        ax_d099.set_title('100*(0.99*P(s\')-P(s))', fontsize=12)
        ax_d099.grid(True, alpha=0.3)
        dot_d099, = ax_d099.plot([], [], 'o', color='orange', markersize=6)

        # Bottom-left: GT Reward
        ax_gt = axes[1, 0]
        line_gt, = ax_gt.plot([], [], 'r-', linewidth=2)
        ax_gt.set_xlim(0, num_frames)
        gm = max(0.1, (np.max(gt_rewards_subset) - np.min(gt_rewards_subset)) * 0.1)
        ax_gt.set_ylim(np.min(gt_rewards_subset) - gm, np.max(gt_rewards_subset) + gm)
        ax_gt.set_xlabel('Step')
        ax_gt.set_ylabel('GT Reward')
        ax_gt.set_title('GT Reward', fontsize=12)
        ax_gt.grid(True, alpha=0.3)
        dot_gt, = ax_gt.plot([], [], 'ro', markersize=6)

        # Bottom-middle: smooth reward_hat diff (padded, first value = 0)
        smooth_rhat_diffs_padded = np.concatenate([[0.0], np.diff(smooth_rhat_subset)])
        ax_diff = axes[1, 1]
        diff_vals = smooth_rhat_diffs_padded[1:]  # skip leading 0 for axis range
        dm = max(0.005, (np.max(np.abs(diff_vals)) if len(diff_vals) > 0 else 0.01) * 1.2)
        ax_diff.set_xlim(0, num_frames)
        ax_diff.set_ylim(-dm, dm)
        ax_diff.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        line_diff, = ax_diff.plot([], [], 'm-', linewidth=2)
        ax_diff.set_xlabel('Step')
        ax_diff.set_ylabel('smooth_diff')
        ax_diff.set_title('Smoothed reward_hat diff', fontsize=12)
        ax_diff.grid(True, alpha=0.3)
        dot_diff, = ax_diff.plot([], [], 'mo', markersize=6)

        # Bottom-right: 1000*(0.999*P(s')-P(s))
        ax_d0999 = axes[1, 2]
        line_d0999, = ax_d0999.plot([], [], color='purple', linewidth=2)
        ax_d0999.set_xlim(0, num_frames)
        d0999_margin = max(0.1, (np.max(diff0999_subset) - np.min(diff0999_subset)) * 0.1)
        ax_d0999.set_ylim(np.min(diff0999_subset) - d0999_margin, np.max(diff0999_subset) + d0999_margin)
        ax_d0999.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax_d0999.set_xlabel('Step')
        ax_d0999.set_ylabel('1000*(0.999*P(s\')-P(s))')
        ax_d0999.set_title('1000*(0.999*P(s\')-P(s))', fontsize=12)
        ax_d0999.grid(True, alpha=0.3)
        dot_d0999, = ax_d0999.plot([], [], 'o', color='purple', markersize=6)

        plt.suptitle(f'Expert Trajectory Analysis (Smoothed) - {env_name}', fontsize=14)
        plt.tight_layout()

        def init():
            line_smooth.set_data([], [])
            line_gt.set_data([], [])
            line_diff.set_data([], [])
            line_d099.set_data([], [])
            line_d0999.set_data([], [])
            dot_smooth.set_data([], [])
            dot_gt.set_data([], [])
            dot_diff.set_data([], [])
            dot_d099.set_data([], [])
            dot_d0999.set_data([], [])
            step_text.set_text('')
            return line_smooth, line_gt, line_diff, line_d099, line_d0999, dot_smooth, dot_gt, dot_diff, dot_d099, dot_d0999, step_text, im

        def animate(frame):
            im.set_array(images_subset[frame])
            status = "SUCCESS!" if success_step_local is not None and frame >= success_step_local else ""
            step_text.set_text(f'Step: {frame}/{num_frames-1} {status}')
            x_data = np.arange(frame + 1)
            line_smooth.set_data(x_data, smooth_rhat_subset[:frame + 1])
            line_gt.set_data(x_data, gt_rewards_subset[:frame + 1])
            line_diff.set_data(x_data, smooth_rhat_diffs_padded[:frame + 1])
            line_d099.set_data(x_data, diff099_subset[:frame + 1])
            line_d0999.set_data(x_data, diff0999_subset[:frame + 1])
            dot_smooth.set_data([frame], [smooth_rhat_subset[frame]])
            dot_gt.set_data([frame], [gt_rewards_subset[frame]])
            dot_diff.set_data([frame], [smooth_rhat_diffs_padded[frame]])
            dot_d099.set_data([frame], [diff099_subset[frame]])
            dot_d0999.set_data([frame], [diff0999_subset[frame]])
            return line_smooth, line_gt, line_diff, line_d099, line_d0999, dot_smooth, dot_gt, dot_diff, dot_d099, dot_d0999, step_text, im

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=num_frames, interval=50, blit=True)

        print(f"Saving video to {video_path} ({num_frames} frames, fps={fps})...")
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='RL-VLM-F'), bitrate=2400)
        anim.save(video_path, writer=writer)
        print(f"Saved video to {video_path}")
        plt.close(fig)

    # Generate full smooth video
    print(f"\n=== Generating Full Animation Video (Smooth) ===")
    video_path_smooth = os.path.join(args.output_dir, 'trajectory_analysis_smooth.mp4')
    generate_video_smooth(images, smooth_reward_hats,
                          gt_rewards, video_path_smooth, args.env, success_step)

    # Generate first 30 steps smooth video
    print(f"\n=== Generating First 30 Steps Video (10x slower, Smooth) ===")
    num_frames_short_smooth = min(30, len(images))
    success_step_short_smooth = success_step if success_step is not None and success_step < num_frames_short_smooth else None
    video_path_short_smooth = os.path.join(args.output_dir, 'trajectory_analysis_first30_smooth.mp4')
    generate_video_smooth(images[:num_frames_short_smooth],
                          smooth_reward_hats[:num_frames_short_smooth],
                          gt_rewards[:num_frames_short_smooth],
                          video_path_short_smooth, args.env, success_step_short_smooth, fps=2)

    # Generate pre-success smooth video
    if success_step is not None and success_step > 0:
        print(f"\n=== Generating Pre-Success Video (10x slower, Smooth) ===")
        video_path_presuccess_smooth = os.path.join(args.output_dir, 'trajectory_analysis_presuccess_smooth.mp4')
        generate_video_smooth(images[:pre_success_end],
                              smooth_reward_hats[:pre_success_end],
                              gt_rewards[:pre_success_end],
                              video_path_presuccess_smooth, args.env + " (Pre-Success)",
                              success_step, fps=2)


    return reward_hats, reward_hat_diffs, gt_rewards, task_progress, progress_diffs


if __name__ == '__main__':
    main()

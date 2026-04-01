import math

import numpy as np


def get_progress_diff_reward_scale(reward_scale, discount, scale_by_inv_one_minus_gamma=False):
    if scale_by_inv_one_minus_gamma:
        if not math.isfinite(discount) or discount < 0.0 or discount >= 1.0:
            raise ValueError(
                "progress_diff_scale_by_inv_one_minus_gamma=true requires "
                "0 <= progress_diff_discount < 1"
            )
        inv_one_minus_gamma_scale = 1.0 / (1.0 - discount)
    else:
        inv_one_minus_gamma_scale = 1.0

    return reward_scale * inv_one_minus_gamma_scale, inv_one_minus_gamma_scale


def compute_progress_diff_rewards(values, discount, reward_scale=1.0,
                                  scale_by_inv_one_minus_gamma=False):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.empty((0,), dtype=np.float64)

    effective_reward_scale, _ = get_progress_diff_reward_scale(
        reward_scale=reward_scale,
        discount=discount,
        scale_by_inv_one_minus_gamma=scale_by_inv_one_minus_gamma,
    )
    return (discount * values[1:] - values[:-1]) * effective_reward_scale

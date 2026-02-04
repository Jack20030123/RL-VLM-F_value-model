"""
Video Visualizer for RL-VLM-F
Creates side-by-side videos showing environment frames and reward_hat curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import cv2
import os
import imageio
import gc

class RewardVideoVisualizer:
    """Visualizer for environment + reward curves"""

    def __init__(self, env_frame_size=(300, 300), plot_size=(300, 300),
                 fps=30, max_episode_length=500, save_dir="./videos"):
        self.env_frame_size = env_frame_size
        self.plot_size = plot_size
        self.base_fps = fps  # Base fps for < 40 or > 200 frames
        self.max_episode_length = max_episode_length
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reset_episode()

    def reset_episode(self):
        self.frames = []
        self.rewards = []
        self.steps = []
        self.episode_step = 0

    def add_frame(self, env_frame, reward_hat):
        if env_frame.shape[:2] != self.env_frame_size:
            target_h, target_w = self.env_frame_size
            src_h, src_w = env_frame.shape[:2]
            scale = min(target_w / src_w, target_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            resized = cv2.resize(env_frame, (new_w, new_h))
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            env_frame = padded
        self.frames.append(env_frame)
        self.rewards.append(reward_hat)
        self.steps.append(self.episode_step)
        self.episode_step += 1

    def create_reward_plot(self, current_step):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        steps_to_plot = self.steps[:current_step+1]
        rewards_to_plot = self.rewards[:current_step+1]

        ax.plot(steps_to_plot, rewards_to_plot, color="#2E86AB", linewidth=2)

        if len(rewards_to_plot) > 0:
            ax.scatter([steps_to_plot[-1]], [rewards_to_plot[-1]], color="#A23B72", s=50, zorder=5)

        ax.set_xlabel("Time Step", fontsize=7)
        ax.set_title("reward_hat", fontsize=8, fontweight="bold")
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, alpha=0.3)
        # Use actual episode length for x-axis (with small margin)
        total_steps = len(self.steps)
        ax.set_xlim(0, max(total_steps + 1, 2))  # At least show 0-2 range for very short episodes

        if len(rewards_to_plot) > 0:
            y_min = min(rewards_to_plot)
            y_max = max(rewards_to_plot)
            y_range = y_max - y_min
            if y_range < 0.01:  # Very small or zero range (single point or constant rewards)
                y_margin = 0.5  # Use fixed margin
            else:
                y_margin = y_range * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            ax.set_ylim(-1, 1)

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return plot_img

    def create_combined_frame(self, step_idx):
        env_frame = self.frames[step_idx].copy()
        plot_frame = self.create_reward_plot(step_idx)

        # Get environment frame height
        env_h = env_frame.shape[0]

        # Resize plot to match environment frame height while preserving aspect ratio
        plot_h, plot_w = plot_frame.shape[:2]
        scale = env_h / plot_h
        new_plot_w = int(plot_w * scale)
        new_plot_h = env_h
        plot_frame = cv2.resize(plot_frame, (new_plot_w, new_plot_h))

        # Add step text
        cv2.putText(env_frame, f"Step: {self.steps[step_idx]}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return np.hstack([env_frame, plot_frame])

    def _calculate_fps(self, num_frames):
        """Calculate fps based on frame count to achieve target video duration.

        Target durations:
        - 1 frame: 1 second (fps=1)
        - 2-39 frames: 2 seconds
        - 40-99 frames: ~10 seconds
        - 100-200 frames: ~20 seconds
        - > 200 frames: ~30 seconds
        """
        if num_frames <= 0:
            return self.base_fps

        if num_frames == 1:
            # Single frame: 1 second
            fps = 1.0
        elif num_frames < 40:
            # 2-39 frames: ~2 seconds
            fps = num_frames / 2.0
        elif num_frames <= 99:
            # 40-99 frames: ~10 seconds
            fps = num_frames / 10.0
        elif num_frames <= 200:
            # 100-200 frames: ~20 seconds
            fps = num_frames / 20.0
        else:
            # > 200 frames: ~30 seconds
            fps = num_frames / 30.0

        return max(fps, 1.0)  # At least 1 fps

    def save_video(self, filename, episode_num=None):
        if len(self.frames) == 0:
            print("Warning: No frames to save")
            return None

        if episode_num is not None:
            video_path = os.path.join(self.save_dir, f"{filename}_ep{episode_num}.mp4")
        else:
            video_path = os.path.join(self.save_dir, f"{filename}.mp4")

        num_frames = len(self.frames)
        fps = self._calculate_fps(num_frames)
        avg_reward = np.mean(self.rewards)

        # Force garbage collection before spawning ffmpeg subprocess
        # This helps prevent "Cannot allocate memory" errors during fork()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

        # Use streaming write to avoid loading all frames into memory at once
        writer = imageio.get_writer(video_path, fps=fps)
        try:
            for i in range(num_frames):
                combined_frame = self.create_combined_frame(i)
                writer.append_data(combined_frame)
                # Clear processed frame to free memory
                self.frames[i] = None
        finally:
            writer.close()

        print(f"Video saved to: {video_path}")
        print(f"  - Total frames: {num_frames}")
        print(f"  - FPS: {fps:.1f}")
        print(f"  - Duration: {num_frames/fps:.1f}s")
        print(f"  - Avg reward: {avg_reward:.3f}")

        return video_path

    def save_gif(self, filename, episode_num=None, max_frames=100):
        if len(self.frames) == 0:
            return None

        if episode_num is not None:
            gif_path = os.path.join(self.save_dir, f"{filename}_ep{episode_num}.gif")
        else:
            gif_path = os.path.join(self.save_dir, f"{filename}.gif")

        if len(self.frames) > max_frames:
            indices = np.linspace(0, len(self.frames)-1, max_frames, dtype=int)
        else:
            indices = range(len(self.frames))

        combined_frames = [self.create_combined_frame(i) for i in indices]
        # Calculate dynamic fps for gif (capped at 10 for compatibility)
        fps = min(self._calculate_fps(len(combined_frames)), 10)
        imageio.mimsave(gif_path, combined_frames, fps=fps, loop=0)
        print(f"GIF saved to: {gif_path} (fps={fps:.1f})")
        return gif_path


class EpisodeRecorder:
    """Helper to record episodes during training"""

    def __init__(self, visualizer, record_frequency=10, max_videos=5):
        self.visualizer = visualizer
        self.record_frequency = record_frequency
        self.max_videos = max_videos
        self.episode_count = 0
        self.recording = False
        self.saved_videos = []

    def should_record(self):
        return self.episode_count % self.record_frequency == 0

    def start_episode(self):
        if self.should_record():
            self.recording = True
            self.visualizer.reset_episode()
        else:
            self.recording = False

    def add_step(self, env_frame, reward_hat):
        if self.recording:
            self.visualizer.add_frame(env_frame, reward_hat)

    def end_episode(self, save_video=True, save_gif=False):
        if self.recording:
            if save_video:
                video_path = self.visualizer.save_video("training", episode_num=self.episode_count)
                if video_path:
                    self.saved_videos.append(video_path)
                    if len(self.saved_videos) > self.max_videos:
                        old_video = self.saved_videos.pop(0)
                        if os.path.exists(old_video):
                            os.remove(old_video)

            if save_gif:
                self.visualizer.save_gif("training", episode_num=self.episode_count)

            self.recording = False

        self.episode_count += 1

    def get_latest_video(self):
        if len(self.saved_videos) > 0:
            return self.saved_videos[-1]
        return None

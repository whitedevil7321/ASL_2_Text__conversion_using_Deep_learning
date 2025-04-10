import os
import cv2
import torch
import ffmpeg
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pytorchvideo.transforms import UniformTemporalSubsample

TARGET_FRAMES = 16
TARGET_HEIGHT = 500
TARGET_WIDTH = 500


def read_video_ffmpeg_gpu(video_path):
    """Read video using ffmpeg with GPU acceleration and resize."""
    try:
        out, _ = (
            ffmpeg
            .input(video_path, hwaccel='cuda')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{TARGET_WIDTH}x{TARGET_HEIGHT}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).copy().reshape([-1, TARGET_HEIGHT, TARGET_WIDTH, 3])
        return torch.from_numpy(video)  # (T, H, W, C)
    except Exception as e:
        raise RuntimeError(f"⚠️ Failed to load {video_path} with GPU ffmpeg: {e}")


def smart_pad_frames(frames, target_frames=TARGET_FRAMES):
    current_len = frames.shape[0]
    if current_len >= target_frames:
        return frames[:target_frames]

    n_to_add = target_frames - current_len
    indices_to_duplicate = torch.linspace(1, current_len - 2, steps=n_to_add).long()
    padded = []
    for i in range(current_len):
        padded.append(frames[i])
        if i in indices_to_duplicate:
            padded.append(frames[i])
    return torch.stack(padded[:target_frames])


def normalize_frame_count(frames, target_frames=TARGET_FRAMES):
    if frames.shape[0] < target_frames:
        print(f"➕ Padding from {frames.shape[0]} to {target_frames} frames")
        return smart_pad_frames(frames, target_frames)
    elif frames.shape[0] > target_frames:
        print(f"✂️ Subsampling from {frames.shape[0]} to {target_frames} frames")
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        frames = UniformTemporalSubsample(num_samples=target_frames)(frames)
        return frames.permute(1, 0, 2, 3)  # Back to (T, C, H, W)
    return frames


def save_video(frames, output_path, fps=30):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i, frame in enumerate(frames):
        if frame.shape[-1] != 3:
            raise ValueError(f"❌ Invalid frame shape at index {i}: {frame.shape}")
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def process_video(video_path, num_frames):
    try:
        frames = read_video_ffmpeg_gpu(str(video_path))
        print(f"📥 {video_path.name} - Loaded shape: {frames.shape}")

        if frames.shape[0] == 0:
            raise ValueError("Empty video")

        # Attempt to fix dimensions
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Invalid shape: {frames.shape}")

        if frames.shape[1] != TARGET_HEIGHT or frames.shape[2] != TARGET_WIDTH:
            print(f"⚠️ Resizing frames from {frames.shape[1:3]} to ({TARGET_HEIGHT}, {TARGET_WIDTH})")
            resized_frames = []
            for i, frame in enumerate(frames):
                resized = cv2.resize(frame.numpy(), (TARGET_WIDTH, TARGET_HEIGHT))
                resized_frames.append(torch.from_numpy(resized))
            frames = torch.stack(resized_frames)

        # Convert to (T, C, H, W)
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Cannot permute to (T, C, H, W) from {frames.shape}")

        print(f"🔁 Permuted: {frames.shape}")

        # Normalize frame count
        frames = normalize_frame_count(frames, target_frames=num_frames)

        # Back to (T, H, W, C)
        final_frames = frames.permute(0, 2, 3, 1).byte().numpy()
        print(f"🎞️ Final output shape: {final_frames.shape}")
        return final_frames

    except Exception as e:
        print(f"⚠️ Skipped {Path(video_path).name}: {e}")
        return None


def process_all_videos(root_dir, num_frames=TARGET_FRAMES):
    root_path = Path(root_dir)
    for folder in sorted(root_path.iterdir()):
        if folder.is_dir():
            video_files = list(sorted(folder.glob("*.mp4")))
            print(f"\n📁 Processing folder: {folder.name} ({len(video_files)} videos)")
            for video_file in tqdm(video_files, desc=f"🎞️ {folder.name}", unit="video"):
                processed = process_video(video_file, num_frames)
                if processed is not None:
                    try:
                        save_video(processed, video_file)
                    except Exception as e:
                        tqdm.write(f"❌ Failed to save {video_file.name}: {e}")
                else:
                    tqdm.write(f"🗑️ Could not fix or process: {video_file.name}")
                    try:
                        video_file.unlink()
                    except Exception:
                        pass


if __name__ == "__main__":
    process_all_videos(r"D:\archive\dataset\SL", num_frames=TARGET_FRAMES)

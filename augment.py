import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random
np.float = float  # Fix deprecated alias for older vidaug compatibility

from vidaug import augmentors as va

def read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def save_frames_to_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

def resize_frames(frames, size=(224, 224)):
    return [cv2.resize(frame, size) for frame in frames]

def process_video(video_path, target_size=(224, 224)):
    frames = read_video_frames(video_path)
    if not frames:
        print(f"⚠️ Failed to read: {video_path}")
        return

    resized_frames = resize_frames(frames, size=target_size)

    # Save scaled version
    scaled_path = video_path.parent / f"scaled_{video_path.name}"
    save_frames_to_video(resized_frames, scaled_path)

    # Create new augmentor pipeline for each video
    seq = va.Sequential([
        va.HorizontalFlip(),
        va.Add(random.randint(-20, 20)),
        va.Multiply(round(random.uniform(0.8, 1.2), 2)),
        va.Pepper(5)
    ])

    augmented_frames = seq(resized_frames)
    augmented_path = video_path.parent / f"augmented_{video_path.name}"
    save_frames_to_video(augmented_frames, augmented_path)

    # 🗑️ Delete original video
    try:
        os.remove(video_path)
        print(f"🗑️ Deleted: {video_path.name}")
    except Exception as e:
        print(f"❌ Could not delete {video_path.name}: {e}")

def process_folder(root_dir, size=(500, 500)):
    root = Path(root_dir)
    for class_dir in tqdm(list(root.glob("*")), desc="Processing classes"):
        if not class_dir.is_dir():
            continue
        for video_file in class_dir.glob("*.mp4"):
            process_video(video_file, target_size=size)

if __name__ == "__main__":
    process_folder("D:/Downloads/archive/dataset/SL")

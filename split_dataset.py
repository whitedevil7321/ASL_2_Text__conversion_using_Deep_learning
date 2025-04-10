
# split_dataset.py
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    assert source_dir.exists(), f"Source path {source_dir} doesn't exist"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    for label_dir in source_dir.iterdir():
        if not label_dir.is_dir():
            continue

        videos = list(label_dir.glob("*.mp4"))
        random.shuffle(videos)
        total = len(videos)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            'train': videos[:train_end],
            'val': videos[train_end:val_end],
            'test': videos[val_end:],
        }

        for split_name, split_videos in splits.items():
            split_dir = dest_dir / split_name / label_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for video_path in split_videos:
                shutil.copy(video_path, split_dir / video_path.name)

    print(f"✅ Dataset split into {dest_dir} (train/val/test)")

# Example usage:
# split_dataset("D:/Downloads/archive/dataset/SL", "data")


import os
from torchvision.io import read_video
from pathlib import Path
from tqdm import tqdm

def check_video_dimensions(video_path):
    try:
        frames, _, _ = read_video(video_path, pts_unit='sec')

        if frames.shape[0] == 0:
            return "❌ No frames"

        return f"{frames.shape}"  # (T, H, W, C)

    except Exception as e:
        return f"❌ Error: {str(e)}"

def check_all_videos(root_dir):
    root_path = Path(root_dir)
    for folder in sorted(root_path.iterdir()):
        if folder.is_dir():
            video_files = list(sorted(folder.glob("*.mp4")))
            print(f"\n📁 Checking folder: {folder.name} ({len(video_files)} videos)")

            for video_file in tqdm(video_files, desc=f"🔍 {folder.name}", unit="video"):
                shape_info = check_video_dimensions(str(video_file))
                tqdm.write(f"{video_file.name}: {shape_info}")

if __name__ == "__main__":
    check_all_videos(r"D:\archive\dataset\SL")

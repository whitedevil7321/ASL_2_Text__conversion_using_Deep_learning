import os
import shutil
import random
from pathlib import Path

def normalize_and_rename_videos(root_dir, target_count=100):
    root = Path(root_dir)

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        video_files = sorted(class_dir.glob("*.mp4"))
        current_count = len(video_files)

        print(f"\n📁 Processing '{class_name}': {current_count} videos")

        # Delete excess videos
        if current_count > target_count:
            for file in video_files[target_count:]:
                try:
                    file.unlink()
                    print(f"🗑️ Deleted: {file.name}")
                except Exception as e:
                    print(f"❌ Failed to delete {file.name}: {e}")

        # Refresh video list
        video_files = sorted(class_dir.glob("*.mp4"))

        # Duplicate videos if needed
        while len(video_files) < target_count:
            to_copy = random.choice(video_files)
            new_index = len(video_files) + 1
            new_file = class_dir / f"{class_name}_dup_{new_index}.mp4"
            shutil.copy(to_copy, new_file)
            video_files.append(new_file)
            print(f"📄 Duplicated: {to_copy.name} → {new_file.name}")

        # Final renaming
        for idx, file in enumerate(sorted(class_dir.glob("*.mp4")), start=1):
            new_name = f"{class_name}_{idx}.mp4"
            new_path = class_dir / new_name
            try:
                file.rename(new_path)
            except Exception as e:
                print(f"❌ Failed to rename {file.name}: {e}")

        print(f"✅ Normalized to 100 videos in '{class_name}'")

# Run the script
if __name__ == "__main__":
    normalize_and_rename_videos(r"D:\archive\dataset\SL")

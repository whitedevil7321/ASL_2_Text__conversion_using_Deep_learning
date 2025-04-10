import os
import shutil
import random

# === Define base paths ===
base_dir = "G:/dataset_split/SL"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# === Ensure all directories exist ===
for path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing folder: {path}")

# === Step 1: Pick 100 random folders from train ===
all_train_classes = sorted(os.listdir(train_dir))
selected_classes = sorted(random.sample(all_train_classes, 100))

# === Step 2: Delete non-selected folders from each split ===
def keep_only_selected_folders(split_path, allowed_folders):
    for folder in os.listdir(split_path):
        full_path = os.path.join(split_path, folder)
        if folder not in allowed_folders and os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"❌ Deleted: {full_path}")
        else:
            print(f"✅ Kept: {full_path}")

print("\n🔁 Cleaning TRAIN folders...")
keep_only_selected_folders(train_dir, selected_classes)

print("\n🔁 Cleaning VALIDATION folders...")
keep_only_selected_folders(val_dir, selected_classes)

print("\n🔁 Cleaning TEST folders...")
keep_only_selected_folders(test_dir, selected_classes)

# === Done ===
print("\n✅ Dataset cleaned and reduced to 100 consistent classes.")
print("🗂️ Sample of selected classes:", selected_classes[:10])

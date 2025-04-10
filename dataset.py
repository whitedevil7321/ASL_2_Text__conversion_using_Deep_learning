# dataset.py

import os
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir: str, label2id: Dict[str, int], transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label2id = label2id
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for label in os.listdir(self.root_dir):
            label_dir = self.root_dir / label
            if label_dir.is_dir():
                for file in label_dir.glob("*.mp4"):
                    samples.append((file, self.label2id[label]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _, _ = read_video(str(video_path), pts_unit='sec')
        video = video.permute(0, 3, 1, 2)
        if self.transform:
            video = self.transform(video)
        return {"video": video, "label": label}

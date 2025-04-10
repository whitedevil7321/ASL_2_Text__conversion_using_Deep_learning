# infer.py
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.io import read_video
from utils import get_transforms

class SignPredictor:
    def __init__(self, model_dir):
        self.model = VideoMAEForVideoClassification.from_pretrained(model_dir)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, video_path, id2label):
        video, _, _ = read_video(video_path, pts_unit="sec")
        video = video.permute(0, 3, 1, 2)
        transform = get_transforms(self.processor, mode="val")
        video = transform(video).permute(1, 0, 2, 3).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(pixel_values=video)
            predicted_class = outputs.logits.argmax(-1).item()
        return id2label[predicted_class]


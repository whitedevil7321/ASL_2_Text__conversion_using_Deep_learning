    

# export_predictions.py
import os
import csv
from dataset import SignLanguageDataset
from infer import SignPredictor
from utils import get_transforms
from transformers import VideoMAEImageProcessor

class BatchPredictor:
    def __init__(self, model_dir, label2id):
        self.id2label = {i: label for label, i in label2id.items()}
        self.processor = VideoMAEImageProcessor.from_pretrained(model_dir)
        self.predictor = SignPredictor(model_dir)

    def export_to_csv(self, test_dir, output_csv="predictions.csv"):
        dataset = SignLanguageDataset(test_dir, label2id, get_transforms(self.processor, mode="val"))
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Video", "TrueLabel", "PredictedLabel"])
            for sample in dataset.samples:
                path, true_label = sample
                true_label_name = self.id2label[true_label]
                pred = self.predictor.predict(str(path), self.id2label)
                writer.writerow([str(path), true_label_name, pred])
        print(f"✅ Predictions saved to {output_csv}")


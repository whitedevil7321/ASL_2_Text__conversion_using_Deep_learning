# train.py
import os
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
from dataset import SignLanguageDataset
from utils import get_transforms
import tensorflow as tf
from split_dataset import split_dataset
import evaluate
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from custom_callback import MetricsLoggerCallback  # ✅ Import custom callback

# === Split dataset if not already done ===
if not os.path.exists("D:/archive/dataset_split1/SL/train"):
    split_dataset("D:/archive/dataset_split1/SL", "D:/archive/dataset_split1/SL")

# === Generate label maps ===
labels = sorted(os.listdir("D:/archive/dataset_split1/SL/train"))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# === Model checkpoint ===
model_ckpt = "MCG-NJU/videomae-base"

# === Timestamped output dir ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"videomae-sign-{timestamp}"

# === Load model and processor ===
processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# === Transforms & Dataset ===
resize_to = processor.size.get("shortest_edge", 224)
train_transform = get_transforms(resize_to)
val_transform = get_transforms(resize_to)
train_dataset = SignLanguageDataset("D:/archive/dataset_split1/SL/train", label2id, train_transform)
val_dataset = SignLanguageDataset("D:/archive/dataset_split1/SL/val", label2id, val_transform)

# === Metric ===
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["video"] for example in examples])  # already (T, C, H, W)
    pixel_values = pixel_values.permute(0, 1, 2, 3, 4)  # (B, T, C, H, W)
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# === Training arguments ===
args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=20,
    logging_strategy="no",  # ✅ Turn off HuggingFace internal step logging
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# === Trainer ===
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[MetricsLoggerCallback()],  # ✅ Add custom callback
)

# === Train ===
print(f"🚀 Starting training for {args.num_train_epochs} epochs...")
with tf.device('/GPU:1'):
    train_results = trainer.train()
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

# === Confusion Matrix ===
preds = trainer.predict(val_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png")
print("✅ Training complete. Model and confusion matrix saved.")

# === Save labels ===
with open(f"{output_dir}/labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")
print(f"✅ Labels saved to {output_dir}/labels.txt")

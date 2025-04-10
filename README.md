<!-- # requirements.txt
transformers>=4.37.0
pytorchvideo>=0.1.5
torch>=1.13.0
torchvision>=0.14.0
evaluate>=0.4.0
gradio>=3.32.0
scikit-learn
matplotlib
numpy
opencv-python -->


# README.md
# Sign Language Classification using VideoMAE

This project fine-tunes the Hugging Face [VideoMAE model](https://huggingface.co/MCG-NJU/videomae-base) to classify word-level ASL (American Sign Language) videos.

---

## 📁 Project Structure
```
sign_language_videomae/
├── dataset.py               # Custom dataset class
├── utils.py                 # Data transforms
├── train.py                 # Model training and evaluation
├── split_dataset.py         # Script to split raw data into train/val/test
├── infer.py                 # Inference class for single video
├── predict_cli.py           # CLI prediction script
├── export_predictions.py    # Export test predictions to CSV
├── web_ui.py                # Gradio-based video classifier UI
├── requirements.txt         # Dependencies
├── README.md                # Instructions
└── data/                    # Auto-created train/val/test folders
```

---

## 📦 Installation
```bash
git clone <this-repo>
cd sign_language_videomae
pip install -r requirements.txt
```

---

## 🧾 Dataset Structure
Your raw dataset should be located at:
```
D:/Downloads/archive/dataset/SL/
```
And look like:
```
SL/
├── Hello/
├── Thanks/
├── ... (239 classes)
```

---

## 🚀 Training
```bash
python train.py
```
- Automatically splits dataset into train/val/test (80/10/10).
- Trains VideoMAE on your sign classes.
- Saves model, config, confusion matrix, and labels.txt in a timestamped folder.

---

## 🧪 Inference Options
### 1. Python API
```python
from infer import SignPredictor
predictor = SignPredictor("videomae-sign-YYYYMMDD-HHMMSS")
result = predictor.predict("path/to/video.mp4", id2label)
print(result)
```

### 2. CLI Tool
```bash
python predict_cli.py --video path/to/video.mp4 \
  --model videomae-sign-YYYYMMDD-HHMMSS \
  --labels "Hello,Thanks,Yes,..."
```

### 3. Web App
```bash
python web_ui.py
```
Opens a Gradio UI in your browser.

### 4. Export Test Predictions
```python
from export_predictions import BatchPredictor
predictor = BatchPredictor("videomae-sign-YYYYMMDD-HHMMSS", label2id)
predictor.export_to_csv("data/test")
```

---

## ✅ Output
After training, the model folder will contain:
```
videomae-sign-YYYYMMDD-HHMMSS/
├── pytorch_model.bin
├── config.json
├── preprocessor_config.json
├── labels.txt
├── confusion_matrix.png
```

---

## 📌 Notes
- Videos should be short (1–2 seconds, ~30–60 frames).
- Uses 16 uniformly sampled frames for training.
- No Mediapipe required — full end-to-end using raw video.

---

## 🙌 Credits
- [VideoMAE on Hugging Face](https://huggingface.co/MCG-NJU/videomae-base)
- [PyTorchVideo](https://pytorchvideo.org)
- Hugging Face Transformers + Gradio

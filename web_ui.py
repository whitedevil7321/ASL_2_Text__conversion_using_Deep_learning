# web_ui.py
import os
import gradio as gr
from infer import SignPredictor

def get_latest_model_dir(base_dir="."):
    candidates = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if d.startswith("videomae-sign-") and os.path.isdir(os.path.join(base_dir, d))
    ]
    latest = max(candidates, key=os.path.getmtime) if candidates else None
    return latest


# Load label map from model folder
def load_labels(model_dir):
    labels_txt = os.path.join(model_dir, "labels.txt")
    if os.path.exists(labels_txt):
        with open(labels_txt) as f:
            labels = [line.strip() for line in f.readlines()]
        return {i: label for i, label in enumerate(labels)}
    else:
        raise FileNotFoundError("labels.txt not found in model folder")

# Setup
model_dir = get_latest_model_dir()
if not model_dir:
    raise RuntimeError("No model folder found. Please train a model first.")

id2label = load_labels(model_dir)
predictor = SignPredictor(model_dir)

# Predict function
def predict_from_video(video):
    if video is None:
        return "No video uploaded."
    return predictor.predict(video, id2label)

# Interface
interface = gr.Interface(
    fn=predict_from_video,
    inputs=gr.Video(label="Upload Sign Video"),
    outputs=gr.Textbox(label="Predicted Sign"),
    title="ASL Sign Classifier (VideoMAE)",
    description="Upload a short video of a sign to get the predicted label."
)

if __name__ == "__main__":
    interface.launch()

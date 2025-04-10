# predict_cli.py
import argparse
from infer import SignPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--model", required=True, help="Path to the trained model directory")
    parser.add_argument("--labels", required=True, help="Comma-separated list of labels (e.g., Hello,Thanks,Yes)")
    args = parser.parse_args()

    id2label = {i: label for i, label in enumerate(args.labels.split(","))}
    predictor = SignPredictor(args.model)
    prediction = predictor.predict(args.video, id2label)
    print(f"🎯 Predicted Sign: {prediction}")


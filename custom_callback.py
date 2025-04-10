from transformers import TrainerCallback
import torch
import numpy as np

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None

    def on_init_end(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer")

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        logs = self.trainer.state.log_history[-1] if self.trainer.state.log_history else {}
        loss = logs.get("loss")

        if loss is not None:
            # Training metrics (per batch)
            print(f"📦 Step {state.global_step} | 🔻 Train Loss: {loss:.4f}")

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        # Validation metrics
        val_metrics = self.trainer.evaluate()
        val_loss = val_metrics.get("eval_loss", 0.0)
        val_acc = val_metrics.get("eval_accuracy", 0.0)

        # Training metrics
        train_metrics = self.trainer.evaluate(self.trainer.train_dataset)
        train_loss = train_metrics.get("eval_loss", 0.0)
        train_acc = train_metrics.get("eval_accuracy", 0.0)

        print(f"\n📊 Epoch {int(state.epoch)} Summary:")
        print(f"   🔹 Train Loss     : {train_loss:.4f}")
        print(f"   🔹 Train Accuracy : {train_acc:.4f}")
        print(f"   🔹 Val Loss       : {val_loss:.4f}")
        print(f"   🔹 Val Accuracy   : {val_acc:.4f}\n")

        return control

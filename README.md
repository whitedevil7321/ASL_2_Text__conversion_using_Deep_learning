# 🤟 ASL to Text Conversion using Deep Learning

This project implements **real-time American Sign Language (ASL) recognition** using deep learning video classification techniques. Built using the **VideoMAE** architecture, this system can detect and convert ASL gestures into text from video input — supporting multiple signs per video.

---

## 📂 Project Overview

- ✅ **Goal**: Recognize ASL signs from video clips and convert them into grammatically corrected text.
- 🎥 **Input**: Video clips of ASL gestures.
- 🧠 **Output**: Recognized sign text, optionally passed to an LLM for grammar correction.
- 🛠️ **Model**: [VideoMAE](https://arxiv.org/abs/2203.12602) – Masked Autoencoders for Video classification.
- 📦 **Dataset**: Filtered and augmented subset of a 86GB ASL video dataset (239 signs).

---

## 📊 Dataset Preparation

- Original dataset: ~20GB with 6–7 videos per sign.
- Selected **239 key signs** for focused training.
- Used `VideoAug` library for augmentation.
- **Class balancing**:
  - `80` videos for training
  - `10` for validation
  - `10` for testing per sign
- Applied `UniversalTemporalSubsampling` to extract **16 frames per video** (required by VideoMAE).

---

## 🧠 Model Details

- **Architecture**: VideoMAE  
- **Checkpoint Used**: `MCG-NJU/videomae-base`  
- **Frameworks**: PyTorch + Hugging Face Transformers  
- **Preprocessing**:
  - Removed rotation (ASL is performed upright)
  - Fixed zoom levels across all videos
  - Subsampled videos to uniform frame length (16)

---

## 🖥️ Training Setup

- Training on local laptops was impractical (~1 year estimated time).
- Received **GPU support** from **Prof. Elham Buxton** to accelerate training.
- Final model trained with **98% accuracy**.
- Supports **long-form video inference** with **multi-sign prediction**.

---

## 🛠️ Installation

```bash
git clone https://github.com/whitedevil7321/ASL_2_Text__conversion_using_Deep_learning.git
cd ASL_2_Text__conversion_using_Deep_learning
pip install -r requirements.txt

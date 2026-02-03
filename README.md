# A Confidence-Gated Hybrid LSTM-LLM Framework for Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Ollama](https://img.shields.io/badge/Backend-Ollama-black)](https://ollama.com/)

This repository contains the official implementation of the paper: **"A Confidence-Gated Hybrid LSTM-LLM Framework for Intrusion Detection under Cross-Class Distribution Shift"**.

We propose a hybrid intrusion detection system (IDS) that integrates the speed of **LSTMs** (Long Short-Term Memory) with the reasoning capabilities of **LLMs** (Large Language Models) to detect Zero-Day attacks effectively.

---

## 🚀 Overview

Deep Learning models (like LSTMs) often fail when testing data differs significantly from training data (Distribution Shift).
* **The Problem:** An LSTM trained only on **DoS Attacks** achieves near-zero recall (~0.02) when facing unknown **Web Attacks**.
* **The Solution:** A "Fast & Slow" system.
    * **System 1 (LSTM):** Handles 90% of traffic. High confidence predictions are accepted immediately.
    * **System 2 (LLM):** Activated **only** when the LSTM is uncertain ($0.3 < P_{attack} < 0.7$). The LLM analyzes the flow's semantic features (Port, Duration, Size) to make a final verdict.

![Framework Architecture](results/flow_chart.pdf)
*(Note: Please convert the flow_chart.pdf to .png for better GitHub rendering)*

---

## 📂 Project Structure

```text
├── data/                   # Place CIC-IDS-2017 CSV files here
├── results/                # Generated plots, confusion matrices, and logs
│   ├── zeroday_training_curve.png
│   ├── Zero-Day_Confusion_Matrix.png
│   └── 2_Summary_Comparison.png
├── LSTM.py                 # Stage 1: Train the LSTM baseline (DoS -> Web Attack split)
├── model.py                # Stage 2: Hybrid Inference (LSTM + Confidence Gating + Ollama)
├── requirements.txt        # Python dependencies
└── README.md               # This file

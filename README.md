# Filter-Bank-Analysis-for-Automatic-Birdsong-Classification

Automated system for real-time bird species recognition using subband filter bank analysis, spectral energy feature extraction, and distance-based classification in Python.

---

## 📌 Project Overview

This project implements a Digital Signal Processing (DSP) pipeline designed to classify bird songs into three distinct species[cite: 2]. The system operates in two main phases:

1. **Training / Calibration Phase:** Analyzes pre-recorded 2-second audio samples ($f_s = 44.1 \text{ kHz}$) to construct reference energy profiles across 3 subbands for each species[cite: 1, 2].
2. **Real-Time Classification Phase:** Captures 2 seconds of live audio from a microphone, computes its spectral energy vector, and determines the species using minimum absolute/Euclidean distance matching against the reference bank[cite: 1, 2].

---

## 🛠️ System Architecture & Methodology

1. **Audio Acquisition:** Captures 2-second audio buffers sampled at $44,100 \text{ Hz}$.
2. **Spectral Transformation:** Converts time-domain signals to the frequency domain using the Fast Fourier Transform ($\text{FFT}$) and computes their magnitude spectrum[cite: 1].
3. **Filter Bank Partitioning:** Splits the frequency spectrum into **3 subbands**[cite: 1, 2].
4. **Feature Extraction:** Calculates the average energy for each subband to build a 3-dimensional feature vector ($F = [E_1, E_2, E_3]$)[cite: 1].
5. **Pattern Matching & Classification:** Evaluates the input feature vector against the species database using minimum absolute error / Euclidean distance metrics[cite: 1].

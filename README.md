### ParkinSeen

This repository contains a Python-based project for detecting Parkinson's Disease (PD) using machine learning models. The project includes two tests: a video test and an audio test.

**Video Test**: Subjects spin around in front of a camera while their left ankle coordinates are tracked using MediaPipe. Frames are marked as having Freezing of Gait (FOG) or not.
**Audio Test:** Subjects say "aaa" for a few seconds, and vocal biomarkers are extracted using Librosa.
The machine learning models are trained on datasets consisting of recordings from PD patients and healthy individuals.

# Features:

Flask web interface for conducting tests and displaying results.
Machine learning models for classification.
Visualization tools for data analysis.

# Requirements:

Python 3.x
Libraries: Flask, MediaPipe, Librosa, NumPy, pandas, scikit-learn, matplotlib, etc.

# Usage:

Clone the repository.
Install dependencies using pip install -r requirements.txt.
Run the Flask app with python app.py.

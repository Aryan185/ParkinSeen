# ParkinSeen

This project aims to detect Parkinson's Disease (PD) through two types of tests: a video-based test and an audio-based test. The video test involves tracking ankle movement while the subject spins around, and the audio test analyzes vocal biomarkers.

**Video Test**: Subjects spin around in front of a camera while their left ankle coordinates are tracked using MediaPipe. Frames are marked as having Freezing of Gait (FOG) or not.

**Audio Test:** Subjects say "aaa" for a few seconds, and vocal biomarkers are extracted using Librosa.
The machine learning models are trained on datasets consisting of recordings from PD patients and healthy individuals.

### Features:

*  Video Test: The subject spins around in front of a camera, and their left ankle coordinates are tracked using MediaPipe. Frames are marked as having Freezing of Gait (FOG) or not. The dataset used for training consists of recordings of 73 individuals, where FOG is labeled as 1 and no FOG as 0.

*  Audio Test: The subject is asked to say "aaa" for a few seconds. Vocal biomarkers are extracted using Librosa. The dataset contains recordings of 81 individuals, 40 with PD and 41 healthy.

*  Machine Learning Models: Machine learning models are trained on the collected datasets to classify whether a subject has Parkinson's Disease or not.

*  Flask Web Interface: The project includes a Flask web application that provides an intuitive interface for conducting the tests and viewing the results.

### Requirements:

*  Python 3.x
*  Flask
*  MediaPipe
*  Librosa
*  NumPy
*  pandas
*  scikit-learn
*  matplotlib

### Usage:

1. Clone the repository:
```
git clone https://github.com/Aryan185/ParkinSeen.git
cd ParkinSeen
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the Flask app:
```
python app.py
```


### Contributing
Contributions and suggestions are welcome! Feel free to submit issues and pull requests.

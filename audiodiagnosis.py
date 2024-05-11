import pandas as pd
import numpy as np
import crepe
import librosa
import csv
import re
import pickle


def compute_jitter(y, sr):
    _, frequency, _, _ = crepe.predict(y, sr, viterbi=False)
    jitter = np.mean(np.abs(np.diff(frequency))) / np.mean(frequency) * 100
    return jitter


def compute_shimmer(y, sr):
    _, frequency, _, _ = crepe.predict(y, sr, viterbi=False)

    shimmer = np.std(y)
    return shimmer


def compute_absolute_jitter(y, sr):
    period = 1 / librosa.feature.zero_crossing_rate(y)[0]
    absolute_jitter = np.mean(np.abs(np.diff(period))) * 1000
    return absolute_jitter


def diagnose_audio(audio_file):
    features = {}
    y, sr = librosa.load(audio_file, sr=None)

    features['MDVP:Fo(Hz)'] = np.mean(librosa.effects.harmonic(y))
    features['MDVP:Fhi(Hz)'] = np.max(librosa.effects.harmonic(y))
    features['MDVP:Flo(Hz)'] = np.min(librosa.effects.harmonic(y))
    jitter = compute_jitter(y, sr)
    features['MDVP:Jitter(%)'] = jitter
    rms = np.sqrt(np.mean(y ** 2))
    features['MDVP:RAP'] = rms
    features['MDVP:PPQ'] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    features['Jitter:DDP'] = rms ** 2
    shimmer = compute_shimmer(y, sr)
    features['MDVP:Shimmer'] = shimmer
    features['MDVP:Shimmer'] = np.mean(shimmer)
    features['Shimmer:APQ3'] = np.mean(librosa.effects.trim(y)[1])
    features['Shimmer:APQ5'] = np.mean(librosa.effects.trim(y)[1])
    features['MDVP:APQ'] = np.mean(librosa.effects.trim(y)[1])
    features['Shimmer:DDA'] = np.mean(librosa.effects.trim(y)[1])
    features['NHR'] = librosa.effects.split(y)
    features['HNR'] = librosa.effects.split(y)
    features['RPDE'] = librosa.effects.split(y)
    features['DFA'] = librosa.effects.split(y)
    features['spread1'] = librosa.effects.split(y)
    features['spread2'] = librosa.effects.split(y)
    features['D2'] = librosa.effects.split(y)
    features['PPE'] = librosa.effects.split(y)

    csv_file = "static/csv/audio_extracted_features.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=features.keys())
        writer.writeheader()
        writer.writerow(features)

    def clean_csv(input_file, output_file):
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                cleaned_row = []
                for cell in row:
                    cleaned_cell = cell.strip("[[    ").strip("]]")
                    cleaned_cell = re.sub(r'(\d+)\s(\d+)', r'\1.\2', cleaned_cell)
                    cleaned_row.append(cleaned_cell)
                writer.writerow(cleaned_row)

    input_file = 'static/csv/audio_extracted_features.csv'
    output_file = 'static/csv/output.csv'
    clean_csv(input_file, output_file)

    test = pd.read_csv('static/csv/output.csv')
    test.describe().transpose()
    pickled = pickle.load(open('static/model/audiomodel.pkl', 'rb'))
    test_pred = pickled.predict(test)
    if test_pred == 1:
        return "You are diagnosed with Parkinson's"
    else:
        return "You do not have Parkinson's"

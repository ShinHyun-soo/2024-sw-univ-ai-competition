#skiprow 랑 3_1,2,3,4,... 다 바꿔야해요.. 81행이랑 94행

import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import os
import random
from tqdm import tqdm

# Load the CSV file
csv_path = './train.csv'
data = pd.read_csv(csv_path, skiprows=range(2001, 3001), nrows=1000)

# Create output directory
output_dir = './audio_final'
os.makedirs(output_dir, exist_ok=True)

# Prepare to store the results
results = []

# Filter files by label
label_0_files = data[data['label'] == 0]['path'].tolist()
label_1_files = data[data['label'] == 1]['path'].tolist()

# Calculate lengths of all files with tqdm progress bar
file_lengths = {}
for file_path in tqdm(label_0_files + label_1_files, desc="Calculating lengths"):
    try:
        y, sr = librosa.load(file_path, sr=None)
        file_length = len(y)
        file_lengths[file_path] = (file_length, sr)
    except Exception as e:
        print(f"Could not process file {file_path}: {e}")

# Find pairs with tqdm progress bar
pairs = []
used_files = set()
for file1 in tqdm(label_0_files, desc="Finding pairs"):
    if file1 in file_lengths and file1 not in used_files:
        for file2 in label_1_files:
            if file2 in file_lengths and file2 not in used_files:
                pairs.append((file1, file2))
                used_files.add(file1)
                used_files.add(file2)
                break  # Stop after finding the first match for this file1

# Process the audio files 4 times with tqdm progress bar
for repetition in range(1):
    for i, (file1_path, file2_path) in tqdm(enumerate(pairs), total=len(pairs), desc=f"Processing pairs repetition {repetition+1}"):
        # Check if files exist
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Skipping pair ({file1_path}, {file2_path}) due to missing file")
            continue

        # Load audio files
        y1, sr1 = librosa.load(file1_path, sr=None)
        y2, sr2 = librosa.load(file2_path, sr=None)

        # Ensure both audio files have the same sampling rate
        if sr1 != sr2:
            print(f"Skipping pair ({file1_path}, {file2_path}) due to sampling rate mismatch")
            continue

        # Make both audio files the same length by padding the shorter one with zeros
        if len(y1) > len(y2):
            y2 = np.pad(y2, (0, len(y1) - len(y2)), mode='constant')
        elif len(y2) > len(y1):
            y1 = np.pad(y1, (0, len(y2) - len(y1)), mode='constant')

        # Generate a random mixing ratio
        ratio = random.random()

        # Mix the audio files
        mixed = ratio * y1 + (1 - ratio) * y2

        # Normalize the mixed signal
        mixed = mixed / np.max(np.abs(mixed))

        # Save the mixed audio file
        output_file = os.path.join(output_dir, f'3_4_{i}.ogg')
        sf.write(output_file, mixed, sr1)

        # Store the result
        results.append({
            'path': output_file,
            'label': '3',
        })

# Create a new dataframe with the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv_path = './3_4.csv'
results_df.to_csv(output_csv_path, index=False)

print(f"Mixed audio files and ratios saved to {output_csv_path}")

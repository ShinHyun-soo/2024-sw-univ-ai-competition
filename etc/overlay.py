import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import os
import random
from tqdm import tqdm

# Load the CSV file
csv_path = './train.csv'
data = pd.read_csv(csv_path)

# Create output directory
output_dir = './overlay'
os.makedirs(output_dir, exist_ok=True)

# Prepare to store the results
results = []

# Filter files by label
label_0_files = data[data['label'] == 1]['path'].tolist()

# Create a dictionary to store file lengths
file_lengths = {}

# Calculate lengths of all files with tqdm progress bar
for file_path in tqdm(label_0_files, desc="Calculating lengths"):
    try:
        y, sr = librosa.load(file_path, sr=None)
        file_length = len(y)
        file_lengths[file_path] = file_length
    except Exception as e:
        print(f"Could not process file {file_path}: {e}")

# Find pairs with the same length with tqdm progress bar
pairs = []
used_files = set()

for file1 in tqdm(label_0_files, desc="Finding pairs"):
    if file1 in file_lengths and file1 not in used_files:
        length1 = file_lengths[file1]
        for file2 in label_0_files:
            if file2 in file_lengths and file2 not in used_files and file2 != file1 and file_lengths[file2] == length1:
                pairs.append((file1, file2))
                used_files.add(file1)
                used_files.add(file2)
                break  # Stop after finding the first match for this file1

# Process the audio files 4 times with tqdm progress bar
for repetition in range(2):
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

        # Generate a random mixing ratio
        ratio = random.random()

        # Mix the audio files
        mixed = ratio * y1 + (1 - ratio) * y2

        # Normalize the mixed signal
        mixed = mixed / np.max(np.abs(mixed))

        # Save the mixed audio file
        output_file = os.path.join(output_dir, f'real_mixed3_{repetition}_{i}.ogg')
        sf.write(output_file, mixed, sr1)

        # Store the result
        results.append({
            'path': output_file,
            'label': '1',
        })

# Create a new dataframe with the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv_path = './real_mixed2.csv'
results_df.to_csv(output_csv_path, index=False)

print(f"Mixed audio files and ratios saved to {output_csv_path}")

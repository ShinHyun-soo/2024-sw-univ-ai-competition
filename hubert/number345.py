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
output_dir = './overlay2'
os.makedirs(output_dir, exist_ok=True)

# Prepare to store the results
results = []

# Filter files by label
label_0_files = data[data['label'] == 0]['path'].tolist()
label_1_files = data[data['label'] == 1]['path'].tolist()

# Create a dictionary to store file lengths
file_lengths = {}

# Calculate lengths of all files with tqdm progress bar
all_files = label_0_files + label_1_files
for file_path in tqdm(all_files, desc="Calculating lengths"):
    try:
        y, sr = librosa.load(file_path, sr=None)
        file_length = len(y)
        file_lengths[file_path] = file_length
    except Exception as e:
        print(f"Could not process file {file_path}: {e}")

# Function to pad audio to match lengths
def pad_audio(y, target_length):
    if len(y) < target_length:
        padding = np.zeros(target_length - len(y))
        y = np.concatenate((y, padding))
    return y

# Function to find pairs with different lengths
def find_pairs(files1, files2, target_count):
    pairs = []
    used_files = set()

    for file1 in files1:
        if file1 in file_lengths and file1 not in used_files:
            for file2 in files2:
                if file2 in file_lengths and file2 not in used_files and file2 != file1:
                    pairs.append((file1, file2))
                    used_files.add(file1)
                    used_files.add(file2)
                    if len(pairs) >= target_count:
                        return pairs
    return pairs

# Find pairs for each combination
target_count = 15000
pairs_0_1 = find_pairs(label_0_files, label_1_files, 30000)
pairs_1_1 = find_pairs(label_1_files, label_1_files, 15000)
pairs_0_0 = find_pairs(label_0_files, label_0_files, 15000)

# Combine all pairs into one list and shuffle
all_pairs = pairs_0_0 + pairs_1_1 + pairs_0_1
random.shuffle(all_pairs)

# Process the audio files with tqdm progress bar
def process_pairs(pairs, label, repetition):
    for i, (file1_path, file2_path) in tqdm(enumerate(pairs), total=len(pairs), desc=f"Processing pairs {label} repetition {repetition+1}"):
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

        # Adjust lengths with padding
        target_length = max(len(y1), len(y2))
        y1 = pad_audio(y1, target_length)
        y2 = pad_audio(y2, target_length)

        # Generate a random mixing ratio
        ratio = random.random()

        # Mix the audio files
        mixed = ratio * y1 + (1 - ratio) * y2

        # Normalize the mixed signal
        mixed = mixed / np.max(np.abs(mixed))

        # Save the mixed audio file
        output_file = os.path.join(output_dir, f'mixed_{label}_{repetition}_{i}.ogg')
        sf.write(output_file, mixed, sr1)

        # Store the result
        results.append({
            'path': output_file,
            'label': label,
        })

# Process pairs for each label combination
process_pairs(pairs_0_1, 3, 0)
process_pairs(pairs_1_1, 4, 0)
process_pairs(pairs_0_0, 5, 0)

# Create a new dataframe with the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv_path = './real_mixed3.csv'
results_df.to_csv(output_csv_path, index=False)

print(f"Mixed audio files and ratios saved to {output_csv_path}")

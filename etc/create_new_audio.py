import pandas as pd
import numpy as np
import soundfile as sf
import os
import random
from tqdm import tqdm

# Create output directory
output_dir = './overlay'
os.makedirs(output_dir, exist_ok=True)

# Prepare to store the results
results = []

# Sampling rate
sr = 16000

# Number of random audio files to create
num_files = 10000

# Process the creation of random audio files with tqdm progress bar
for i in tqdm(range(num_files), desc="Creating random audio files"):
    # Generate a random duration between 1 and 5 seconds
    duration = random.uniform(1, 5)
    num_samples = int(duration * sr)

    # Generate random white noise
    y = np.random.uniform(-1, 1, num_samples)

    # Normalize the audio
    y = y / np.max(np.abs(y))

    # Save the new audio file
    output_file = os.path.join(output_dir, f'random_clip_{i}.ogg')
    sf.write(output_file, y, sr)

    # Store the result
    results.append({
        'path': output_file,
        'label': '3',
    })

# Create a new dataframe with the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv_path = './random_audio_clips_results.csv'
results_df.to_csv(output_csv_path, index=False)

print(f"Random audio clips and labels saved to {output_csv_path}")

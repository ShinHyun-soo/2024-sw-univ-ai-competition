import pandas as pd

# Load the CSV file
file_path = './combined_train_aug_fixed_1.csv'
df = pd.read_csv(file_path)

shuffled_df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataframe to a new CSV file
shuffled_file_path = './train_final.csv'
shuffled_df.to_csv(shuffled_file_path, index=False)

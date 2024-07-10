import pandas as pd

# Load the CSV files
file_1_path = './submission/wavlm_deepfake_fold_2_pure_datset_val_interval_1_epoch0.csv'
file_2_path = './submission/35_fold_5_multi_label.csv'
file_3_path = './submission/wav2vec_deepfake_fold_2_pure_datset_val_interval_1_epoch0.csv'
file_4_path = './submission/fold_5_multi_label_with_auc.csv'
file_5_path = './submission/fold_5_0.csv'
file_6_path = './submission/fold_20.csv'
file_7_path = './submission/wav2vec_deepfake_fold_2_pure_datset_val_interval_1_epoch0_fake_only.csv'

df1 = pd.read_csv(file_1_path)
df2 = pd.read_csv(file_2_path)
df3 = pd.read_csv(file_3_path)
df4 = pd.read_csv(file_4_path)
df5 = pd.read_csv(file_5_path)
df6 = pd.read_csv(file_6_path)
df7 = pd.read_csv(file_7_path)

# Merge the four dataframes on the 'id' column
merged_df = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))
merged_df = pd.merge(merged_df, df3, on='id', suffixes=('_2', '_3'))
merged_df = pd.merge(merged_df, df4, on='id', suffixes=('_3', '_4'))
merged_df = pd.merge(merged_df, df5, on='id', suffixes=('_4', '_5'))
merged_df = pd.merge(merged_df, df6, on='id', suffixes=('_5', '_6'))
merged_df = pd.merge(merged_df, df7, on='id', suffixes=('_6', '_7'))
# Print column names of merged dataframe to debug
print("merged_df columns:", merged_df.columns)

# Calculate the average of the 'fake' and 'real' columns from all four dataframes
merged_df['fake'] = (merged_df['fake_1'] + merged_df['fake_2'] + merged_df['fake_3'] + merged_df['fake_4'] + merged_df['fake_5']+ merged_df['fake_6']+ merged_df['fake']) / 7
merged_df['real'] = (merged_df['real_1'] + merged_df['real_2'] + merged_df['real_3'] + merged_df['real_4'] + merged_df['real_5']+ merged_df['real_6']+ merged_df['real']) / 7

# Select relevant columns to display
result_df = merged_df[['id', 'fake', 'real']]

result_df.to_csv('assemble_V10.csv', index=False)

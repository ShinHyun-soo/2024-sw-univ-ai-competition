import pandas as pd

# 본인 파일명에 맞게 수정
file_1_path = './submission/hubert_bf16_fold_1.csv'
file_2_path = './submission/hubert_bf16_fold_2.csv'
file_3_path = './submission/hubert_bf16_fold_3.csv'
file_4_path = './submission/hubert_bf16_fold_4.csv'
file_5_path = './submission/hubert_bf16_fold_5.csv'
file_6_path = './submission/hubert_bf16_fold_6.csv'
file_7_path = './submission/hubert_bf16_fold_7.csv'
file_8_path = './submission/hubert_bf16_fold_8.csv'
file_9_path = './submission/hubert_bf16_fold_9.csv'
file_10_path = './submission/hubert_bf16_fold_10.csv'
file_11_path = './submission/hubert_bf16_fold_11.csv'
file_12_path = './submission/hubert_bf16_fold_12.csv'
file_13_path = './submission/hubert_bf16_fold_13.csv'
file_14_path = './submission/hubert_bf16_fold_14.csv'
file_15_path = './submission/hubert_bf16_fold_15.csv'
file_16_path = './submission/hubert_bf16_fold_16.csv'
file_17_path = './submission/hubert_bf16_fold_17.csv'
file_18_path = './submission/hubert_bf16_fold_18.csv'
file_19_path = './submission/hubert_bf16_fold_19.csv'
file_20_path = './submission/hubert_bf16_fold_0.csv'



df1 = pd.read_csv(file_1_path)
df2 = pd.read_csv(file_2_path)
df3 = pd.read_csv(file_3_path)
df4 = pd.read_csv(file_4_path)
df5 = pd.read_csv(file_5_path)
df6 = pd.read_csv(file_6_path)
df7 = pd.read_csv(file_7_path)
df8 = pd.read_csv(file_8_path)
df9 = pd.read_csv(file_9_path)
df10 = pd.read_csv(file_10_path)
df11 = pd.read_csv(file_11_path)
df12 = pd.read_csv(file_12_path)
df13 = pd.read_csv(file_13_path)
df14 = pd.read_csv(file_14_path)
df15 = pd.read_csv(file_15_path)
df16 = pd.read_csv(file_16_path)
df17 = pd.read_csv(file_17_path)
df18 = pd.read_csv(file_18_path)
df19 = pd.read_csv(file_19_path)
df20 = pd.read_csv(file_20_path)

# Merge the four dataframes on the 'id' column
merged_df = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))
merged_df = pd.merge(merged_df, df3, on='id', suffixes=('_2', '_3'))
merged_df = pd.merge(merged_df, df4, on='id', suffixes=('_3', '_4'))
merged_df = pd.merge(merged_df, df5, on='id', suffixes=('_4', '_5'))
merged_df = pd.merge(merged_df, df6, on='id', suffixes=('_5', '_6'))
merged_df = pd.merge(merged_df, df7, on='id', suffixes=('_6', '_7'))
merged_df = pd.merge(merged_df, df8, on='id', suffixes=('_7', '_8'))
merged_df = pd.merge(merged_df, df9, on='id', suffixes=('_8', '_9'))
merged_df = pd.merge(merged_df, df10, on='id', suffixes=('_9', '_10'))
merged_df = pd.merge(merged_df, df11, on='id', suffixes=('_10', '_11'))
merged_df = pd.merge(merged_df, df12, on='id', suffixes=('_11', '_12'))
merged_df = pd.merge(merged_df, df13, on='id', suffixes=('_12', '_13'))
merged_df = pd.merge(merged_df, df14, on='id', suffixes=('_13', '_14'))
merged_df = pd.merge(merged_df, df15, on='id', suffixes=('_14', '_15'))
merged_df = pd.merge(merged_df, df16, on='id', suffixes=('_15', '_16'))
merged_df = pd.merge(merged_df, df17, on='id', suffixes=('_16', '_17'))
merged_df = pd.merge(merged_df, df18, on='id', suffixes=('_17', '_18'))
merged_df = pd.merge(merged_df, df19, on='id', suffixes=('_18', '_19'))
merged_df = pd.merge(merged_df, df20, on='id', suffixes=('_19', '_20'))
# Print column names of merged dataframe to debug
print("merged_df columns:", merged_df.columns)

# Calculate the average of the 'fake' and 'real' columns from all four dataframes
# 안될수도 있는데 merged_df columns 보고 fake_20 이 fake 면 fake 로 바꾸면됨.
merged_df['fake'] = (merged_df['fake_1'] + merged_df['fake_2'] + merged_df['fake_3'] + merged_df['fake_4'] + merged_df['fake_5'] + merged_df['fake_6'] + merged_df['fake_7'] + merged_df['fake_8'] + merged_df['fake_9'] + merged_df['fake_10'] + merged_df['fake_11'] + merged_df['fake_12'] + merged_df['fake_13'] + merged_df['fake_14'] + merged_df['fake_15'] + merged_df['fake_16'] + merged_df['fake_17'] + merged_df['fake_18'] + merged_df['fake_19'] + merged_df['fake_20']) / 20
merged_df['real'] = (merged_df['real_1'] + merged_df['real_2'] + merged_df['real_3'] + merged_df['real_4'] + merged_df['real_5'] + merged_df['real_6'] + merged_df['real_7'] + merged_df['real_8'] + merged_df['real_9'] + merged_df['real_10'] + merged_df['real_11'] + merged_df['real_12'] + merged_df['real_13'] + merged_df['real_14'] + merged_df['real_15'] + merged_df['real_16'] + merged_df['real_17'] + merged_df['real_18'] + merged_df['real_19'] + merged_df['real_20']) / 20

# Select relevant columns to display
result_df = merged_df[['id', 'fake', 'real']]

result_df.to_csv('assemble_V10.csv', index=False)

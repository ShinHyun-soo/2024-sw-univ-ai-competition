import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('./data/train.csv')

# 문자열을 수정하고자 하는 열의 이름 (예: 'column_name')
column_name = 'path'
additional_string = './data'

# 해당 열의 모든 데이터에서 마침표 제거하고 문자열 추가
df[column_name] = additional_string + df[column_name].str.replace('.', '')

# 수정된 DataFrame을 새로운 CSV 파일로 저장
df.to_csv('your_modified_file2.csv', index=False)


# 이거 하면 .ogg가 ogg 될텐데 엑셀에서 컨트롤 에프 하고 ogg 찾은다음에 .ogg 모두 바꾸기 하면 됩니다

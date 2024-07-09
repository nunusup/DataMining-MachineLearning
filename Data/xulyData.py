import re
import pandas as pd

# Đọc dữ liệu từ DataFrame
data = pd.read_csv('D:\\MachineLearning\\DMML_Kmeans\\Data\\data_goodreads.csv', dtype={'Avg_Rating': 'str'})

# Loại bỏ các dòng bị trùng lặp
data = data.drop_duplicates(subset=['Book'])
data = data.drop_duplicates(subset=['Description'])

# Xóa các cột không cần thiết
data = data.drop(columns=['Unnamed: 0', 'Description', 'URL'])

# Kiểm tra và loại bỏ các dòng có dữ liệu là []
data = data[~data['Genres'].str.contains(r'\[\]', na=False)]
data = data[~data['Book'].str.contains('#', na=False)]

# Hàm để loại bỏ square brackets và quotes từ dữ liệu trong cột Genres
def clean_and_format_genres(genres_str):
    genres_cleaned = re.sub(r"[\[\]']", "", genres_str)
    return genres_cleaned

# Áp dụng hàm clean_and_format_genres cho cột Genres
data['Genres'] = data['Genres'].apply(clean_and_format_genres)

# Kiểm tra dữ liệu thiếu trong dataframe
missing_data = data.isnull().sum()
print("Dữ liệu thiếu:\n", missing_data)
print("----------------------:\n")

# Kiểm tra dữ liệu trùng lặp trong dataframe
duplicate_rows = data[data.duplicated()]
print("Dữ liệu trùng lặp:\n", duplicate_rows)
print("----------------------:\n")

# Đếm số dòng trong cột 'genres' có giá trị là ''
num_rows_with_empty_genres = len(data[data['Genres'] == ''])
print("Số dòng trong cột 'Genres' có giá trị là '' là:", num_rows_with_empty_genres)
print("----------------------:\n")

# Lưu trữ dữ liệu tiền xử lý vào tệp mới (nếu cần)
#data.to_csv('D:\\BTL khai phá\\khaiPha\\du_lieu_da_tien_xu_ly_goodreads_data.csv', index=False)

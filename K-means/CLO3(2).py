# import pandas as pd

# # Đọc dữ liệu từ tệp CSV
# data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned2.csv")

# # Kiểm tra kiểu dữ liệu của cột "Avg_Rating" và "Diem"
# print(data['Avg_Rating'].dtype)
# print(data['Diem'].dtype)

# # Nếu cột chứa dữ liệu kiểu object, chúng tôi sẽ chuyển đổi nó thành dạng số
# for col in ['Avg_Rating', 'Diem']:
#     if data[col].dtype == 'object':
#         data[col] = pd.to_numeric(data[col], errors='coerce')

# # Tính trung bình của hai cột "Avg_Rating" và "Diem"
# data['TestFeature'] = data[['Avg_Rating', 'Diem']].mean(axis=1)

# # Hiển thị 5 dòng đầu tiên của dữ liệu đã được cập nhật
# print(data.head())

import pandas as pd

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned2.csv")

# Chuyển đổi kiểu dữ liệu của các cột "Avg_Rating" và "Diem" sang số
data['Avg_Rating'] = pd.to_numeric(data['Avg_Rating'], errors='coerce')
data['Diem'] = pd.to_numeric(data['Diem'], errors='coerce')

# Tính trung bình của hai cột "Avg_Rating" và "Diem"
data['TestFeature'] = data[['Avg_Rating', 'Diem']].mean(axis=1)

# In cột "Avg_Rating"
print(data['Avg_Rating'])

# In ra các giá trị duy nhất trong cột "Diem"
print(data['Diem'].unique())

# Kiểm tra kiểu dữ liệu của cột "Diem"
print(data['Diem'].dtype)

# Hiển thị 5 dòng đầu tiên của dữ liệu đã được cập nhật
print(data.head())

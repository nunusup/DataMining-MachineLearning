import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Đọc dữ liệu từ file CSV
data = pd.read_csv('train_data.csv')

# Tính trung bình cộng của thuộc tính thứ 2 và 3
data['Mean'] = (data['DomesticOpening'] + data['DomesticSales']) / 2

# Khởi tạo MinMaxScaler
scaler = MinMaxScaler()

# Số hóa đặc trưng thứ 2 và 3 và ánh xạ vào khoảng [0, 1]
# data['Title'] = scaler.fit_transform(data['Title'].values.reshape(-1, 1))
data['Year'] = scaler.fit_transform(data['Year'].values.reshape(-1, 1))

# Đọc dữ liệu từ file CSV
# data = pd.read_csv('test_data.csv')

# In ra dữ liệu sau khi thêm thuộc tính mới
print(data)

# Lưu dữ liệu vào file CSV mới
data.to_csv('new_dataT.csv', index=False)
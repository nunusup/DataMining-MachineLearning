import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Đọc dữ liệu
data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned2.csv")

# Kiểm tra kiểu dữ liệu của cột "Avg_Rating" và "Diem"
print(data['Avg_Rating'].dtype)
print(data['Diem'].dtype)

# Nếu cột chứa dữ liệu kiểu object, chúng tôi sẽ chuyển đổi nó thành dạng số
for col in ['Avg_Rating', 'Diem']:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Tiếp tục tính trung bình của hai cột "Avg_Rating" và "Diem"
data['TestFeature'] = data[['Avg_Rating', 'Diem']].mean(axis=1)

# Loại bỏ cột không cần thiết
data = data.drop(columns=['Author'])

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data[['Avg_Rating', 'Diem']] = scaler.fit_transform(data[['Avg_Rating', 'Diem']])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

# Huấn luyện model với 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)

# Dự đoán nhãn trên tập kiểm tra
predicted_labels = kmeans.predict(X_test)

# Đánh giá hiệu suất của mô hình
silhouette_avg = silhouette_score(X_test, predicted_labels)
print("Silhouette Score:", silhouette_avg)

# Vẽ biểu đồ KMeans Clustering
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10, 6))
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.style.use("fivethirtyeight")
plt.xlabel('Avg_Rating')
plt.ylabel('Diem')
plt.title('KMeans Clustering')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
print(sys.stdout.encoding)

# Đọc dữ liệu
data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned.csv")

# Chuẩn bị dữ liệu
X = data['Genres']  # Đặc trưng: Thể loại sách
y = data['Book']    # Nhãn: Tên sách

# Mã hóa dữ liệu
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)

# Huấn luyện mô hình
kmeans = KMeans(n_clusters=5, random_state=50)
kmeans.fit(X_encoded)

# Lưu mô hình
# joblib.dump(kmeans, "kmeans_model_old.joblib")

# Hiển thị sách trong tất cả các cụm
def display_books_in_clusters():
    clusters = {}
    for i in range(kmeans.n_clusters):
        books_in_cluster = data[kmeans.labels_ == i]['Book'].tolist()
        clusters[i] = books_in_cluster
    return clusters

# Hiển thị các cụm sách
def display_all_clusters():
    clusters = display_books_in_clusters()
    for cluster, books in clusters.items():
        print(f"Cum {cluster}:")
        print(books)

# Sử dụng hàm để hiển thị tất cả các cụm
display_all_clusters()

# Inertia:
# Inertia là tổng bình phương khoảng cách của mỗi mẫu đến trung tâm của cụm gần nhất của nó.
print("Inertia:", kmeans.inertia_)

# Silhouette Score:
# Silhouette score đo độ tương đồng của các điểm trong cùng một cụm và độ khác biệt giữa các cụm khác nhau.
silhouette_avg = silhouette_score(X_encoded, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Davies-Bouldin Index:
# Davies-Bouldin Index đánh giá sự tách biệt giữa các cụm.
davies_bouldin = davies_bouldin_score(X_encoded, kmeans.labels_)
print("Davies-Bouldin Index:", davies_bouldin)

# Calinski-Harabasz Index:
# Calinski-Harabasz Index đo sự tách biệt giữa các cụm dựa trên sự tương tự bên trong cụm so với sự khác biệt giữa các cụm.
calinski_harabasz = calinski_harabasz_score(X_encoded, kmeans.labels_)
print("Calinski-Harabasz Index:", calinski_harabasz)

# Phân giải dữ liệu để có thể vẽ biểu đồ phân tán
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)

# Vẽ mô hình phân cụm
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering')
plt.legend()
plt.show()
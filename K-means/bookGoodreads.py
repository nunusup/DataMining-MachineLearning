import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned.csv")
X= data.drop(columns=['Author'])# features, chọn all cột trừ cột Author

# #Chuyển dữ liệu thành số
le = LabelEncoder()
X['Book'] = le.fit_transform(X['Book'])#Phương thức fit_transform sẽ tính toán các giá trị cần thiết từ cột dữ liệu 'Book', 'Avg_Rating', và 'Genres' trong DataFrame X để có thể thực hiện mã hóa.
X['Avg_Rating'] = le.fit_transform(X['Avg_Rating'])#Sau khi đã tính toán xong, phương thức sẽ áp dụng việc mã hóa lên các giá trị trong cột dữ liệu tương ứng và trả về một mảng NumPy chứa các giá trị đã được mã hóa.
X['Genres'] = le.fit_transform(X['Genres'])

# #chia dữ liệu thành tập huấn luyện và kiểm tra
# #Đặt tỷ lệ dữ liệu kiểm tra là 20%, nghĩa là 80% dữ liệu sẽ được sử dụng để huấn luyện mô hình và 20% còn lại sẽ được sử dụng để kiểm tra mô hình.
# #Cung cấp một số ngẫu nhiên cho việc chia dữ liệu nhằm đảm bảo rằng việc chia dữ liệu sẽ tạo ra kết quả nhất quán mỗi khi chạy.
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# #Huấn luyện model với 5 cluster 
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)
# kmeans.fit(X)

# predicted_labels=kmeans.predict(X)
predicted_labels=kmeans.predict(X_train)

predict_X_test=kmeans.predict(X_test)
# Dự đoán nhãn trên tập kiểm tra

# Đánh giá hiệu suất của tập kiểm tra

# Silhouette score đo độ tương đồng của các điểm trong cùng một cụm và độ khác biệt giữa các cụm khác nhau.
silhouette_avg = silhouette_score(X_test, predict_X_test)
print("Silhouette Score:", silhouette_avg)
# features, true_labels = make_moons(
# n_samples=250, noise=0.05, random_state=42)
kmeans_silhouette = silhouette_score(
X_train, kmeans.labels_).round(2)
# # Davies-Bouldin Index đánh giá sự tách biệt giữa các cụm.
davies_bouldin = davies_bouldin_score(X_test, predict_X_test)
print("Davies-Bouldin Index:", davies_bouldin)

# # Calinski-Harabasz Index đo sự tách biệt giữa các cụm dựa trên sự tương tự bên trong cụm so với sự khác biệt giữa các cụm.
calinski_harabasz = calinski_harabasz_score(X_test, predict_X_test)
print("Calinski-Harabasz Index:", calinski_harabasz)

#Lưu model vào file Kmeans_python.jolib
# model_filename="Kmeans_Model.joblib"
# joblib.dump(kmeans, model_filename)

# Xuất dataframe thành tệp Excel
# cluster_df.to_excel("predicted_clusters2.xlsx", index=False)

centroids = kmeans.cluster_centers_
# # # Vẽ mô hình
plt.figure(figsize=(10, 6))
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.style.use("fivethirtyeight")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering')
plt.legend()
plt.show()

# #Hiển thị các tâm cluster
print("Centroids:")
print(centroids)

#=====================================================
#Dùng phương pháp Elobow để tìm số lượng cụm cần chia

# Khởi tạo danh sách để lưu độ biến động
inertia_values = []

# # # # Thử nghiệm số lượng clusters từ 1 đến 20
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=50)
    kmeans.fit(X_train)
    inertia_values.append(kmeans.inertia_)

# # Vẽ đồ thị elbow
plt.style.use("fivethirtyeight")
plt.plot(range(1, 21), inertia_values)
plt.xticks(range(1, 21))
plt.xlabel('Số lượng clusters')
plt.ylabel('Độ biến động')
plt.title('Phương pháp Elbow')
plt.show()


# Hiển thị sách trong tất cả các cụm ra màn hình console
# def display_books_in_clusters():
#     clusters = {}
#     for i in range(kmeans.n_clusters):
#         books_in_cluster = data[kmeans.labels_ == i]['Book'].tolist()
#         clusters[i] = books_in_cluster
#     return clusters
# #data full
# def display_books_in_clusters():
#     clusters = {}
#     for i in range(kmeans.n_clusters):
#         books_in_cluster = data[predicted_labels == i]['Book'].tolist()
#         clusters[i] = books_in_cluster
#     return clusters

# # Hiển thị các cụm sách
# def display_all_clusters():
#     clusters = display_books_in_clusters()
#     for cluster, books in clusters.items():
#         print(f"Cụm {cluster}:")
#         print(books)
# #data full
# def display_all_clusters():
#     clusters = display_books_in_clusters()
#     for cluster, books in clusters.items():
#         print(f"Cụm {cluster}:")
#         print(books)

# # Sử dụng hàm để hiển thị tất cả các cụm
# display_all_clusters()

# # Kiểm tra kích thước data
# print("Kích thước của data:", data.shape)
# print("Kích thước của kmeans.labels_:", kmeans.labels_.shape)

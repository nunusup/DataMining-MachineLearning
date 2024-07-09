import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import random

# Load mô hình KMeans từ file
kmeans_model = joblib.load("Kmeans_Model.joblib")

# Khởi tạo LabelEncoder cho mỗi feature
label_encoder_book = LabelEncoder()
label_encoder_genres = LabelEncoder()
label_encoder_avg_rating = LabelEncoder()

# Đọc dữ liệu đã huấn luyện để lấy label encoder
train_data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\Data\\google_book_predict_data.csv")
# label_encoder_book.fit(train_data['Book'])
# label_encoder_genres.fit(train_data['Genres'])
# label_encoder_avg_rating.fit(train_data['Avg_Rating'])
label_encoder_book.fit(train_data['title'])
label_encoder_genres.fit(train_data['categories']) #genres
label_encoder_avg_rating.fit(train_data['average_rating'])

# Hàm dự đoán kết quả cho dữ liệu mới nhập vào
def predict_new_data(book_name, genres_name, avg_rating):
    # Chuyển đổi dữ liệu mới thành mã hóa
    book_encoded = label_encoder_book.transform([book_name])[0]
    genres_encoded = label_encoder_genres.transform([genres_name])[0]
    avg_rating_encoded = label_encoder_avg_rating.transform([avg_rating])[0]
    
    # Dự đoán cụm cho dữ liệu mới
    new_data = [[book_encoded, genres_encoded, avg_rating_encoded]]
    predicted_cluster = kmeans_model.predict(new_data)[0]
    
    return predicted_cluster

# In ra các sách trong cụm đã dự đoán
# def print_books_in_cluster(cluster):
#     books_in_cluster = train_data[kmeans_model.labels_ == cluster]['Book'].tolist()
#     result_text = f""
#     for book in books_in_cluster[:7]:  # Chỉ hiển thị 7 sách đầu tiên
#         result_text += book + "\n"
#     return result_text
# def print_books_in_cluster(cluster):
#     books_in_cluster = train_data[kmeans_model.labels_ == cluster]['title'].tolist()
#     result_text = f""
#     for book in books_in_cluster[:7]:  # Chỉ hiển thị 7 sách đầu tiên
#         result_text += book + "\n"
#     return result_text

def print_books_in_cluster(cluster):
    books_in_cluster = train_data[kmeans_model.labels_ == cluster]['title'].tolist()
    random.shuffle(books_in_cluster)  # Trộn ngẫu nhiên danh sách sách trong cụm
    result_text = ""
    for book in books_in_cluster[:7]:  # Chỉ hiển thị 7 sách đầu tiên
        result_text += book + "\n"
    return result_text

# Hàm xử lý sự kiện khi nhấn nút "Dự đoán"
def on_predict_button_click():
    book_name = entry_book.get()
    genres_name = entry_genres.get()
    avg_rating = entry_avg_rating.get()

    # Dự đoán kết quả
    predicted_cluster = predict_new_data(book_name, genres_name, avg_rating)
    result_label.config(text=f"Đề xuất:")
    
    # Hiển thị sách trong cụm đã dự đoán
    result_text = print_books_in_cluster(predicted_cluster)
    books_in_cluster_label.config(text=result_text)

# Tạo giao diện tkinter
root = tk.Tk()
root.title("Dự đoán cụm sách")
root.geometry("400x300")

# Tạo các thành phần giao diện
label_book = ttk.Label(root, text="Tên sách:")
label_book.pack()

entry_book = ttk.Entry(root)
entry_book.pack()

label_genres = ttk.Label(root, text="Thể loại:")
label_genres.pack()

entry_genres = ttk.Entry(root)
entry_genres.pack()

label_avg_rating = ttk.Label(root, text="Đánh giá trung bình:")
label_avg_rating.pack()

entry_avg_rating = ttk.Entry(root)
entry_avg_rating.pack()

predict_button = ttk.Button(root, text="Dự đoán", command=on_predict_button_click)
predict_button.pack()

result_label = ttk.Label(root, text="")
result_label.pack()

books_in_cluster_label = ttk.Label(root, text="")
books_in_cluster_label.pack()

root.mainloop()

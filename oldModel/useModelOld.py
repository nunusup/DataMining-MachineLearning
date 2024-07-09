import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QFont
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import pandas as pd

class BookRecommendationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Book Recommendation App")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Nhãn và ô nhập tên sách
        self.book_name_label = QLabel("Nhập tên sách:", self)
        self.book_name_input = QLineEdit(self)
        layout.addWidget(self.book_name_label)
        layout.addWidget(self.book_name_input)

        # Button để hiển thị sách cùng cụm
        self.display_books_button = QPushButton("Dự Đoán", self)
        self.display_books_button.clicked.connect(self.display_similar_books)
        layout.addWidget(self.display_books_button)

        # Nhãn để hiển thị kết quả
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def display_similar_books(self):
        # Đọc mô hình đã lưu
        kmeans = joblib.load("kmeans_model_old.joblib")

        # Đọc dữ liệu
        data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned.csv")
        X = data['Genres']
        y = data['Book']

        # Mã hóa dữ liệu
        mlb = MultiLabelBinarizer()
        X_encoded = mlb.fit_transform(X)

        # Dự đoán cụm cho sách được nhập
        book_name = self.book_name_input.text()
        book_index = y[y == book_name].index[0]
        book_genre = X_encoded[book_index]
        predicted_cluster = kmeans.predict([book_genre])[0]

        # Hiển thị sách cùng cụm
        similar_books = data[kmeans.labels_ == predicted_cluster]['Book'].tolist()

        # Giới hạn số lượng sách hiển thị
        max_books = 7
        if len(similar_books) > max_books:
            similar_books = similar_books[:max_books]

        if similar_books:
            result_text = "Các sách cùng cụm:\n" + "\n".join(similar_books)
        else:
            result_text = "Không tìm thấy sách cùng cụm."
        self.result_label.setText(result_text)


def main():
    app = QApplication(sys.argv)
    window = BookRecommendationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

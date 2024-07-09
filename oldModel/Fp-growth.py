from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv("D:\\MachineLearning\\DMML_Kmeans\\Data\\data_cleaned.csv")

# Chuyển đổi dữ liệu thể loại thành danh sách
listData = data['Genres'].apply(lambda x: x.split(',')).tolist()

# Mã hóa dữ liệu để sử dụng cho FP-Growth
te = TransactionEncoder()
data_transformed = te.fit_transform(listData)
df = pd.DataFrame(data_transformed, columns=te.columns_)

# Khởi tạo K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List để lưu trữ kết quả của các luật kết hợp từ từng fold
all_rules = []

# List để lưu trữ các thông số đánh giá mô hình
evaluation_metrics = []

# Lặp qua các fold
for train_index, test_index in kf.split(df):
    # Tạo tập dữ liệu huấn luyện và tập dữ liệu kiểm tra từ fold hiện tại
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    # Tìm các tập phổ biến từ tập dữ liệu huấn luyện
    frequent_itemsets_FPGrowth = fpgrowth(train_data, min_support=0.03, use_colnames=True)

    # Tìm các luật kết hợp từ các tập phổ biến với ngưỡng Confidence
    rules = association_rules(frequent_itemsets_FPGrowth, metric="confidence", min_threshold=0.7)

    # Thêm các luật kết hợp từ fold hiện tại vào list
    all_rules.append(rules)

    # Đánh giá mô hình trên tập kiểm tra
    evaluation_metrics.append({
        'support': len(rules),  # Số lượng luật kết hợp
        'confidence_mean': rules['confidence'].mean(),  # Độ tin cậy trung bình của các luật
        # Tính Lift và Zhang's metric
        'lift_mean': (rules['confidence'] / frequent_itemsets_FPGrowth['support']).mean(),
        'zhangs_metric_mean': (rules['support'] * (rules['confidence'] - frequent_itemsets_FPGrowth['support']) /
                               (rules['support'] - frequent_itemsets_FPGrowth['support'])).mean(),
        # Các thông số khác như độ phủ, lift có thể được tính và thêm vào đây
    })

# Kết hợp các luật từ tất cả các fold
combined_rules = pd.concat(all_rules, ignore_index=True)

# Xuất các luật kết hợp ra tệp Excel
# combined_rules.to_excel('D:\\BTL khai phá\\khaiPha\\combined_association_rules.xlsx', index=False)

# # Tính toán các thông số đánh giá trung bình trên tất cả các fold
evaluation_metrics_df = pd.DataFrame(evaluation_metrics)
# evaluation_metrics_df.to_excel('D:\\BTL khai phá\\khaiPha\\evaluation_metrics.xlsx', index=False)

# In ra các thông số đánh giá trung bình
print("Evaluation Metrics:")
print(evaluation_metrics_df.mean())
print(combined_rules)
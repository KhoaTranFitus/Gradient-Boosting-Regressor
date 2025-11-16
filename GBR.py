import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# =========================
# 1. Load CSV từ Kaggle
# =========================

# Thay đường dẫn theo file của bạn
df = pd.read_csv("fashion-mnist_test.csv")

# Cột đầu là label (0-9)
y = df["label"].astype(float).values

# Các cột pixel là dữ liệu
X = df.drop("label", axis=1).values / 255.0   # chuẩn hóa

print("Loaded CSV:", X.shape, y.shape)

# Chia train/test 80/20
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2. Hàm đánh giá
# =========================
def evaluate_model(model, X_train, y_train, X_val, y_val, name="Model"):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_pred = model.predict(X_val)
    end_pred = time.time()

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"=== {name} ===")
    print(f"Train time: {end_train - start_train}")
    print(f"Predict time: {end_pred - start_pred}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print()
    return y_pred

# =========================
# 3. Mô hình
# =========================
dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42)

gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# =========================
# 4. Train & evaluate
# =========================
y_pred_dt = evaluate_model(
    dt_reg, X_train, y_train, X_val, y_val,
    name="DecisionTreeRegressor"
)

y_pred_gbr = evaluate_model(
    gbr, X_train, y_train, X_val, y_val,
    name="GradientBoostingRegressor"
)

# =========================
# 5. Tính accuracy bằng cách làm tròn
# =========================
y_val_int = y_val.astype(int)

acc_dt = np.mean(np.rint(y_pred_dt).astype(int) == y_val_int)
acc_gbr = np.mean(np.rint(y_pred_gbr).astype(int) == y_val_int)

print("Decision Tree Accuracy (rounded):", acc_dt)
print("Gradient Boosting Accuracy (rounded):", acc_gbr)

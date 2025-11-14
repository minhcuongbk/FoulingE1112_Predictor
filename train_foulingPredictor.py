import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ===========================
# 1. Đọc dữ liệu
# ===========================
df = pd.read_excel("C:\\Users\\cuongdm\\Downloads\\FoulingE-1112.xlsx")

# Đổi tên cột cho dễ dùng (sửa lại nếu tên khác)
df = df.rename(columns={
    "Nhiệt độ đầu vào": "Tin",
    "Nhiệt độ đầu ra": "Tout",
    "Độ mở van": "Valve",
    "Công suất CDU": "CDU",
    "Lưu lượng": "Flow",
    "Delta temp In Outlet E1112": "DeltaT",
    "Độ ẩm": "Humidity",
    "Nhiệt độ môi trường": "Tamb",
    # output
    "E-1112A": "E1112A",
    "E-1112B": "E1112B",
    "E-1112C": "E1112C",
    "E-1112D": "E1112D",
    "E-1112E": "E1112E",
    "E-1112F": "E1112F",
})

# ===========================
# 2. Feature engineering
# ===========================
# approach nhiệt độ so với môi trường
df["Approach"] = df["Tout"] - df["Tamb"]
# hiệu suất tương đối (chỉ để thêm thông tin)
df["Eff"] = df["DeltaT"] / df["CDU"].replace(0, 1)

# ===========================
# 3. Input / Output
# ===========================
X = df[[
    "Tin", "Tout", "Valve", "CDU",
    "Flow", "DeltaT", "Humidity",
    "Tamb", "Approach", "Eff"
]]

Y = df[["E1112A", "E1112B", "E1112C",
        "E1112D", "E1112E", "E1112F"]]

# ===========================
# 4. Train / Test split
# ===========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=42
)

# ===========================
# 5. Random Forest multi-output
# (RandomForestRegressor hỗ trợ multi-output trực tiếp)
# ===========================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, Y_train)

# ===========================
# 6. Đánh giá
# ===========================
Y_pred = rf.predict(X_test)

mae_vec = mean_absolute_error(Y_test, Y_pred, multioutput="raw_values")
r2_vec = [
    r2_score(Y_test[col], Y_pred[:, i])
    for i, col in enumerate(Y.columns)
]

print("Kết quả Random Forest:")
for i, col in enumerate(Y.columns):
    print(f"{col:7s} | MAE = {mae_vec[i]:8.3f} | R2 = {r2_vec[i]:6.3f}")

# ===========================
# 7. Lưu model
# ===========================
joblib.dump(rf, "rf_fouling_E1112.joblib")
print("Đã lưu model → rf_fouling_E1112.joblib")

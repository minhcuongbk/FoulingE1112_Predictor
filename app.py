from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import io

app = Flask(__name__)

# ==== 1. Load model đã train ====
model = joblib.load("C:\\Python\\Cuong PYTHON\\Fouling predictors\\rf_fouling_E1112.joblib")

# ==== 2. Hàm tạo feature giống lúc train ====
def make_features_from_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Đổi tên cột nếu dùng tên tiếng Việt trong file
    rename_map = {
        "Nhiệt độ đầu vào": "Tin",
        "Nhiệt độ đầu ra": "Tout",
        "Độ mở van": "Valve",
        "Công suất CDU": "CDU",
        "Lưu lượng": "Flow",
        "Delta temp In Outlet E1112": "DeltaT",
        "Độ ẩm": "Humidity",
        "Nhiệt độ môi trường": "Tamb",
    }
    df = df_raw.rename(columns=rename_map).copy()

    # Tính feature thêm (giống lúc training)
    df["Approach"] = df["Tout"] - df["Tamb"]
    df["Eff"] = df["DeltaT"] / df["CDU"].replace(0, 1)

    # Lấy đúng thứ tự cột input
    X = df[[
        "Tin", "Tout", "Valve", "CDU",
        "Flow", "DeltaT", "Humidity",
        "Tamb", "Approach", "Eff"
    ]]
    return X

# ==== 3. Trang chính ====
@app.route("/", methods=["GET", "POST"])
def index():
    manual_result = None
    table_html = None
    error_msg = None

    if request.method == "POST":
        action = request.form.get("action")

        # ---- Mode 1: nhập tay ----
        if action == "manual":
            try:
                Tin = float(request.form.get("Tin"))
                Tout = float(request.form.get("Tout"))
                Valve = float(request.form.get("Valve"))
                CDU = float(request.form.get("CDU"))
                Flow = float(request.form.get("Flow"))
                DeltaT = float(request.form.get("DeltaT"))
                Humidity = float(request.form.get("Humidity"))
                Tamb = float(request.form.get("Tamb"))

                # tạo DataFrame 1 dòng
                df_manual = pd.DataFrame([{
                    "Tin": Tin,
                    "Tout": Tout,
                    "Valve": Valve,
                    "CDU": CDU,
                    "Flow": Flow,
                    "DeltaT": DeltaT,
                    "Humidity": Humidity,
                    "Tamb": Tamb
                }])

                X = make_features_from_df(df_manual)
                y_pred = model.predict(X)[0]  # 6 giá trị

                manual_result = {
                    "E1112A": round(float(y_pred[0]), 3),
                    "E1112B": round(float(y_pred[1]), 3),
                    "E1112C": round(float(y_pred[2]), 3),
                    "E1112D": round(float(y_pred[3]), 3),
                    "E1112E": round(float(y_pred[4]), 3),
                    "E1112F": round(float(y_pred[5]), 3),
                }
            except Exception as e:
                error_msg = f"Lỗi khi xử lý input tay: {e}"

        # ---- Mode 2: upload file ----
        elif action == "file":
            file = request.files.get("file")
            if file and file.filename:
                try:
                    if file.filename.lower().endswith(".xlsx"):
                        df_file = pd.read_excel(file)
                    else:
                        df_file = pd.read_csv(file)

                    X = make_features_from_df(df_file)
                    preds = model.predict(X)

                    cols_pred = ["E1112A_pred", "E1112B_pred", "E1112C_pred",
                                 "E1112D_pred", "E1112E_pred", "E1112F_pred"]
                    df_pred = pd.DataFrame(preds, columns=cols_pred)

                    df_out = pd.concat([df_file, df_pred], axis=1)

                    # Lưu tạm vào session bằng HTML table
                    table_html = df_out.head(100).to_html(classes="table table-striped", index=False)

                    # Lưu ra biến global tạm (đơn giản)
                    # Nếu muốn cho tải file, ta ghi vào buffer và trả về khi user click nút tải
                    global last_file_result
                    last_file_result = df_out

                except Exception as e:
                    error_msg = f"Lỗi khi xử lý file: {e}"
            else:
                error_msg = "Chưa chọn file input!"

    return render_template("index.html",
                           manual_result=manual_result,
                           table_html=table_html,
                           error_msg=error_msg)

# ==== 4. Route tải file kết quả (tùy chọn) ====
@app.route("/download")
def download():
    global last_file_result
    if last_file_result is None:
        return "Chưa có kết quả để tải.", 400

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        last_file_result.to_excel(writer, index=False, sheet_name="Result")
    output.seek(0)

    return send_file(output,
                     download_name="Du_bao_fouling_RF.xlsx",
                     as_attachment=True)

# biến global để lưu kết quả file gần nhất
last_file_result = None

if __name__ == "__main__":
    app.run(debug=True)

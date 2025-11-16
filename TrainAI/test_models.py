import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------
# 🔹 ฟังก์ชันทำนายราคาทอง
# -----------------------------
def predict_gold_prices_from_csv(csv_path, model_path, scaler_path, n_test=5, window_size=120):
    """
    ทำนายราคาทองคำจากไฟล์ gold_data.csv ด้วยโมเดล LSTM และคืนผลลัพธ์เป็น DataFrame
    """
    # โหลดโมเดลและ scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # โหลดข้อมูลจาก CSV
    data = pd.read_csv(csv_path)

    # ตรวจสอบคอลัมน์ที่จำเป็น
    if 'Datetime' not in data.columns or 'Close' not in data.columns:
        raise ValueError("ไฟล์ gold_data.csv ต้องมีคอลัมน์ 'Datetime' และ 'Close'")

    # แปลง Datetime เป็นชนิด datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.sort_values('Datetime').reset_index(drop=True)

    close_prices = data[['Close']].values

    if len(close_prices) <= n_test:
        raise ValueError("จำนวนข้อมูลไม่พอสำหรับการทดสอบ")

    # แบ่งข้อมูล train/test
    train_prices = close_prices[:-n_test]
    test_prices = close_prices[-n_test:]
    test_times = data['Datetime'].iloc[-n_test:]

    # สเกลข้อมูล
    scaled_train = scaler.transform(train_prices)

    # ฟังก์ชันทำนายต่อเนื่อง
    def forecast_future(model, data_scaled, n_future, window_size):
        preds = []
        last_window = data_scaled[-window_size:].reshape(1, window_size, 1)
        for _ in range(n_future):
            pred = model.predict(last_window, verbose=0)
            preds.append(pred[0, 0])
            last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)
        return np.array(preds).reshape(-1, 1)

    # ทำนาย
    window_size = min(window_size, len(scaled_train))
    future_scaled = forecast_future(model, scaled_train, n_future=n_test, window_size=window_size)
    future_pred = scaler.inverse_transform(future_scaled)

    # รวมผลเป็น DataFrame
    forecast_df = pd.DataFrame({
        "Datetime": test_times.values,
        "Predicted_Price": future_pred.flatten(),
        "Actual_Price": test_prices.flatten()
    })

    # เพิ่ม margin ±1%
    forecast_df["Lower_Bound (-1%)"] = forecast_df["Predicted_Price"] * 0.99
    forecast_df["Upper_Bound (+1%)"] = forecast_df["Predicted_Price"] * 1.01

    return data, forecast_df


# -----------------------------
# 🔹 ฟังก์ชันแสดงกราฟ
# -----------------------------
def plot_gold_prediction(data, forecast_df, n_test=5):
    """
    แสดงกราฟราคาทองคำจริงและที่ทำนายไว้
    """
    plt.figure(figsize=(12,6))

    # ข้อมูลจริงย้อนหลัง 60 ชั่วโมง
    plt.plot(data['Datetime'].iloc[-(60+n_test):-n_test],
             data['Close'].iloc[-(60+n_test):-n_test],
             label='Actual (last 60 hrs)', color='blue')

    # ข้อมูลจริงในช่วง test
    plt.plot(forecast_df["Datetime"], forecast_df["Actual_Price"], 'o-', color='green', label='Actual (last hrs)')

    # ข้อมูลทำนาย
    plt.plot(forecast_df["Datetime"], forecast_df["Predicted_Price"], '--o', color='red', label='Predicted')

    # Margin
    plt.fill_between(forecast_df["Datetime"],
                     forecast_df["Lower_Bound (-1%)"],
                     forecast_df["Upper_Bound (+1%)"],
                     color='red', alpha=0.2, label='Margin ±1%')

    # plt.title(f"📈 การทำนายราคาทองคำล่วงหน้า {n_test} ชั่วโมง (จากไฟล์ gold_data.csv)")
    plt.xlabel("Datetime")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Data/gold_price_prediction.png")
    print("💾 Graph saved to: Data/gold_price_prediction.png")


def plot_gold_prediction2(data, forecast_df):
    import matplotlib.dates as mdates

    # ------------------------------
    # FIX 1: Ensure datetime is parsed correctly
    # ------------------------------
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="coerce")

    # ------------------------------
    # FIX 2: Remove timezone to prevent huge date values
    # ------------------------------
    if data["date"].dt.tz is not None:
        data["date"] = data["date"].dt.tz_localize(None)

    if forecast_df["date"].dt.tz is not None:
        forecast_df["date"] = forecast_df["date"].dt.tz_localize(None)

    # ------------------------------
    # FIX 3: Remove NaT values
    # ------------------------------
    data = data.dropna(subset=["date"])
    forecast_df = forecast_df.dropna(subset=["date"])

    # ------------------------------
    # FIX 4: Sort by date
    # ------------------------------
    data = data.sort_values("date")
    forecast_df = forecast_df.sort_values("date")

    # ------------------------------
    # Plot
    # ------------------------------
    plt.figure(figsize=(14,7))

    plt.plot(data["date"], data["Close"], label="Actual Close Price", color="blue")
    plt.plot(
        forecast_df["date"], 
        forecast_df["predicted_close"], 
        label="Predicted Close Price", 
        linestyle="--", 
        color="red"
    )

    plt.xlabel("Date")
    plt.ylabel("Gold Price (XAUUSD)")
    plt.title("Gold Price Prediction (Transformer)")
    plt.legend()
    plt.grid(True)

    # --- Nice Date Formatting ---
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


##############################################################################################################################################

# def predict_gold_prices_tranformer(csv_path, model_path, scaler_path, n_test=5, window_size=20):
#     """
#     รองรับ LSTM / Transformer — ใช้หลายฟีเจอร์ได้
#     """
#     model = load_model(model_path)
#     scaler = joblib.load(scaler_path)

#     data = pd.read_csv(csv_path)

#     if 'Datetime' not in data.columns or 'Close' not in data.columns:
#         raise ValueError("CSV ต้องมีคอลัมน์ 'Datetime' และ 'Close'")

#     data['Datetime'] = pd.to_datetime(data['Datetime'])
#     data = data.sort_values('Datetime').reset_index(drop=True)

#     # ✅ ใช้ทุกฟีเจอร์ยกเว้น Datetime
#     feature_cols = [c for c in data.columns if c not in ['Datetime']]
#     raw_values = data[feature_cols].values

#     if len(raw_values) <= n_test:
#         raise ValueError("ข้อมูลไม่พอ")

#     train_data = raw_values[:-n_test]
#     test_prices = data['Close'].values[-n_test:]
#     test_times = data['Datetime'].iloc[-n_test:]

#     # ✅ Scale ทุกฟีเจอร์
#     scaled_train = scaler.transform(train_data)

#     # ฟังก์ชันทำนายอนาคต
#     def forecast_future(model, data_scaled, n_future, window_size):
#         preds = []
#         last_window = data_scaled[-window_size:].reshape(1, window_size, data_scaled.shape[1])

#         for _ in range(n_future):
#             pred = model.predict(last_window, verbose=0)  # shape: (1,1)
#             preds.append(pred[0, 0])

#             # ✅ update window ใช้ predict เป็น Close เท่านั้น
#             new_row = np.copy(last_window[:, -1, :])
#             new_row[0, feature_cols.index('Close')] = pred[0, 0]  # update close only

#             last_window = np.append(last_window[:, 1:, :], new_row.reshape(1,1,-1), axis=1)

#         return np.array(preds).reshape(-1, 1)

#     window_size = min(window_size, len(scaled_train))
#     future_scaled = forecast_future(model, scaled_train, n_test, window_size)

#     # ✅ inverse scale เฉพาะ Close column
#     close_index = feature_cols.index("Close")
#     dummy = np.zeros((len(future_scaled), len(feature_cols)))
#     dummy[:, close_index] = future_scaled.flatten()
#     future_pred = scaler.inverse_transform(dummy)[:, close_index]

#     forecast_df = pd.DataFrame({
#         "Datetime": test_times,
#         "Predicted_Price": future_pred,
#         "Actual_Price": test_prices
#     })

#     forecast_df["Lower_Bound (-1%)"] = forecast_df["Predicted_Price"] * 0.99
#     forecast_df["Upper_Bound (+1%)"] = forecast_df["Predicted_Price"] * 1.01

#     return data, forecast_df


#################################################################################################################################

def predict_gold_sarima(csv_path, model_path, scaler_path, n_test=5, window_size=60):

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # --------------------------
    # Load data
    # --------------------------
    data = pd.read_csv(csv_path)
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    data = data.sort_values("Datetime")

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    raw_values = data[feature_cols].values

    train_data = raw_values[:-n_test]
    test_times = data["Datetime"].iloc[-n_test:]
    test_prices = data["Close"].iloc[-n_test:].values

    # scale
    scaled_train = scaler.transform(train_data)

    # --------------------------
    # Forecast function
    # --------------------------
    def forecast_future(model, scaled, n_future, window):
        preds = []
        last_window = scaled[-window:].reshape(1, window, len(feature_cols))

        for _ in range(n_future):
            pred = model.predict(last_window, verbose=0)[0][0]
            preds.append(pred)

            # update window
            new_row = last_window[:, -1, :]
            new_row = new_row.copy()
            new_row[0, feature_cols.index("Close")] = pred

            last_window = np.append(last_window[:, 1:, :],
                                    new_row.reshape(1, 1, -1),
                                    axis=1)

        return np.array(preds).reshape(-1, 1)

    window_size = min(window_size, len(scaled_train))
    future_scaled = forecast_future(model, scaled_train, n_test, window_size)

    # --------------------------
    # Inverse scale only Close
    # --------------------------
    dummy = np.zeros((len(future_scaled), len(feature_cols)))
    dummy[:, feature_cols.index("Close")] = future_scaled.flatten()

    future_pred = scaler.inverse_transform(dummy)[:, feature_cols.index("Close")]

    # --------------------------
    # Build df
    # --------------------------
    df = pd.DataFrame({
        "Datetime": test_times.values,
        "Predicted_Price": future_pred,
        "Actual_Price": test_prices
    })

    df["Lower_Bound (-1%)"] = df["Predicted_Price"] * 0.99
    df["Upper_Bound (+1%)"] = df["Predicted_Price"] * 1.01

    return data, df


#############################################################################################################################################

# =========================================
# 🔥 PREDICT FUTURE GOLD PRICE (Transformer)
# =========================================
def predict_gold_prices_transformer(
    csv_path,
    model_path="gold_transformer_model.keras",
    scaler_path="gold_transformer_scaler.pkl",
    n_test=10
):
    # Load data
    data = pd.read_csv(csv_path)
    data["date"] = pd.to_datetime(data["Datetime"])

    features = ['Open','High','Low','Close','Volume']
    dataset = data[features].values

    # Load model + scaler
    model = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    scaled = scaler.transform(dataset)

    window = 180
    last_window = scaled[-window:]

    preds = []

    # ================================
    # 🔄 Auto-regressive forecasting
    # ================================
    current_window = last_window.copy()

    for _ in range(n_test):
        X = np.expand_dims(current_window, axis=0)
        pred_scaled = model.predict(X)[0][0]  # raw scaled prediction

        # inverse only for Close
        empty = np.zeros((1, 5))
        empty[0, 4] = pred_scaled
        pred_real = scaler.inverse_transform(empty)[0, 4]

        preds.append(pred_real)

        new_row = current_window[-1].copy()
        new_row[3] = pred_scaled  
        current_window = np.vstack([current_window[1:], new_row])

    # build df
    future_dates = pd.date_range(
        start=data["date"].iloc[-1] + pd.Timedelta(hours=1),
        periods=n_test,
        freq="H"
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_close": preds
    })

    return data, forecast_df




#############################################################################################################################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# import joblib


# # -----------------------------
# # 🔹 ฟังก์ชันทำนายราคาทอง
# # -----------------------------
# def predict_gold_prices_from_csv(csv_path, model_path, scaler_path, n_test=5, window_size=120):
#     """
#     ทำนายราคาทองคำจากไฟล์ gold_data.csv ด้วยโมเดล LSTM และคืนผลลัพธ์เป็น DataFrame
#     """
#     # โหลดโมเดลและ scaler
#     model = load_model(model_path)
#     scaler = joblib.load(scaler_path)

#     # โหลดข้อมูลจาก CSV
#     data = pd.read_csv(csv_path)

#     # ตรวจสอบคอลัมน์
#     required_columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
#     for col in required_columns:
#         if col not in data.columns:
#             raise ValueError(f"❌ CSV ต้องมีคอลัมน์: {required_columns}")

#     # แปลง datetime
#     data['Datetime'] = pd.to_datetime(data['Datetime'])
#     data = data.sort_values('Datetime').reset_index(drop=True)

#     # ใช้ฟีเจอร์ทุกตัว
#     features = data[["Open", "High", "Low", "Close", "Volume"]].values
#     close_prices = data["Close"].values

#     if len(features) <= n_test:
#         raise ValueError("จำนวนข้อมูลไม่พอสำหรับการทดสอบ")

#     # แบ่ง train/test
#     train_features = features[:-n_test]
#     test_close = close_prices[-n_test:]
#     test_times = data['Datetime'].iloc[-n_test:]

#     # สเกล
#     scaled_train = scaler.transform(train_features)

#     # ฟังก์ชันทำนาย
#     def forecast_future(model, data_scaled, n_future, window_size):
#         preds = []
#         last_window = data_scaled[-window_size:].reshape(1, window_size, data_scaled.shape[1])
#         for _ in range(n_future):
#             pred = model.predict(last_window, verbose=0)
#             preds.append(pred[0, 0])

#             # เพิ่ม prediction กลับเข้า sequence (เฉพาะ close column index=3)
#             new_step = last_window[:, -1, :].copy()
#             new_step[0, 3] = pred[0, 0]

#             last_window = np.append(last_window[:, 1:, :], new_step.reshape(1,1,-1), axis=1)
#         return np.array(preds).reshape(-1, 1)

#     # ทำนาย
#     window_size = min(window_size, len(scaled_train))
#     future_scaled = forecast_future(model, scaled_train, n_test, window_size)

#     # ใส่กลับ array 5 features เพื่อ inverse scale
#     tmp = np.zeros((len(future_scaled), 5))
#     tmp[:, 3] = future_scaled.flatten()  # แทน close position

#     future_pred = scaler.inverse_transform(tmp)[:, 3]

#     # รวมผลลัพธ์
#     forecast_df = pd.DataFrame({
#         "Datetime": test_times.values,
#         "Predicted_Price": future_pred,
#         "Actual_Price": test_close
#     })

#     forecast_df["Lower_Bound (-1%)"] = forecast_df["Predicted_Price"] * 0.99
#     forecast_df["Upper_Bound (+1%)"] = forecast_df["Predicted_Price"] * 1.01

#     return data, forecast_df


# # -----------------------------
# # 🔹 ฟังก์ชันแสดงกราฟ
# # -----------------------------
# def plot_gold_prediction(data, forecast_df, n_test=5):
#     plt.figure(figsize=(12,6))

#     plt.plot(data['Datetime'].iloc[-(60+n_test):-n_test],
#              data['Close'].iloc[-(60+n_test):-n_test],
#              label='Actual (last 60 hrs)', color='blue')

#     plt.plot(forecast_df["Datetime"], forecast_df["Actual_Price"], 'o-', color='green', label='Actual (last hrs)')
#     plt.plot(forecast_df["Datetime"], forecast_df["Predicted_Price"], '--o', color='red', label='Predicted')

#     plt.fill_between(forecast_df["Datetime"],
#                      forecast_df["Lower_Bound (-1%)"],
#                      forecast_df["Upper_Bound (+1%)"],
#                      color='red', alpha=0.2, label='Margin ±1%')

#     plt.xlabel("Datetime")
#     plt.ylabel("Price (USD)")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("Data/gold_price_prediction.png")
#     print("💾 Graph saved to: Data/gold_price_prediction.png")

# =============================
# GOLD PRICE LSTM TRAINER (SAVE MODEL + SCALER)
# =============================

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib  # ✅ ใช้สำหรับบันทึก/โหลด scaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_gold_lstm(
    csv_path,
    save_model_path="gold_lstm_model.keras",
    save_scaler_path="gold_scaler.pkl",
    epochs=25,
    batch_size=32
):
    """
    เทรนโมเดล LSTM สำหรับพยากรณ์ราคาทองคำ
    Parameters:
        csv_path: str — path ของไฟล์ข้อมูล (เช่น gold_data.csv)
        save_model_path: str — path ที่ต้องการบันทึกโมเดล (.keras หรือ .h5)
        save_scaler_path: str — path ที่ต้องการบันทึก scaler (.pkl)
    Return:
        model, scaler, rmse
    """

    # 1️⃣ โหลดข้อมูล
    data = pd.read_csv(csv_path)
    data = data.rename(columns={
        'Open': 'open',
        'Close': 'close',
        'High': 'high',
        'Low': 'low',
        'Datetime': 'date',
        'Volume': 'volume'
    })
    data['date'] = pd.to_datetime(data['date'])
    close_data = data[['close']].values

    # 2️⃣ เตรียมข้อมูล
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(close_data)

    training_ratio = 0.95
    training_data_len = int(len(scaled_data) * training_ratio)
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 30:]

    def create_sequences(data, window_size=30):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 3️⃣ สร้างโมเดล LSTM
    model = keras.models.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])

    # 4️⃣ เทรนโมเดล
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1)

    # 5️⃣ ทำนายและประเมินผล
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    real_prices = close_data[training_data_len:]
    rmse = np.sqrt(mean_squared_error(real_prices, predictions))
    print(f"✅ RMSE: {rmse:.2f}")

    # 6️⃣ บันทึกโมเดลและ Scaler
    model.save(save_model_path)
    joblib.dump(scaler, save_scaler_path)

    print(f"💾 Model saved to: {save_model_path}")
    print(f"💾 Scaler saved to: {save_scaler_path}")

    return model, scaler, rmse

def continue_train_gold_lstm(
    csv_path,
    model_path="gold_lstm_model.keras",
    scaler_path="gold_scaler.pkl",
    epochs=50,
    batch_size=32
):
    """
    เทรนต่อจากโมเดลและ scaler เดิม
    """

    # -------------------------
    # 1️⃣ โหลดโมเดลและ scaler เดิม
    # -------------------------
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # -------------------------
    # 2️⃣ โหลดและเตรียมข้อมูลใหม่
    # -------------------------
    data = pd.read_csv(csv_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    close_data = data[['Close']].values

    scaled_data = scaler.transform(close_data)

    def create_sequences(data, window_size=30):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # -------------------------
    # 3️⃣ เทรนต่อจากโมเดลเดิม
    # -------------------------
    print("🧩 Continue training existing LSTM model...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    # -------------------------
    # 4️⃣ บันทึกโมเดลและ scaler กลับ
    # -------------------------
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print("💾 Updated model & scaler saved.")

    return model, scaler

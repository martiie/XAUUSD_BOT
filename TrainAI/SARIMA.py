# ==========================================
#  GOLD PRICE PREDICTOR — LSTM + 1D CNN
#  (Refactored in Medium.com Style)
# ==========================================

import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# -------------------------------------------------
# 🔧 สร้าง Sequence
# -------------------------------------------------
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 3])  # predict Close
    return np.array(X), np.array(y)


# -------------------------------------------------
# 🔥 เทรนโมเดล LSTM + CNN Hybrid
# -------------------------------------------------
def train_gold_lstm_cnn(
    csv_path,
    save_model_path="gold_lstm_cnn_model.keras",
    save_scaler_path="gold_scaler.pkl",
    window=60,
    epochs=35,
    batch_size=32
):
    # ---------------------------------------------------------------------------------
    # 1) Load & Prepare Data
    # ---------------------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    features = ["Open", "High", "Low", "Close", "Volume"]
    data = df[features].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    joblib.dump(scaler, save_scaler_path)

    train_size = int(len(scaled) * 0.9)
    train = scaled[:train_size]
    test = scaled[train_size-window:]

    X_train, y_train = create_sequences(train, window)
    X_test, y_test = create_sequences(test, window)

    # reshape for CNN/LSTM → (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], window, len(features)))
    X_test = X_test.reshape((X_test.shape[0], window, len(features)))

    # ---------------------------------------------------------------------------------
    # 2) Hybrid Model — LSTM + 1D-CNN (แบบ Medium)
    # ---------------------------------------------------------------------------------
    inputs = keras.Input(shape=(window, len(features)))

    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.LSTM(64, return_sequences=True)(x)
    x = keras.layers.LSTM(32, return_sequences=False)(x)

    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    print("🚀 Training LSTM+CNN Hybrid Model...")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # ---------------------------------------------------------------------------------
    # 3) Predict + Evaluate
    # ---------------------------------------------------------------------------------
    pred = model.predict(X_test)
    pred_full = np.zeros((pred.shape[0], 5))
    pred_full[:, 3] = pred.flatten()
    pred_actual = scaler.inverse_transform(pred_full)[:, 3]

    real = df["Close"].values[train_size:]
    rmse = np.sqrt(mean_squared_error(real[-len(pred_actual):], pred_actual))

    print(f"✅ Model RMSE: {rmse:.2f}")

    # ---------------------------------------------------------------------------------
    # 4) Save Model
    # ---------------------------------------------------------------------------------
    model.save(save_model_path)

    print("💾 Model saved:", save_model_path)
    print("💾 Scaler saved:", save_scaler_path)

    return model, scaler, rmse

def continue_train_gold_lstm_cnn(
    csv_path,
    model_path="gold_lstm_cnn_model.keras",
    scaler_path="gold_scaler.pkl",
    window=60,
    epochs=20,
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
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    features = ["Open", "High", "Low", "Close", "Volume"]
    data = df[features].values

    scaled = scaler.transform(data)

    train_size = int(len(scaled) * 0.9)
    train = scaled[:train_size]
    test = scaled[train_size-window:]

    X_train, y_train = create_sequences(train, window)
    X_test, y_test = create_sequences(test, window)

    X_train = X_train.reshape((X_train.shape[0], window, len(features)))
    X_test = X_test.reshape((X_test.shape[0], window, len(features)))

    # -------------------------
    # 3️⃣ เทรนโมเดลต่อ
    # -------------------------
    print("🚀 Continuing Training LSTM+CNN Hybrid Model...")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # -------------------------
    # 4️⃣ ทำนายและประเมินผล
    # -------------------------
    pred = model.predict(X_test)
    pred_full = np.zeros((pred.shape[0], 5))
    pred_full[:, 3] = pred.flatten()
    pred_actual = scaler.inverse_transform(pred_full)[:, 3]

    real = df["Close"].values[train_size:]
    rmse = np.sqrt(mean_squared_error(real[-len(pred_actual):], pred_actual))

    print(f"✅ Continued Model RMSE: {rmse:.2f}")

    # -------------------------
    # 5️⃣ บันทึกโมเดลและ Scaler
    # -------------------------
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print("💾 Updated Model Saved")
    return model, scaler, rmse


def predict_next_prices_lstm_cnn(model, scaler, last_window_data, n_future=10):
    features = last_window_data.shape[1]
    window = last_window_data.shape[0]

    preds = []

    current = last_window_data.copy()
    for _ in range(n_future):
        x = current.reshape(1, window, features)
        pred = model.predict(x, verbose=0)[0,0]

        new_step = current[-1].copy()
        new_step[3] = pred  # update Close
        current = np.vstack([current[1:], new_step])

        preds.append(pred)

    # inverse scale
    full = np.zeros((len(preds), 5))
    full[:,3] = preds
    return scaler.inverse_transform(full)[:,3]

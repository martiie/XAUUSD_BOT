# =============================
# GOLD PRICE LSTM TRAINER (MULTI-FEATURE + TUNED MODEL)
# =============================

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_gold_lstm(
    csv_path,
    save_model_path="gold_lstm_model2.keras",
    save_scaler_path="gold_scaler2.pkl",
    epochs=50,
    batch_size=32
):
    data = pd.read_csv(csv_path)
    data['date'] = pd.to_datetime(data['Datetime'])

    # ✅ เลือก Features หลายตัว (เพิ่มความแม่นยำ)
    features = ['Open','High','Low','Close','Volume']
    dataset = data[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    window = 180  # ✅ ยาวขึ้นเพื่อจับเทรนด์ดีขึ้น
    train_size = int(len(scaled_data) * 0.95)
    
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - window:]

    def create_sequences(data):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i][3])  # index=3 คือ Close
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    # ✅ LSTM Model (tuned)
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.Dropout(0.2),

        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    # ✅ Callback ป้องกัน Overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,  # ✅ อย่า shuffle time-series
        callbacks=callbacks,
        verbose=1
    )

    # ✅ Evaluate
    preds = model.predict(X_test)
    preds = scaler.inverse_transform(np.hstack((np.zeros((len(preds),4)), preds.reshape(-1,1))))[:,4]
    real = data['Close'].values[train_size:]

    rmse = np.sqrt(mean_squared_error(real, preds))
    print(f"✅ RMSE: {rmse:.2f}")

    # ✅ Save
    model.save(save_model_path)
    joblib.dump(scaler, save_scaler_path)

    print("💾 Model saved:", save_model_path)
    print("💾 Scaler saved:", save_scaler_path)

    return model, scaler, rmse


def continue_train_gold_lstm(
    csv_path,
    model_path="gold_lstm_model2.keras",
    scaler_path="gold_scaler2.pkl",
    epochs=30,
    batch_size=32
):
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    data = pd.read_csv(csv_path)
    features = ['Open','High','Low','Close','Volume']
    dataset = data[features].values
    scaled = scaler.transform(dataset)

    window = 180

    def create_sequences(data):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i][3])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled)

    print("🧩 Continue LSTM training...")
    model.fit(
        X, y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1
    )

    model.save(model_path)
    print("💾 Updated Model Saved")

    return model, scaler

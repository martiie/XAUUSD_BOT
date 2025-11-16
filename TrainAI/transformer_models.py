# =============================
# GOLD PRICE TRANSFORMER TRAINER (MULTI-FEATURE)
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


# =========================================
# 🔥 Positional Encoding (สำคัญสำหรับ Transformer)
# =========================================
def positional_encoding(position, d_model):
    angles = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    PE = np.zeros((position, d_model))
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return PE


# =========================================
# 🔥 Transformer Encoder Block
# =========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)

    x = keras.layers.Add()([x, inputs])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward
    ff = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = keras.layers.Dense(inputs.shape[-1])(ff)

    x = keras.layers.Add()([x, ff])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    return x


# =========================================
# ⭐ TRAIN FUNCTION
# =========================================
def train_gold_transformer(
    csv_path,
    save_model_path="gold_transformer_model.keras",
    save_scaler_path="gold_transformer_scaler.pkl",
    epochs=50,
    batch_size=32
):

    data = pd.read_csv(csv_path)
    data['date'] = pd.to_datetime(data['Datetime'])

    features = ['Open','High','Low','Close','Volume']
    dataset = data[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    window = 180
    train_size = int(len(scaled_data) * 0.95)

    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size-window:]

    def create_sequences(data):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i][3])    # Close
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    # =========================================
    # 🔥 Transformer Model
    # =========================================
    inputs = keras.layers.Input(shape=(window, X_train.shape[2]))

    # add positional encoding
    pe = positional_encoding(window, X_train.shape[2])
    x = inputs + pe

    # 3 blocks transformer
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0007),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    # Predict
    preds = model.predict(X_test)

    preds = scaler.inverse_transform(
        np.hstack((np.zeros((len(preds), 4)), preds))
    )[:, 4]

    real = data["Close"].values[train_size:]

    rmse = np.sqrt(mean_squared_error(real, preds))
    print(f"🔥 Transformer RMSE: {rmse:.5f}")

    # Save model + scaler
    model.save(save_model_path)
    joblib.dump(scaler, save_scaler_path)

    print("💾 Model saved:", save_model_path)
    print("💾 Scaler saved:", save_scaler_path)

    return model, scaler, rmse


# =========================================
# ⭐ CONTINUE TRAIN
# =========================================
def continue_train_gold_transformer(
    csv_path,
    model_path="gold_transformer_model.keras",
    scaler_path="gold_transformer_scaler.pkl",
    epochs=30,
    batch_size=32
):
    model = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

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

    print("🔄 Continue Transformer Training...")
    model.fit(
        X, y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1
    )

    model.save(model_path)
    print("💾 Updated Transformer Model Saved")

    return model, scaler

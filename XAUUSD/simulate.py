from XAUUSD import XAUUSD_Moduls
from tensorflow.keras.models import load_model
import joblib

def run():

    model = load_model("Models/gold_lstm_model.keras")
    scaler = joblib.load("Models/gold_scaler.pkl")

    XAUUSD_Moduls.auto_trading_with_lstm(
        model=model,
        scaler=scaler,
        symbol="GC=F",
        data_path="Data/gold_data_2y.csv",
        log_path="Data/trade_log.csv",
        interval_sec=1   # ✅ ดึงราคาทุกชั่วโมง
    )

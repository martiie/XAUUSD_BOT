import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import numpy as np

def get_historical_data(symbol="GC=F", drop_recent_months=0, save_path="Data/gold_data.csv",period="2y"):

    # ถ้ามีไฟล์อยู่แล้ว → โหลดจากไฟล์
    if os.path.exists(save_path):
        print("📂 โหลดข้อมูลจากไฟล์เก่า:", save_path)
        data = pd.read_csv(save_path, parse_dates=['Datetime'])
        # timezone ให้เป็น Bangkok (+07:00) เสมอ
        if data['Datetime'].dt.tz is None:
            data['Datetime'] = data['Datetime'].dt.tz_localize('Asia/Bangkok')
        else:
            data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Bangkok')
    else:
        # ถ้าไม่มีไฟล์ → ดึงข้อมูลใหม่
        print("⬇️ ดึงข้อมูลใหม่จาก yfinance...")
        data = yf.download(symbol, period=period, interval="1h")

        # แปลง MultiIndex columns เป็น single level
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] != 'Adj Close' else 'Close' for col in data.columns]

        data = data.reset_index()  # Datetime จาก index

        # timezone ให้เป็น Bangkok (+07:00)
        if data['Datetime'].dt.tz is None:
            data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
        else:
            data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Bangkok')


    # เลือกเฉพาะ columns ที่ต้องการ
    columns_needed = ['Datetime','Open','High','Low','Close','Volume']
    data = data[columns_needed]

    # ลบเดือนล่าสุด (ถ้ามีการระบุ)
    if drop_recent_months > 0:
        latest_date = data['Datetime'].max()
        cutoff_date = latest_date - pd.DateOffset(months=drop_recent_months)
        data = data[data['Datetime'] < cutoff_date]

    data.to_csv(save_path, index=False)
    print("✅ บันทึกข้อมูลลงไฟล์:", save_path)
    return data


def update_latest_data(symbol="GC=F", save_path="Data/gold_data.csv"):
    print("🔹 เริ่ม update_latest_data()")

    if not os.path.exists(save_path):
        print("⚠️ ไม่พบไฟล์ข้อมูลเก่า กำลังสร้างใหม่...")
        return get_historical_data(symbol=symbol)

    # โหลดข้อมูลเก่า
    data = pd.read_csv(save_path, parse_dates=['Datetime'])

    # timezone
    if data['Datetime'].dt.tz is None:
        data['Datetime'] = data['Datetime'].dt.tz_localize('Asia/Bangkok')
    else:
        data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Bangkok')

    last_time_local = data['Datetime'].max()
    last_time_utc = last_time_local.tz_convert('UTC')
    next_hour_utc = last_time_utc + timedelta(hours=1)
    next_hour_local = next_hour_utc.tz_convert('Asia/Bangkok')
    now_utc = datetime.now(timezone.utc)

    print(f"⏱ เวลาล่าสุดในไฟล์: {last_time_local}")
    print(f"⏱ เวลาถัดไปที่จะดึง: {next_hour_local}")
    print(f"⏱ เวลาปัจจุบัน UTC: {now_utc}")

    if next_hour_utc >= now_utc:
        print("⚠️ ยังไม่มีชั่วโมงใหม่")
        return data

    print(f"⬇️ กำลังดึงข้อมูลจาก {next_hour_utc+ timedelta(hours=7)} ...")
    new_data = yf.download(symbol, start=next_hour_utc, end=next_hour_utc + timedelta(hours=54), interval="1h")

    if new_data.empty:
        print(f"⚠️ ไม่มีข้อมูลจาก yfinance → ใช้ราคาปิดล่าสุดแทน")
        return data

    # แปลง MultiIndex columns เป็น single level
    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = [col[0] for col in new_data.columns]

    new_data = new_data.reset_index()  # Datetime จาก index

    # timezone: ปลอดภัยสำหรับ tz-naive
    if new_data['Datetime'].dt.tz is None:
        new_data['Datetime'] = new_data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    else:
        new_data['Datetime'] = new_data['Datetime'].dt.tz_convert('Asia/Bangkok')

    # เลือกเฉพาะ columns ที่ต้องการ
    columns_needed = ['Datetime','Open','High','Low','Close','Volume']
    new_data = new_data[columns_needed]

    # เอาแค่แถวแรก (ชั่วโมงถัดไป)
    new_row = new_data.iloc[[0]]
    data = pd.concat([data, new_row], ignore_index=True)

    data.to_csv(save_path, index=False)
    print(f"✅ เพิ่มข้อมูลชั่วโมงใหม่เรียบร้อย: {next_hour_local}, ขนาด DataFrame: {len(data)} แถว")

    return data


def create_trade_log(log_path="Data/trade_log.csv"):
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
    else:
        cols = ['Datetime', 'Action', 'Buy_Price', 'Sell_Price', 'Profit/Loss']
        log = pd.DataFrame(columns=cols)
        log.to_csv(log_path, index=False)
    return log

def save_trade_log(trade_log, log_path="Data/trade_log.csv"):
    trade_log.to_csv(log_path, index=False)
    print("💾 บันทึกประวัติการซื้อขาย:", log_path)


def trading_decision_with_lstm(current_row, predicted_price, last_trade, stop_loss=20, take_profit=30, sensitivity=0.001):
    """
    ตัดสินใจเทรดตามการทำนายของ LSTM
    Parameters:
        current_row: แถวข้อมูลล่าสุด {'Open','High','Low'}
        predicted_price: ราคาที่โมเดลทำนาย
        last_trade: dict เก็บข้อมูลการเทรดล่าสุด (Action, Buy_Price, Sell_Price, Profit/Loss)
        stop_loss: จุดตัดขาดทุน (หน่วยเป็นราคาต่าง)
        take_profit: จุดทำกำไร
        sensitivity: ค่าความไว (เช่น 0.001 = 0.1%)
    """

    open_price = current_row['Open']
    low_price = current_row['Low']
    high_price = current_row['High']

    # ---------------- ไม่มีสถานะ → เปิดสถานะใหม่ ----------------
    if last_trade is None or last_trade['Action'] in ['SELL', 'CLOSE']:

        # ถ้าทำนายว่าจะขึ้น → เปิด BUY
        if predicted_price > open_price * (1 + sensitivity):
            return {'Action': 'BUY', 'Buy_Price': open_price, 'Sell_Price': None, 'Profit/Loss': None}

        # ถ้าทำนายว่าจะลง → เปิด SELL
        elif predicted_price < open_price * (1 - sensitivity):
            return {'Action': 'SELL', 'Buy_Price': None, 'Sell_Price': open_price, 'Profit/Loss': None}

    # ---------------- กรณีมีสถานะ BUY ----------------
    elif last_trade['Action'] == 'BUY':
        buy_price = last_trade['Buy_Price']

        # Stop loss → ปิด SELL
        if low_price < buy_price - stop_loss:
            return {'Action': 'CLOSE', 'Buy_Price': buy_price, 'Sell_Price': low_price, 'Profit/Loss': low_price - buy_price}

        # Take profit → ปิด SELL
        elif high_price > buy_price + take_profit:
            return {'Action': 'CLOSE', 'Buy_Price': buy_price, 'Sell_Price': high_price, 'Profit/Loss': high_price - buy_price}

        # ถ้าทำนายว่าราคาจะลง → ปิด BUY
        elif predicted_price < buy_price:
            sell_price = min(low_price, predicted_price)
            return {'Action': 'CLOSE', 'Buy_Price': buy_price, 'Sell_Price': sell_price, 'Profit/Loss': sell_price - buy_price}

    # ---------------- กรณีมีสถานะ SELL ----------------
    elif last_trade['Action'] == 'SELL':
        sell_price = last_trade['Sell_Price']

        # Stop loss (ราคาขึ้นเกินไป)
        if high_price > sell_price + stop_loss:
            return {'Action': 'CLOSE', 'Buy_Price': high_price, 'Sell_Price': sell_price, 'Profit/Loss': sell_price - high_price}

        # Take profit (ราคาลงตามคาด)
        elif low_price < sell_price - take_profit:
            return {'Action': 'CLOSE', 'Buy_Price': low_price, 'Sell_Price': sell_price, 'Profit/Loss': sell_price - low_price}

        # ถ้าทำนายว่าราคาจะขึ้น → ปิด SELL
        elif predicted_price > sell_price:
            close_price = max(high_price, predicted_price)
            return {'Action': 'CLOSE', 'Buy_Price': close_price, 'Sell_Price': sell_price, 'Profit/Loss': sell_price - close_price}

    return None


# ============================================================
# 2️⃣ เตรียมข้อมูลสำหรับทำนาย
# ============================================================
def prepare_data_for_prediction(data_path="gold_data.csv", window_size=60):
    data = pd.read_csv(data_path, parse_dates=['Datetime'])
    if len(data) >= window_size:
        recent_data = data.iloc[-window_size:]
    else:
        recent_data = data
    return recent_data

# ============================================================
# 3️⃣ ทำนายราคาทองคำด้วยโมเดล LSTM
# ============================================================
def predict_gold_prices(model, scaler, data, n_future=1, window_size=60):
    close_prices = data[['Close']].values
    scaled_data = scaler.transform(close_prices)

    preds = []
    last_window = scaled_data[-window_size:].reshape(1, window_size, 1)

    for _ in range(n_future):
        pred = model.predict(last_window, verbose=0)
        preds.append(pred[0, 0])
        last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    future_scaled = np.array(preds).reshape(-1, 1)
    future_pred = scaler.inverse_transform(future_scaled)
    return future_pred[-1][0]


# def run_trading_latest_with_lstm(model, scaler, latest_row, data, log_path="Data/trade_log.csv"):
#     trade_log = create_trade_log(log_path)
#     last_trade = None if trade_log.empty else trade_log.iloc[-1].to_dict()

#     # ✅ ทำนายราคา
#     predicted_prices = predict_gold_prices(model, scaler, data)
#     #print(f"🔮 Predicted Prices (LSTM): {predicted_prices}")
#     predicted_price = float(predicted_prices)#[-1]
#     print(f"🔮 Predicted Price (LSTM): {predicted_price:.2f}")

#     current_row = latest_row.iloc[0]

#     # ✅ ถ้ามีออเดอร์ค้าง → ปิดก่อนทันที
#     if last_trade is not None and last_trade['Action'] in ['BUY', 'SELL']:
#         if last_trade['Action'] == 'BUY':
#             close_price = current_row['Close']
#             profit = close_price - last_trade['Buy_Price']
#         else:
#             close_price = current_row['Close']
#             profit = last_trade['Sell_Price'] - close_price

#         close_record = {
#             'Datetime': current_row['Datetime'],
#             'Action': 'CLOSE',
#             'Buy_Price': last_trade.get('Buy_Price'),
#             'Sell_Price': last_trade.get('Sell_Price'),
#             'Profit/Loss': profit
#         }

#         trade_log = pd.concat([trade_log, pd.DataFrame([close_record])], ignore_index=True)
#         save_trade_log(trade_log, log_path)

#         print(f"✅ ปิดออเดอร์ก่อนเปิดใหม่: กำไร/ขาดทุน = {profit:.2f}")

#     # ✅ เปิดสถานะใหม่ตามโมเดล
#     action = None
#     open_price = current_row['Open']

#     if predicted_price > open_price:  # ขึ้น
#         action = {'Action': 'BUY', 'Buy_Price': open_price, 'Sell_Price': None, 'Profit/Loss': None}
#     elif predicted_price < open_price:  # ลง
#         action = {'Action': 'SELL', 'Buy_Price': None, 'Sell_Price': open_price, 'Profit/Loss': None}

#     if action:
#         open_record = {'Datetime': current_row['Datetime'], **action}
#         trade_log = pd.concat([trade_log, pd.DataFrame([open_record])], ignore_index=True)
#         save_trade_log(trade_log, log_path)

#         print(f"🚀 เปิดออเดอร์ใหม่: {action['Action']} @ {open_price}")
#         return open_record

#     print("⏳ ไม่มีสัญญาณเปิดเทรดใหม่ (แต่ระบบปิดออเดอร์ไปแล้วนะ)")
#     return None

def run_trading_latest_with_lstm(
        model, scaler, latest_row, data, 
        log_path="Data/trade_log.csv", 
        stop_loss=20, take_profit=30
    ):
    
    trade_log = create_trade_log(log_path)
    last_trade = None if trade_log.empty else trade_log.iloc[-1].to_dict()

    # ✅ Predict price
    predicted_price = float(predict_gold_prices(model, scaler, data))
    current = latest_row.iloc[0]

    open_price = current['Open']
    high = current['High']
    low = current['Low']
    close = current['Close']

    print(f"🔮 Predicted: {predicted_price:.2f} | Open:{open_price}")

    # ==========================
    # ✅ ถ้ามี Order ค้างอยู่ → เช็ค SL/TP ก่อน
    # ==========================
    if last_trade is not None and last_trade['Action'] in ['BUY', 'SELL']:
        if last_trade['Action'] == 'BUY':
            entry = last_trade['Buy_Price']

            # Take Profit
            if high >= entry + take_profit:
                profit = (entry + take_profit) - entry
                exit_price = entry + take_profit

            # Stop Loss
            elif low <= entry - stop_loss:
                profit = (entry - stop_loss) - entry
                exit_price = entry - stop_loss

            else:
                # ปิดตอนแท่งนี้จบ
                profit = close - entry
                exit_price = close

            # ✅ บันทึกการปิด
            close_record = {
                'Datetime': current['Datetime'],
                'Action': 'CLOSE',
                'Buy_Price': entry,
                'Sell_Price': exit_price,
                'Profit/Loss': profit
            }

        elif last_trade['Action'] == 'SELL':
            entry = last_trade['Sell_Price']

            # Take Profit
            if low <= entry - take_profit:
                profit = entry - (entry - take_profit)
                exit_price = entry - take_profit

            # Stop Loss
            elif high >= entry + stop_loss:
                profit = entry - (entry + stop_loss)
                exit_price = entry + stop_loss

            else:
                # ปิดแท่งนี้
                profit = entry - close
                exit_price = close

            # ✅ บันทึกการปิด
            close_record = {
                'Datetime': current['Datetime'],
                'Action': 'CLOSE',
                'Buy_Price': exit_price,
                'Sell_Price': entry,
                'Profit/Loss': profit
            }

        trade_log = pd.concat([trade_log, pd.DataFrame([close_record])], ignore_index=True)
        save_trade_log(trade_log, log_path)
        print(f"✅ ปิดออเดอร์: กำไร/ขาดทุน = {profit:.2f}")

    # ==========================
    # ✅ เปิดออเดอร์ใหม่ตามสัญญาณ LSTM
    # ==========================
    action = None
    if predicted_price > open_price:
        action = { 'Action':'BUY', 'Buy_Price':open_price, 'Sell_Price':None, 'Profit/Loss':None }
    elif predicted_price < open_price:
        action = { 'Action':'SELL', 'Buy_Price':None, 'Sell_Price':open_price, 'Profit/Loss':None }

    if action:
        new_record = { 'Datetime': current['Datetime'], **action }
        trade_log = pd.concat([trade_log, pd.DataFrame([new_record])], ignore_index=True)
        save_trade_log(trade_log, log_path)
        print(f"🚀 เปิดออเดอร์ใหม่: {action['Action']} @ {open_price}")
        return new_record

    print("⏳ ไม่มีสัญญาณเปิดใหม่")
    return None



# ============================================================
# 6️⃣ รวมทั้งหมดใน loop อัตโนมัติ
# ============================================================
import time

def auto_trading_with_lstm(model, scaler, symbol="GC=F", data_path="gold_data.csv", log_path="Data/trade_log.csv", interval_sec=1):
    print("🚀 เริ่มระบบ Auto Trading (LSTM)...")
    while True:
        try:
            data = update_latest_data(symbol, save_path=data_path)
            latest_row = data.iloc[[-1]]

            record = run_trading_latest_with_lstm(model, scaler, latest_row, data, log_path)
            if record:
                print(f"📈 {record['Datetime']} → {record['Action']} @ {record.get('Buy_Price', '')} / {record.get('Sell_Price', '')}")
            else:
                print(f"⏳ {latest_row.iloc[0]['Datetime']} → ไม่มี action")

            print(f"⏱ รอ {interval_sec} วินาที...")
            time.sleep(interval_sec)

        except Exception as e:
            print("⚠️ Error:", e)
            print("⏱ รอ 60 วินาทีแล้วลองใหม่...")
            time.sleep(60)

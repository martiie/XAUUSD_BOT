import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from XAUUSD import simulate
from TrainAI import lstm_models
# from TrainAI.test_models import predict_gold_prices_from_csv,plot_gold_prediction
import send_discord
from XAUUSD.ChatBotNews import GoldNewsAnalyzer
from datetime import datetime as time
from XAUUSD import XAUUSD_Moduls

def main():
    ######################################################################################################

    simulate.run()

    ######################################################################################################
    # lstm_models.train_gold_lstm(csv_path="Data/gold_data_2y.csv", 
    #                             epochs=50, 
    #                             batch_size=32,
    #                             save_model_path="Models/gold_lstm_model.keras", 
    #                             save_scaler_path="Models/gold_scaler.pkl")
    # lstm_models.continue_train_gold_lstm(
    #     csv_path="Data/gold_data_2y.csv",
    #     model_path="Models/gold_lstm_model.keras",
    #     scaler_path="Models/gold_scaler.pkl",
    #     epochs=20,
    #     batch_size=32)
    ######################################################################################################
    csv_path = "Data/gold_data_2y.csv"
    model_path = "Models/gold_lstm_model.keras"
    scaler_path = "Models/gold_scaler.pkl"

    data, forecast_df = lstm_models.predict_gold_prices_from_csv(csv_path, model_path, scaler_path)

    lstm_models.plot_gold_prediction(data, forecast_df)

    GOOGLE_API_KEY = ""

    analyzer = GoldNewsAnalyzer(GOOGLE_API_KEY)

    # ดึงข่าวล่าสุด 3 ข่าว
    df_news = analyzer.fetch_gold_news_mining(max_articles=1)
    full_text = analyzer.fetch_full_article(df_news['url'][0])
    latest_price = str(data.iloc[-1])  # ราคาทองล่าสุด
    ml_prediction = str(forecast_df.iloc[0])  # ตัวอย่างผลโมเดล

    # # วิเคราะห์ข่าว
    ai_analysis = analyzer.analyze_news(latest_price, ml_prediction, full_text)

    # # บันทึกลง CSV
    analyzer.save_to_csv(full_text, latest_price, ml_prediction, ai_analysis)

    csv_path = "gold_analysis.csv"
    WEBHOOK_URL = ""
    # ข้อความ
    message = f"📈 ราคาทองล่าสุด: {ml_prediction}\nAI วิเคราะห์: {ai_analysis}"

    # ส่งไป Discord
    send_discord.send_discord_webhook(
        webhook_url=WEBHOOK_URL,
        message=message,
        image_path="Data/gold_price_prediction.png",  # กราฟ
    )




if __name__ == "__main__":
    main()

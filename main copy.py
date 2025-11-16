import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from XAUUSD import simulate
from TrainAI import lstm_models, transformer_models, SARIMA
from TrainAI.test_models import predict_gold_prices_from_csv,plot_gold_prediction,plot_gold_prediction2, predict_gold_prices_transformer,predict_gold_sarima
from XAUUSD.ChatBotNews import GoldNewsAnalyzer

from XAUUSD import XAUUSD_Moduls

def main():

    #
    # XAUUSD_Moduls.get_historical_data(symbol="GC=F", drop_recent_months=0,period="2y", save_path="Data/gold_data_2y.csv")

    ######################################################################################################

    #simulate.run()

    ######################################################################################################
    # SARIMA.train_gold_lstm_cnn(
    #     csv_path="Data/gold_data.csv",
    #     epochs=35,
    #     batch_size=32,
    #     save_model_path="Models/SARIMA_model.keras",
    #     save_scaler_path="Models/SARIMA_scaler.pkl"
    # )
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

    # transformer_models.train_gold_transformer(
    #     csv_path="Data/gold_data.csv",
    #     epochs=50,
    #     batch_size=32,
    #     save_model_path="Models/gold_transformer_model.keras",
    #     save_scaler_path="Models/gold_transformer_scaler.pkl"
    # )
    # transformer_models.continue_train_gold_transformer(
    #     csv_path="Data/gold_data_2y.csv",
    #     model_path="Models/gold_transformer_model.keras",
    #     scaler_path="Models/gold_transformer_scaler.pkl",
    #     epochs=20,
    #     batch_size=32
    # )

    ##########################################################################################################
    csv_path = "Data/gold_data_2y.csv"
    model_path = "Models/gold_lstm_model.keras"
    scaler_path = "Models/gold_scaler.pkl"

    data, forecast_df = predict_gold_prices_from_csv(csv_path, model_path, scaler_path, n_test=5)

    plot_gold_prediction(data, forecast_df)

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
    
    ######################################################################################################
    # csv_path = "Data/gold_data.csv"
    # model_path = "Models/gold_transformer_model.keras"
    # scaler_path = "Models/gold_transformer_scaler.pkl"

    # data, forecast_df = predict_gold_prices_transformer(csv_path, model_path, scaler_path, n_test=10)
    # print(forecast_df["predicted_close"].max())
    # plot_gold_prediction2(data, forecast_df)
    ######################################################################################################
    # csv_path = "Data/gold_data_2y.csv"
    # model_path = "Models/SARIMA_model.keras"
    # scaler_path = "Models/SARIMA_scaler.pkl"
    # data, forecast_df = predict_gold_sarima(csv_path, model_path, scaler_path, n_test=10)
    # plot_gold_prediction(data, forecast_df)

if __name__ == "__main__":
    main()

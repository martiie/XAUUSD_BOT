from TrainAI import lstm_models
# lstm_models.train_gold_lstm(csv_path="Data/gold_data_2y.csv", 
#                             epochs=50, 
#                             batch_size=32,
#                             save_model_path="Models/gold_lstm_model.keras", 
#                             save_scaler_path="Models/gold_scaler.pkl")
lstm_models.continue_train_gold_lstm(
    csv_path="Data/gold_data_2y.csv",
    model_path="Models/gold_lstm_model.keras",
    scaler_path="Models/gold_scaler.pkl",
    epochs=20,
    batch_size=32)
import streamlit as st
from io import BytesIO
import os

# Display title and image
st.title("BrentWave Crude Vision - Future Prediction 📈📊")
st.image("download.jpeg", caption="Brent Crude Oil", use_column_width=True)

usage_documentation = """
## Welcome to Brent Crude Oil Price Prediction App!

This app allows you to predict the future prices of Brent Crude Oil using various machine learning models.

### How to Use:
1. **Select Model:** Choose one of the available models from the dropdown menu.
2. **Adjust Future Days:** Use the slider to select the number of future days for which you want predictions.
3. **View Predictions:** Once you select a model and adjust the future days, the app will display the predicted prices and future predictions.

Feel free to explore and analyze the data!

---

"""

# Provide a button to download usage documentation as a Word document
def download_documentation(usage_documentation, filename="Brent Crude Oil Price Prediction App.docx"):
    byte_data = BytesIO()
    os.system(f"pandoc -s -o {filename} -f markdown -t docx <(echo '{usage_documentation}')")
    with open(filename, "rb") as file:
        byte_data.write(file.read())
    return byte_data.getvalue()

if st.button("Download Usage Documentation"):
    docx_data = download_documentation(usage_documentation)
    st.download_button(label="Download", data=docx_data, file_name="usage_documentation.docx", mime="application/octet-stream")






import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from keras.models import load_model
import joblib
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


st.set_option('deprecation.showPyplotGlobalUse', False)


# Load models
model_lstm = load_model("E:\\stock-prediction-master\\model_lstm.h5")
model_gbr = joblib.load("E:\\stock-prediction-master\\model_gbr.pkl")
model_xgb = xgb.Booster(model_file="E:\\stock-prediction-master\\model_xgb.json")
model_arima = joblib.load("E:\\stock-prediction-master\\model_arima.pkl")
model_svr = joblib.load("E:\\stock-prediction-master\\model_svr.pkl")  # Load SVR model

def scrape_data():
    # Scrape Brent Crude Oil data from Yahoo Finance
    data = yf.download("BZ=F", start="2020-07-30", end='2024-05-05', interval="1d")
    return data

@st.cache_resource
def preprocess_data(df):
    # Perform any necessary preprocessing
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    
    return df_filled

def predict(model, X_test):
    # Make predictions
    predictions = model.predict(X_test)
    return predictions

def predict_future_svr(model, X_test, future_periods):
    # Make predictions for future periods using the SVR model
    future_predictions = []
    input_data = X_test[-1:]  # Last observed data point
    for _ in range(future_periods):
        prediction = model.predict(input_data)[0]
        future_predictions.append(prediction)
        input_data = np.roll(input_data, -1)
        input_data[-1] = prediction
    return future_predictions


def predict_future_lstm(model, X_test, future_periods):
    # Make future predictions using the LSTM model
    future_predictions = []
    input_data = X_test[-1:]  # Last observed data point
    for _ in range(future_periods):
        prediction = model.predict(input_data)[0][0]
        future_predictions.append(prediction)
        input_data = np.roll(input_data, -1)
        input_data[-1] = prediction
    return future_predictions

def predict_future_gbr_svr(model, X_test, future_periods):
    # Make future predictions using the GBR or SVR model
    future_predictions = model.predict(X_test[-1:].reshape(1, -1))
    for _ in range(future_periods - 1):
        future_predictions = np.append(future_predictions, model.predict(future_predictions[-1:].reshape(1, -1)))
    return future_predictions

def predict_future_xgb(model, X_test, future_periods):
    # Make future predictions using the XGBoost model
    future_predictions = []
    input_data = X_test[-1:]  # Last observed data point
    for _ in range(future_periods):
        dmatrix = xgb.DMatrix(input_data)
        prediction = model.predict(dmatrix)[0]
        future_predictions.append(prediction)
        input_data = np.roll(input_data, -1)
        input_data[-1] = prediction
    return future_predictions

def predict_future_arima(model, future_periods):
    # Make future predictions using the ARIMA model
    future_predictions = model.forecast(steps=future_periods)
    return future_predictions

def evaluate_model_svr(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def plot_predictions(predictions):
    # Plot the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted Prices")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.title("Brent Crude Oil Price Prediction")
    plt.legend()
    st.pyplot()

def plot_future_predictions(predictions, data):
    # Plot the future predictions
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(predictions):], predictions, label="Future Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Future Price Predictions")
    plt.legend()
    st.pyplot()


def main():
    st.title("Brent Crude Oil Price Prediction")

    # Load data
    data = scrape_data()

    # Preprocess data
    data = preprocess_data(data)

    # Display raw data
    st.subheader("Raw Data")
    st.write(data.tail())

    # Model selection
    selected_model = st.selectbox("Select Model", ["LSTM", "GBR", "XGBoost", "SVR (Recommended)", "ARIMA"])

    # Split data into features and target
    X = data[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA100']]
    y = data['Close']

    # Scaling features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Make predictions
    if selected_model == "LSTM":
        predictions = predict(model_lstm, X_scaled)
        plot_predictions(predictions)
        future_periods = st.slider("Select Number of Future Days to Predict", 1, 30, 7)
        future_predictions = predict_future_lstm(model_lstm, X_scaled, future_periods)
        
    elif selected_model == "GBR":
        predictions = model_gbr.predict(X_scaled)
        plot_predictions(predictions)
        future_periods = st.slider("Select Number of Future Days to Predict", 1, 30, 7)
        future_predictions = predict_future_gbr_svr(model_gbr, X_scaled, future_periods)
        
    elif selected_model == "XGBoost":
        dmatrix = xgb.DMatrix(X_scaled)
        predictions = model_xgb.predict(dmatrix)
        plot_predictions(predictions)
        future_periods = st.slider("Select Number of Future Days to Predict", 1, 30, 7)
        future_predictions = predict_future_xgb(model_xgb, X_scaled, future_periods)
        
    elif selected_model == "SVR (Recommended)":
        predictions = model_svr.predict(X_scaled)
        plot_predictions(predictions)
        future_periods = st.slider("Select Number of Future Days to Predict", 1, 30, 7)
        future_predictions = predict_future_svr(model_svr, X_scaled, future_periods)  # Use SVR predictions
        
    elif selected_model == "ARIMA":
        predictions = model_arima.predict(n_periods=len(X_scaled))
        plot_predictions(predictions)
        future_periods = st.slider("Select Number of Future Days to Predict", 1, 30, 7)
        future_predictions = predict_future_arima(model_arima, future_periods)

    if future_predictions is not None:
        plot_future_predictions(future_predictions, data)
    else:
        st.write("Future predictions are not available for the selected model.")


if __name__ == "__main__":
    main()



st.write("""
### Disclaimer:
The predictions provided by this app are for educational and informational purposes only. They should not be considered as financial advice or used for making investment decisions. The future performance of the Brent Crude Oil market cannot be guaranteed, and predictions may not always be accurate. Users are advised to use the information provided by this app wisely and to conduct their own research before making any financial decisions.
""")


st.write("""
---
Developed by Karmukilan, Kanagavel, and Chiranjeevi
""")

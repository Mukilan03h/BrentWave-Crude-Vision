# BrentWave Crude Vision - Future Prediction 📈📊

## Welcome to Brent Crude Oil Price Prediction App!

This app allows you to predict the future prices of Brent Crude Oil using various machine learning models.

### How to Use:
1. **Select Model:** Choose one of the available models from the dropdown menu.
2. **Adjust Future Days:** Use the slider to select the number of future days for which you want predictions.
3. **View Predictions:** Once you select a model and adjust the future days, the app will display the predicted prices and future predictions.

Feel free to explore and analyze the data!

---

## App Features

- **Model Selection:** Choose between LSTM, GBR, XGBoost, SVR, and ARIMA models.
- **Data Preprocessing:** Handles missing values and computes moving averages.
- **Predictions:** Provides current and future price predictions.
- **Visualization:** Displays raw data and prediction plots.
- **Documentation Download:** Option to download usage documentation as a Word document.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mukilan03h/BrentWave-Crude-Vision.git
```

2. Change to the project directory:

```bash
cd BrentWave-Crude-Vision
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501` to access the app.

## Models

- **LSTM:** Long Short-Term Memory model.
- **GBR:** Gradient Boosting Regressor.
- **XGBoost:** Extreme Gradient Boosting.
- **SVR:** Support Vector Regressor (Recommended).
- **ARIMA:** AutoRegressive Integrated Moving Average.

## Disclaimer
The predictions provided by this app are for educational and informational purposes only. They should not be considered as financial advice or used for making investment decisions. The future performance of the Brent Crude Oil market cannot be guaranteed, and predictions may not always be accurate. Users are advised to use the information provided by this app wisely and to conduct their own research before making any financial decisions.

---

Developed by Karmukilan, Kanagavel, and Chiranjeevi

## Code Overview

### Main Application

The main application file (`app.py`) includes the following:

- **Title and Image Display:** Displays the app title and an image of Brent Crude Oil.
- **Model Loading:** Loads the pre-trained models (LSTM, GBR, XGBoost, SVR, and ARIMA).
- **Data Scraping and Preprocessing:** Scrapes Brent Crude Oil data from Yahoo Finance and preprocesses it.
- **Prediction Functions:** Contains functions to make predictions using the selected model.
- **Evaluation and Plotting:** Evaluates model performance and plots the predictions.
- **Streamlit Interface:** Provides the user interface for model selection, future days adjustment, and viewing predictions.

### Functions

- **scrape_data():** Scrapes Brent Crude Oil data from Yahoo Finance.
- **preprocess_data(df):** Preprocesses the data by filling missing values and computing moving averages.
- **predict(model, X_test):** Makes predictions using the selected model.
- **predict_future_*:** Functions for making future predictions using different models.
- **evaluate_model_svr(y_true, y_pred):** Evaluates the SVR model performance.
- **plot_predictions(predictions):** Plots the current predictions.
- **plot_future_predictions(predictions, data):** Plots the future predictions.

### Documentation Download

Provides a button to download usage documentation as a Word document using the `download_documentation()` function.

## Future Enhancements

- **Additional Models:** Incorporate more advanced models for better prediction accuracy.
- **Enhanced Visualization:** Add more detailed and interactive visualizations.
- **User Authentication:** Implement user authentication for personalized experience.
- **API Integration:** Integrate with financial APIs for real-time data updates.


Thank you for using BrentWave Crude Vision! We hope you find this app useful for your analysis and predictions of Brent Crude Oil prices.

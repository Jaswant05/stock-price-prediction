# # import liabraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import load_model
# import streamlit as st
# import yfinance as yf

# start="2009-01-01"
# end="2023-01-01"

# st.title('Stock Closing Price Prediction')
# # reading data
# user_input = st.text_input('Enter Stock Ticker', 'GOOGL')
# df = yf.download(user_input, start=start, end=end)

# st.subheader('Dated from 1st Jan, 2009 to 1st Jan, 2023')
# st.write(df.describe())

# # first plot
# st.subheader('Closing Price Vs Time Chart')
# fig1 = plt.figure(figsize = (12, 6))
# plt.plot(df.Close)
# st.pyplot(fig1)

# # moving average
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()

# # second plot
# st.subheader('Closing Price Vs Time Chart with 100 days Moving Average')
# fig2 = plt.figure(figsize = (12, 6))
# plt.plot(df.Close, 'r', label="Per Day Closing")
# plt.plot(ma100, 'g', label="Moving Average 100")
# st.pyplot(fig2)

# # third plot
# st.subheader('Closing Price Vs Time Chart with 100 days and 200 days Moving Average')
# fig2 = plt.figure(figsize = (12, 6))
# plt.plot(ma200, 'b', label="Moving Average 200")
# plt.plot(ma100, 'g', label="Moving Average 100")
# st.pyplot(fig2)

# # data training
# train_df = pd.DataFrame(df['Close'][0: int(len(df)*0.85)])
# test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):int(len(df))])

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))

# train_df_arr = scaler.fit_transform(train_df)
 
# model = load_model('keras_model.h5')
# past_100_days = train_df.tail(100)
# final_df = past_100_days._append(test_df, ignore_index=True)

# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []
# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100: i])    
#     y_test.append(input_data[i, 0])
# x_test, y_test = np.array(x_test), np.array(y_test)

# # model prediction
# y_pred = model.predict(x_test)
# scale = scaler.scale_
# scale_factor = 1/scale[0]
# y_pred = y_pred*scale_factor
# y_test = y_test*scale_factor

# # final plot
# st.subheader('Predicted Vs Original')
# fig3 = plt.figure(figsize = (12, 6))
# plt.plot(y_test, 'g', label="Original Price")
# plt.plot(y_pred, 'r', label="Predicted Price")
# st.pyplot(fig3)
# -----------------------------------------------
# Import libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import load_model
# import streamlit as st
# import yfinance as yf

# from sklearn.preprocessing import MinMaxScaler

# # Set date range
# start = "2009-01-01"
# end = "2023-01-01"

# # Streamlit UI
# st.title('📈 Stock Closing Price Prediction')
# user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL, TSLA)', 'GOOGL')

# # Download stock data using yfinance
# df = yf.download(user_input, start=start, end=end)

# # Show basic info
# st.subheader('Stock Data from 1st Jan 2009 to 1st Jan 2023')
# st.write(df.describe())

# # Plot Closing Price vs Time
# st.subheader('Closing Price Vs Time Chart')
# fig1 = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'])
# plt.xlabel("Date")
# plt.ylabel("Closing Price")
# plt.title(f"{user_input} Closing Price Over Time")
# st.pyplot(fig1)

# # 100-day moving average
# ma100 = df['Close'].rolling(100).mean()

# # Plot with 100-day MA
# st.subheader('Closing Price Vs Time Chart with 100-Day Moving Average')
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'], 'r', label='Per Day Closing')
# plt.plot(ma100, 'g', label='100-Day MA')
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# st.pyplot(fig2)

# # 200-day moving average
# ma200 = df['Close'].rolling(200).mean()

# # Plot with 100 and 200-day MA
# st.subheader('Closing Price Vs Time Chart with 100-Day & 200-Day Moving Average')
# fig3 = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'], 'r', label='Per Day Closing')
# plt.plot(ma100, 'g', label='100-Day MA')
# plt.plot(ma200, 'b', label='200-Day MA')
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# st.pyplot(fig3)

# # Preparing training and test data
# train_df = pd.DataFrame(df['Close'][0: int(len(df)*0.85)])
# train_df = train_df.dropna()  # Handle NaN values

# test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):])
# test_df = test_df.dropna()

# scaler = MinMaxScaler(feature_range=(0, 1))
# train_df_arr = scaler.fit_transform(train_df)

# # Load the pre-trained model
# model = load_model('keras_model.h5')

# # Prepare input for prediction
# past_100_days = train_df.tail(100)
# final_df = past_100_days._append(test_df, ignore_index=True)
# final_df = final_df.dropna()
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100:i])
#     y_test.append(input_data[i, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)

# # Make predictions
# y_pred = model.predict(x_test)

# # Rescale values to original range
# scale = scaler.scale_
# scale_factor = 1 / scale[0]
# y_pred = y_pred * scale_factor
# y_test = y_test * scale_factor

# # Final comparison plot
# st.subheader('📊 Predicted Price Vs Original Price')
# fig4 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'g', label="Original Price")
# plt.plot(y_pred, 'r', label="Predicted Price")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.legend()
# st.pyplot(fig4)
# ----------------------------------------------------------------------------------


# Stock Price Prediction App with Streamlit and LSTM

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import yfinance as yf
# import streamlit as st

# # App title
# st.title("📈 Stock Closing Price Prediction")

# # Select ticker
# ticker = st.selectbox("Choose a Stock Ticker", ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NFLX'])

# # Load data
# start = "2009-01-01"
# end = "2024-01-01"
# df = yf.download(ticker, start=start, end=end)

# st.subheader("📊 Stock Data from 1st Jan 2009 to 1st Jan 2023")
# st.write(df.describe())

# # Plotting the Closing Price
# st.subheader("📉 Closing Price Vs Time Chart")
# fig = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'], label="Closing Price")
# plt.legend()
# st.pyplot(fig)

# # Moving averages
# ma100 = df['Close'].rolling(100).mean()
# ma200 = df['Close'].rolling(200).mean()

# # Plot with MA 100
# st.subheader("📈 Closing Price with 100-Day Moving Average")
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'], 'r', label="Closing Price")
# plt.plot(ma100, 'g', label="100-Day MA")
# plt.legend()
# st.pyplot(fig2)

# # Plot with MA 100 & 200
# st.subheader("📊 Closing Price with 100 & 200-Day Moving Averages")
# fig3 = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'], 'r', label="Closing Price")
# plt.plot(ma100, 'g', label="100-Day MA")
# plt.plot(ma200, 'b', label="200-Day MA")
# plt.legend()
# st.pyplot(fig3)

# # Prepare data for model
# data = df.filter(['Close'])
# dataset = data.values
# training_data_len = int(np.ceil(len(dataset) * 0.8))

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)

# train_data = scaled_data[0:int(training_data_len)]
# x_train, y_train = [], []

# for i in range(100, len(train_data)):
#     x_train.append(train_data[i - 100:i, 0])
#     y_train.append(train_data[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# # Load trained LSTM model
# try:
#     model = load_model('keras_model.h5')
# except Exception as e:
#     st.error("🚫 Error loading model. Make sure 'keras_model.h5' is present and trained properly.")
#     st.stop()

# # Prepare test data
# test_data = scaled_data[training_data_len - 100:]
# x_test, y_test = [], dataset[training_data_len:]

# for i in range(100, len(test_data)):
#     x_test.append(test_data[i - 100:i, 0])

# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# # Predictions
# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)

# # Final comparison chart
# st.subheader("✅ Predicted vs Actual Closing Prices")
# fig4 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'g', label='Actual Price')
# plt.plot(predictions, 'r', label='Predicted Price')
# plt.legend()
# st.pyplot(fig4)
#--------------------------------------------------------------------

#-----------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# Config & API Setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("Stock Price Prediction")

# Alpha Vantage API (replace with st.secrets in production)
API_KEY = "EEDZES6PG9QMO5NS"
ALPHA_BASE_URL = "https://www.alphavantage.co/query"

# -----------------------------
# Stock Selection
# -----------------------------
ticker = st.selectbox("Choose a Stock Ticker", ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NFLX'])

# -----------------------------
# Fetch Stock Data
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": API_KEY
        }
        response = requests.get(ALPHA_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" in data:
            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df = df.rename(columns={"4. close": "Close"})
            df["Close"] = df["Close"].astype(float)
        else:
            st.warning("Alpha Vantage limit hit. Using yfinance instead.")
            stock = yf.Ticker(ticker)
            df = stock.history(period="max")[['Close']]
        
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[df.index >= "2009-01-01"]

    except Exception as e:
        st.error(f"Failed to fetch {ticker} data: {e}")
        return None

# -----------------------------
# LSTM Model Setup
# -----------------------------
def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(100, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def fallback_predict(data, days=5):
    last_100 = data[-100:].flatten()
    prediction = []

    for _ in range(days):
        next_val = np.mean(last_100[-30:])
        prediction.append(next_val)
        last_100 = np.append(last_100[1:], next_val)

    return scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

# -----------------------------
# Main Logic
# -----------------------------
df = fetch_stock_data(ticker)
if df is None:
    st.stop()

st.subheader("Historical Closing Prices")
st.line_chart(df["Close"])

# -----------------------------
# Moving Averages
# -----------------------------
df['MA100'] = df['Close'].rolling(100).mean()
df['MA200'] = df['Close'].rolling(200).mean()

st.subheader("Moving Averages")
fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
ax_ma.plot(df['Close'], label='Close', alpha=0.5)
ax_ma.plot(df['MA100'], 'g', label='100 MA')
ax_ma.plot(df['MA200'], 'b', label='200 MA')
ax_ma.legend()
st.pyplot(fig_ma)

# -----------------------------
# Prepare Data for Prediction
# -----------------------------
data = df[['Close']]
dataset = data.values
train_len = int(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# -----------------------------
# Load or Train Model
# -----------------------------
model = None
model_label = "None"

if st.checkbox("Enable LSTM Predictions", True):
    try:
        model = load_model('keras_model.h5')
        model_label = "LSTM"
        st.success("Pre-trained LSTM loaded")
    except:
        st.warning("Pre-trained model not found.")
        if st.button("Train New LSTM Model"):
            with st.spinner("Training in progress..."):
                model = build_lstm_model()
                X, y = [], []
                for i in range(100, len(scaled_data)):
                    X.append(scaled_data[i-100:i, 0])
                    y.append(scaled_data[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                model_label = "Newly Trained LSTM"
                st.success("Training complete")

# -----------------------------
# Make Predictions
# -----------------------------
def forecast(data, days=5, model=None):
    if model:
        preds = []
        batch = data[-100:].reshape(1, 100, 1)
        for _ in range(days):
            pred = model.predict(batch, verbose=0)[0]
            preds.append(pred)
            batch = np.append(batch[:, 1:, :], [[pred]], axis=1)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    else:
        return fallback_predict(data, days)

days_to_predict = st.slider("Days to predict:", 1, 30, 5)
predictions = forecast(scaled_data, days_to_predict, model)

# -----------------------------
# Show Prediction Results
# -----------------------------
forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days_to_predict + 1)]
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted Close": predictions.flatten()
}).set_index("Date")

st.subheader(f"{days_to_predict}-Day Prediction ({model_label if model else 'Simple MA'})")
st.dataframe(forecast_df.style.format({"Predicted Close": "${:.2f}"}))

fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
ax_forecast.plot(df.index[-100:], df['Close'][-100:], label='Historical', color='blue')
ax_forecast.plot(forecast_df.index, forecast_df['Predicted Close'], 'ro--', label='Predicted')
ax_forecast.set_title(f'{ticker} Forecast')
ax_forecast.set_xlabel('Date')
ax_forecast.set_ylabel('Price ($)')
ax_forecast.legend()
ax_forecast.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig_forecast)

# -----------------------------
# Prediction Summary
# -----------------------------
st.subheader("Prediction Summary")
current_price = df['Close'][-1]
next_price = predictions[0][0]
delta = next_price - current_price
delta_pct = (delta / current_price) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Next Day Prediction", f"${next_price:.2f}", f"{delta_pct:.2f}%", delta_color="inverse")
col3.metric("Prediction Model", model_label if model else "Moving Average")

st.info(f"""
Prediction Info:
- Model Used: {model_label if model else 'Simple Moving Average'}
- Forecast Length: {days_to_predict} day(s)
- Last Historical Date: {df.index[-1].strftime('%Y-%m-%d')}
- Forecast Range: {forecast_dates[0].strftime('%Y-%m-%d')} → {forecast_dates[-1].strftime('%Y-%m-%d')}
""")

# -----------------------------
# Stats for Multi-Day Forecasts
# -----------------------------
if days_to_predict > 1:
    st.subheader("Forecast Stats")
    
    high, low, avg = np.max(predictions), np.min(predictions), np.mean(predictions)
    gain, loss = high - current_price, low - current_price

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("High", f"${high:.2f}")
    stat2.metric("Low", f"${low:.2f}")
    stat3.metric("Average", f"${avg:.2f}")

    col_gain, col_loss = st.columns(2)
    col_gain.metric("Potential Gain", f"${gain:.2f}", f"{(gain / current_price) * 100:.2f}%")
    col_loss.metric("Potential Loss", f"${abs(loss):.2f}", f"-{abs(loss / current_price) * 100:.2f}%", delta_color="inverse")

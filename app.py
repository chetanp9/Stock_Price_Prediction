import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to download historical stock price data
def download_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    # Calculate Exponential Moving Averages (EMA)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Calculate Moving Average Convergence Divergence (MACD)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data

# Function to train prediction model
def train_prediction_model(data):
    X = data[['EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    st.write('Mean Squared Error:', mse)
    st.write('Mean Absolute Error:', mae)

    return model

# Function to make predictions
def predict_prices(model, data):
    X_pred = data[['EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI']]
    predictions = model.predict(X_pred)
    return predictions

# Streamlit UI
st.title('Stock Price Prediction with Technical Indicators')

# Sidebar for user inputs
st.sidebar.header('User Inputs')
stock_symbol = st.sidebar.text_input('Enter stock symbol', 'GOOG')
start_date = st.sidebar.text_input('Enter start date (YYYY-MM-DD)', '2010-01-01')
end_date = st.sidebar.text_input('Enter end date (YYYY-MM-DD)', '2022-01-01')
chart_type = st.sidebar.selectbox('Select Chart Type', ['Line Chart', 'Candlestick Chart'])

# Download stock data and calculate indicators
data = download_stock_data(stock_symbol, start_date, end_date)
data_with_indicators = calculate_technical_indicators(data)

# Train prediction model
model = None
train_model = st.sidebar.button('Train Model')

if train_model:
    model = train_prediction_model(data_with_indicators)

# Make predictions
predicted_prices = None
if model is not None:
    predicted_prices = predict_prices(model, data_with_indicators)

# Display stock data with indicators and predictions
st.subheader('Stock Data with Technical Indicators and Predictions')
st.write(data_with_indicators)
if predicted_prices is not None:
    st.write('Predicted Prices:', predicted_prices)

# Plot technical indicators and predictions
fig = go.Figure()

# Plot Closing Price
if chart_type == 'Candlestick Chart':
    fig.add_trace(go.Candlestick(x=data_with_indicators.index,
                                 open=data_with_indicators['Open'],
                                 high=data_with_indicators['High'],
                                 low=data_with_indicators['Low'],
                                 close=data_with_indicators['Close'],
                                 name='Candlestick'))
else:
    fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Close'], name='Close'))

# Plot EMA
fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['EMA_12'], name='EMA 12'))
fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['EMA_26'], name='EMA 26'))

# Plot MACD and Signal line
fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD'], name='MACD'))
fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD_Signal'], name='MACD Signal'))

# Plot RSI
fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['RSI'], name='RSI'))

# Add Predicted Prices
if predicted_prices is not None:
    fig.add_trace(go.Scatter(x=data_with_indicators.index, y=predicted_prices, mode='lines', name='Predicted Prices'))

# Update layout
fig.update_layout(title='Stock Price and Technical Indicators',
                  xaxis_title='Date',
                  yaxis_title='Value',
                  width=1000,
                  height=600,
                  xaxis=dict(
                      rangeselector=dict(
                          buttons=list([
                              dict(count=1, label='1m', step='month', stepmode='backward'),
                              dict(count=6, label='6m', step='month', stepmode='backward'),
                              dict(count=1, label='YTD', step='year', stepmode='todate'),
                              dict(count=1, label='1y', step='year', stepmode='backward'),
                              dict(step='all')
                          ])
                      ),
                      rangeslider=dict(
                          visible=True
                      ),
                      type='date'
                  ),
                  yaxis=dict(
                      fixedrange=False  # Enable up-down scrolling
                  ),
                  dragmode='zoom')  # Enable zooming with the mouse scroll wheel

# Display plot
st.plotly_chart(fig, use_container_width=True)

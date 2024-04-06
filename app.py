import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

model = load_model(r"C:\Users\chetan\OneDrive\Desktop\j\Stock Predictions Model1.keras")

st.header('Stock Market Predictor')
stock = st.text_input('Enter stock symbol', 'GOOG')
start = '2012-01-01'
end = '2024-02-14'
try:
    # Attempt to download data from yfinance
    data = yf.download(stock, start, end)

    # Check if data is not empty
    if not data.empty:
        st.subheader('Stock Data')
        st.write(data)

        data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

        # Check if data_test is not empty
        if not data_test.empty:
            scaler = MinMaxScaler(feature_range=(0, 1))

            pas_100_days = data_train.tail(100)
            data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
            data_test_scale = scaler.fit_transform(data_test)

            ma_50_days = data.Close.rolling(50).mean()
            ma_100_days = data.Close.rolling(100).mean()
            ma_200_days = data.Close.rolling(200).mean()

            x = []
            y = []
            for i in range(100, data_test_scale.shape[0]):
                x.append(data_test_scale[i - 100:i])
                y.append(data_test_scale[i, 0])

            x = np.array(x)
            y = np.array(y)

            predict = model.predict(x)

            scale = 1 / scaler.scale_
            predict = predict * scale
            y = y * scale

            # Create a single Plotly figure
            fig = go.Figure()

            # Add traces for each line
            fig.add_trace(go.Scatter(x=data.index, y=ma_50_days, name='MA50'))
            fig.add_trace(go.Scatter(x=data.index, y=ma_100_days, name='MA100'))
            fig.add_trace(go.Scatter(x=data.index, y=ma_200_days, name='MA200'))
            fig.add_trace(go.Scatter(x=data.index, y=data.Close, name='Close'))
            fig.add_trace(go.Scatter(x=data.index[-len(predict):], y=predict[:, 0], name='Original Price'))
            fig.add_trace(go.Scatter(x=data.index[-len(y):], y=y, name='Predicted Price'))

            # Update layout
            fig.update_layout(title='Stock Data with Moving Averages and Predictions',
                            xaxis_title='Date',
                            yaxis_title='Price')

            st.plotly_chart(fig)
        else:
            st.error("Test data is empty. Please check your data range.")
    else:
        st.error("Data is empty. Please check your stock symbol or date range.")
except Exception as e:
    st.error(f"An error occurred: {e}")

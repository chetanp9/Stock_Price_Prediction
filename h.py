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

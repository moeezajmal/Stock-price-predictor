# stock_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf

def load_data(ticker):
    """Loads stock data from Yahoo Finance."""
    stock_data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
    return stock_data

def prepare_data(stock_data):
    """Prepares data for model training."""
    stock_data['Prediction'] = stock_data['Close'].shift(-30)  # Predicting 30 days into the future
    X = np.array(stock_data.drop(['Prediction'], axis=1))[:-30]  # Features
    y = np.array(stock_data['Prediction'])[:-30]  # Target variable
    return X, y

def train_model(X, y):
    """Splits data into training and test sets and trains a linear regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict(model, X_test):
    """Generates predictions using the trained model."""
    predictions = model.predict(X_test)
    return predictions

def plot_results(predictions, y_test):
    """Plots the actual vs predicted stock prices."""
    plt.figure(figsize=(10,6))
    plt.plot(y_test, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend(loc='best')
    plt.title('Stock Price Prediction vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL, MSFT): ")
    
    # Load and prepare the data
    stock_data = load_data(ticker)
    X, y = prepare_data(stock_data)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Predict and plot the results
    predictions = predict(model, X_test)
    plot_results(predictions, y_test)

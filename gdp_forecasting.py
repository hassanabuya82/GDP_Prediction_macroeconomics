import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

def load_and_prepare_data(file_path):
    """Load and prepare GDP data for forecasting."""
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df

def plot_gdp(df):
    """Plot historical GDP data."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["GDP"], label="Actual GDP", color='blue')
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.title("GDP Trend Over Time")
    plt.legend()
    plt.show()

def train_arima_model(df):
    """Train ARIMA model for GDP forecasting."""
    auto_model = auto_arima(df["GDP"], seasonal=False, trace=True, suppress_warnings=True)
    print(auto_model)
    # p, d, q = auto_model.order
    model = ARIMA(df["GDP"], order=(5, 2, 1))
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model

def forecast_gdp(fitted_model, steps=5):
    """Forecast future GDP values."""
    forecast = fitted_model.forecast(steps=steps)
    return forecast

def plot_forecast(df, forecast, forecast_steps=5):
    """Plot historical GDP and forecasted GDP."""
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='YE')[1:]
    
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["GDP"], label="Historical GDP", color="blue")
    plt.plot(future_dates, forecast, label="Forecasted GDP", color="red", linestyle="dashed")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.title(f"GDP Forecast for Next {forecast_steps} Years")
    plt.legend()
    plt.show()

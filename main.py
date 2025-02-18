from gdp_forecasting import load_and_prepare_data, plot_gdp, train_arima_model, forecast_gdp, plot_forecast

# File path to the GDP data (replace with your actual file)
file_path = "gdp_data.csv"

# Step 1: Load and Prepare Data
df = load_and_prepare_data(file_path)

# Step 2: Plot Historical GDP
plot_gdp(df)

# Step 3: Train ARIMA Model
fitted_model = train_arima_model(df)

# Step 4: Forecast Future GDP
forecast = forecast_gdp(fitted_model, steps=5)

# Step 5: Plot Forecasted GDP
plot_forecast(df, forecast, forecast_steps=5)

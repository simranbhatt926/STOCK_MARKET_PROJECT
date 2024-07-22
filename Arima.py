import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('KOTAKBANK.NS', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())
data = data[["Date", "Close"]]
print(data.head())
import matplotlib.pyplot as plt # type: ignore
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Close"])
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore

result = seasonal_decompose(data["Close"], model='multiplicative', period=30)
fig = result.plot()
fig.set_size_inches(15, 10)

pd.plotting.autocorrelation_plot(data["Close"])
from statsmodels.graphics.tsaplots import plot_pacf # type: ignore
plot_pacf(data["Close"], lags = 100)
from statsmodels.tsa.arima.model import ARIMA # type: ignore

p, d, q = 5, 1, 2
model = ARIMA(data["Close"], order=(p, d, q))
fitted = model.fit()
print(fitted.summary())

# 1,1,1 ARIMA Model
model = ARIMA(data["Close"], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
from statsmodels.tsa.stattools import acf # type: ignore
import pandas as pd

# Example DataFrame for context
# data = pd.DataFrame({
#     'value': [...],  # your data here
# })

# Create Training and Test
train = data['Close'][:85]
test = data['Close'][85:]

# Fit the model
model = ARIMA(train, order=(p, d, q))  # specify your p, d, q values
fitted = model.fit()

# Forecast
forecast_result = fitted.forecast(steps=119, alpha=0.05)  # 95% conf

# Assuming the forecast returns four values, unpack only the first three
fc, se, conf, *_ = forecast_result

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)

# Proceed with further processing of forecast results


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')

plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Build Model
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit()  
print(fitted.summary())

forecast_result = fitted.get_forecast(steps=len(test), alpha=0.05)  # 95% conf

# Extract the forecast mean, confidence intervals
fc = forecast_result.predicted_mean
conf = forecast_result.conf_int()

# Create the lower and upper series for the confidence intervals
lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
upper_series = pd.Series(conf.iloc[:, 1], index=test.index)

# Make the forecast series
fc_series = pd.Series(fc, index=test.index)


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
import numpy as np
# Accuracy metrics
# Define the forecast_accuracy function
def forecast_accuracy(forecast, actual):
    forecast_np = np.array(forecast)
    actual_np = np.array(actual)
    
    mape = np.mean(np.abs(forecast_np - actual_np) / np.abs(actual_np))  # MAPE
    me = np.mean(forecast_np - actual_np)  # ME
    mae = np.mean(np.abs(forecast_np - actual_np))  # MAE
    mpe = np.mean((forecast_np - actual_np) / actual_np)  # MPE
    rmse = np.mean((forecast_np - actual_np) * 2) * 0.5  # RMSE
    corr = np.corrcoef(forecast_np, actual_np)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast_np[:, None], actual_np[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast_np[:, None], actual_np[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    acf1 = acf(forecast_np - actual_np)[1]  # ACF1
    return {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'rmse': rmse, 'acf1': acf1, 'corr': corr, 'minmax': minmax}

# Calculate forecast accuracy
accuracy = forecast_accuracy(fc, test.values)
print(accuracy)
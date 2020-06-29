import pandas as pd
from fbprophet import Prophet
from statsmodels.tsa.stattools import kpss, adfuller
import matplotlib.pyplot as plt

# Region Visualization
def plot_time_series_with_rolling(x: pd.Series, period: int, *args, **kwargs):
    figsize =  kwargs.get('figsize', (16, 8))
    title =  kwargs.get('title', x.name)
    ylabel = kwargs.get('ylabel', '')
    ax = x.plot(figsize=figsize, label=x.name, title=title)
    x.rolling(period).mean().plot(label=f'{period} mave', color='red', ax=ax)
    x.rolling(period).std().plot( label=f'{period} mstd', color='orange', ax=ax)
    ax.set_ylabel(ylabel)
    plt.legend()
    return ax


def plot_seas_model(model, *args, **kwargs):
    figsize = kwargs.get("figsize", (16,12))
    title = kwargs.get("title", "")
    ylabel = kwargs.get("ylabel", "")
    fig, axs = plt.subplots(3,1, sharex=True, figsize=figsize)
    ax1,ax2,ax3 = axs
    plt.suptitle(f'{title} Time Series Decomposition')
    model.observed.plot(label='observed', legend=True, ax=ax1)
    model.trend.plot(label='trend', legend=True, ax=ax1)
    (model.observed- model.seasonal).plot(label='seas. adj.', legend=True, ax=ax1)
    model.seasonal.plot(label='seasonal', legend=True, ax=ax2)
    (model.observed-model.trend).plot(legend=True, label='detrended', ax=ax2)
    model.resid.plot(legend=True, label='resids', ax=ax3)
    
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    ax3.set_ylabel(ylabel)
    return fig, axs

# Region Statistical Analysis
def wrap_adfuller(data, regression='c'):
    adf_res_dict = {}
    for col in data.columns:
        adf_res = adfuller(data[col], regression =regression)
        adf_res_dict[col] = adf_res
        print(f'{col}: p-value: {adf_res[1]:0.2f}')
    return adf_res_dict

def wrap_kpss(data, regression='c', nlags='auto'):
    kpss_res_dict = {}
    for col in data.columns:
        kpss_res = kpss(data[col], regression =regression)
        kpss_res_dict[col] = kpss_res
        print(f'{col}: p-value: {kpss_res[1]:0.2f}')
    return kpss_res_dict

def pairwise_corr_over_time(data, x: str, ys: list, nlags):
    data_cp = data[[x] + ys].copy()
    x_lags = []
    for tau in range(nlags):
        data_cp[f'{x}_lag{tau}'] = data_cp[x].shift(tau)
        x_lags.append(f'{x}_lag{tau}')
    data_cp = data_cp[nlags:].copy()
    corr = data_cp.corr().loc[x_lags, ys]
    return corr


# Region Forecasting
def exog_forecast(train_endog, train_exog, test_endog, test_exog, res):
    # https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_forecasting.html
    nforecasts = 1
    date_fmt = '%Y-%m-%d'
    flg_refit = True

    forecasts = {}
    params = {}
    forecasts_lst = []
    test_size = len(test_endog)
    assert train_endog.shape[0] == train_exog.shape[0]
    assert test_endog.shape[0] == test_exog.shape[0]


    # Save initial forecast
    forecast_idx = train_endog.index[-1].strftime(date_fmt)
    forecasted_tup = res.get_forecast(steps=nforecasts, exog=test_exog.iloc[0], alpha=0.05)
    forecasted_row = forecasted_tup.conf_int()
    forecasted_row[f'p_{train_endog.name}'] = forecasted_tup.predicted_mean
    forecasts_lst.append(forecasted_row)

    # Step through the rest of the sample
    for t in range(0, test_size-1):
        # Update the results by appending the next observation
        updated_endog = test_endog.iloc[t:t+1]
        updated_exog = test_exog.iloc[t:t+1]
        res = res.append(updated_endog, updated_exog, refit=flg_refit)
        
        # Save the new set of forecasts
        forecast_idx = updated_endog.index[0].strftime(date_fmt)
        forecasted_tup = res.get_forecast(steps=nforecasts, exog=test_exog.iloc[t+1:t+2])
        forecasted_row = forecasted_tup.conf_int()
        forecasted_row[f'p_{train_endog.name}'] = forecasted_tup.predicted_mean
        
        forecasts_lst.append(forecasted_row)

    # Combine all forecasts into a dataframe
    forecasts_df = pd.concat(forecasts_lst, sort=False, axis=0)
    
    return res, forecasts_df

def endog_forecast(train_endog, test_endog, res):
    
    # https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_forecasting.html
    nforecasts = 1
    date_fmt = '%Y-%m-%d'
    flg_refit = True

    forecasts = {}
    forecasts_lst = []
    test_size = len(test_endog)


    # Save initial forecast
    forecast_idx = train_endog.index[-1].strftime(date_fmt)
    forecasted_tup = res.get_forecast(steps=nforecasts, alpha=0.05)
    forecasted_row = forecasted_tup.conf_int().join(forecasted_tup.predicted_mean)
    forecasts_lst.append(forecasted_row)

    # Step through the rest of the sample
    for t in range(0, test_size-1):
        # Update the results by appending the next observation
        updated_endog = test_endog.iloc[t:t+1]
        res = res.append(updated_endog, refit=flg_refit)

        # Save the new set of forecasts
        forecast_idx = updated_endog.index[0].strftime(date_fmt)
        forecasted_tup = res.get_forecast(steps=nforecasts)
        forecasted_tup = res.get_forecast(steps=nforecasts, alpha=0.05)
        forecasted_row = forecasted_tup.conf_int().join(forecasted_tup.predicted_mean)
        forecasts_lst.append(forecasted_row)

    # Combine all forecasts into a dataframe
    forecasts_df = pd.concat(forecasts_lst, sort=False, axis=0)
    
    return res, forecasts_df

def prophet_fit_forecast(model, endog, n_forecasts, freq, return_model=True):
    res = model.fit(df=endog)
    future = model.make_future_dataframe(periods=n_forecasts, freq=freq)
    if return_model:
        return res, res.predict(future)
    else:
          return res.predict(future)
        
def prophet_endog_forecast(train_endog, test_endog, *args, **kwargs):
    nforecasts = 1
    freq = 'M'
    date_fmt = '%Y-%m-%d'

    forecasts_lst = []
    forecast_mods = {}
    test_init = len(train_endog)
    test_size = len(test_endog)

    for t in range(0, test_size):
        forecast_app_idx = test_init+t
        forecast_dt = test_endog.index[t]
        mod_prophet = Prophet(weekly_seasonality=False, daily_seasonality=False, seasonality_mode='additive')
        app_df = pd.concat([train_endog, test_endog.iloc[:t+1]], axis=0)
        fitted_mod, forecasted_df = prophet_fit_forecast(mod_prophet, endog=app_df,
                                              n_forecasts=nforecasts, freq=freq)
        forecasted_row = forecasted_df.iloc[forecast_app_idx:forecast_app_idx+1]
        forecasts_lst.append(forecasted_row)
        forecast_mods[forecast_dt] = fitted_mod
    # Combine all forecasts into a dataframe
    forecasts_df = pd.concat(forecasts_lst, sort=False, axis=0)
    return forecast_mods, forecasts_df


def compute_ape(y_true, y_pred):
    resid = y_true-y_pred
    rabs = abs(resid)
    pct_rabs = rabs/y_true
    return pct_rabs


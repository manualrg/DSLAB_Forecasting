import pandas as pd
from fbprophet import Prophet

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


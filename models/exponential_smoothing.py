import numpy as np
import model_params as params
from utils.split_util import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

def ets(demand, validation_points, trend, seasonal):
    model_count = 0
    for param_trend in trend:
        for param_seasonal in seasonal:
            error = np.empty(shape=(0,0))
            for split_count in range(1,validation_points + 2):
                demand_train, demand_valid = split_data(demand, validation_datapoints, split_count)
                try:
                    ets_fit = ExponentialSmoothing(demand_train, trend=param_trend,
                                                   seasonal=param_seasonal, seasonal_periods=c.seasons).fit()
                    ets_fcast = ets_fit.forecast(steps=3)
                    error = mean_squared_error(demand_valid, ets_fcast)
                except:
                    error = float('Inf')
                ets_cv_error = np.append(ets_cv_error, error)
            ets_mean_error = np.nanmean(ets_cv_error)
            if math.isnan(ets_mean_error):
                ets_mean_error = float('Inf')
            if (model_count == 0):
                ets_best_error = ets_mean_error
                ets_best_model = [param_trend, param_seasonal]
            if ets_mean_error < ets_best_error:
                ets_best_error = ets_mean_error
                ets_best_model = [param_trend, param_seasonal]
            model_count = model_count + 1

    ets_best_fit = ExponentialSmoothing(demand,
                                        trend=ets_best_model[0],
                                        seasonal=(None if ets_best_model[1] is None else ets_best_model[1]),
                                        seasonal_periods=params.seasons,
                                        damped=True
                                        ).fit()
    return [ets_best_model, ets_best_error, ets_best_fit]

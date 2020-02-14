import itertools
import traceback
import numpy as np
import models.model_params as params
from statsmodels.tsa.arima_model import ARIMA


def arima(demand, validation_points):
    model_count = 0
    pdq = list(itertools.product(params.p, params.d, params.q))
    for param in pdq:
        arima_cv_error = np.empty(shape=(0, 0))
        if param == pdq[0]:
            continue
        error = np.empty(shape=(0, 0))
        for split_count in range(1, validation_points - 1):
            demand_train, demand_valid = split_data(demand, validation_points, split_count)
            if len(demand_valid) != params.validation_steps:
                break        
            try:
                arima_fit = ARIMA(demand_train, order=(param[0], param[1], param[2])).fit(solver='bfgs',transparams=True,method='mle')
                arima_fcast = arima_fit.predict(start=len(demand_train), end=len(demand_train) + params.validation_steps - 1, typ='levels')
                error = mean_squared_error(demand_valid, arima_fcast)
            except:
                traceback.print_exc()
                arima_cv_error = np.append(arima_cv_error, error)
            arima_mean_error = np.nanmean(arima_cv_error)
        if math.isnan(arima_mean_error):
            arima_mean_error = float('Inf')
        if (model_count == 0):
            arima_best_error = arima_mean_error
            arima_best_model = param
        if arima_mean_error < arima_best_error:
            arima_best_error = arima_mean_error
            arima_best_model = param
        model = model + 1
    arima_best_fit = arima_fit = ARIMA(demand, order=(arima_best_model[0], arima_best_model[1], arima_best_model[2])).fit(solver='bfgs',transparams=True,method='mle')
    return [arima_best_model, arima_best_error, arima_best_fit]
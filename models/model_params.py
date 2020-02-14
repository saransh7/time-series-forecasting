min_months_for_seasonal = 24
seasons = 12

# ETS:
trend_params = ["add", "mul", None]
seasonal_params = ["add", "mul", None]
damped_params = ['True', 'False']
validation_steps = 3

# ARIMA
p = q = range(0, 3)
d = range(0, 2)
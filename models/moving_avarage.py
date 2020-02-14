import numpy as np


class SMA():
    def __init__(self, train_series, window):
        self.train_series = train_series
        self.window = window

    def fit(self):
        if len(self.train_series) >= self.window:
            frc_in = np.empty(shape=(0, 0))
            for i in range(0, len(self.train_series)):
                pred = np.mean(self.train_series[i - self.window:i])
                frc_in = np.append(frc_in, pred)
        else:
            raise ValueError("Series length less than window length")
        return frc_in

    def forecast(self, steps):
        frc_out = np.empty(shape=(0, 0))
        series = self.train_series.copy()
        for temp in range(1, steps + 1):
            pred = np.mean(series[-self.window:])
            series = np.append(series, pred)
            frc_out = np.append(frc_out, pred)
        return frc_out

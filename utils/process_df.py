import pandas as pd
import support.config as c


def impute_missing_dates(df):
    df.Date = pd.to_datetime(df.Date, format=c.date_format)
    df = df.set_index(c.date_index).apply(
        lambda df: df.resample(c.freq).max().fillna(0))  # can use ffill
    # other imputing methods : https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460
    return df

def set_timelines(time_series):
    validation_points = 0
    demand_len = len(time_series)
    if demand_len >= 21:
        validation_points = 6
    elif demand_len > 12:
        validation_points = 4
    elif demand_len >= 6:
        validation_points = 3
    else:
        validation_points = 0
    return demand_len, validation_points

def split_data(y, validpts, splitcnt):
    y_train, y_pred = y[0:(len(y) - validpts + splitcnt - 1)], y[(len(y) - validpts + splitcnt - 1): (
                len(y) - validpts + splitcnt + 2)]
    return y_train, y_pred
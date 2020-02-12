import numpy as np
import support.config as c


def demand_char_quad(adi, cv2):
    if np.isnan(adi) or np.isnan(cv2):
        value = None
    elif adi <= c.adi_threshold and cv2 <= c.cv2_threshold:
        value = 'Smooth'
    elif adi <= c.adi_threshold and cv2 > c.cv2_threshold:
        value = 'Erratic'
    elif adi > c.adi_threshold and cv2 <= c.cv2_threshold:
        value = 'Intermittent'
    elif adi > c.adi_threshold and cv2 > c.cv2_threshold:
        value = 'Lumpy'
    else:
        value = None
    return value

# input : list


def calc_adi_cv2(time_series):
    demands = [temp for temp in time_series if temp != 0]
    std = np.std(demands)
    mean = np.mean(demands)
    di = []  # demand interval
    sum = 1
    for temp in time_series[1:]:
        if temp == 0:
            sum += 1
        else:
            di.append(sum)
            sum = 1
    adi = np.mean(di)
    cv2 = (float(std)/float(mean))**2
    quad_class = demand_char_quad(adi, cv2)
    return adi, cv2, quad_class

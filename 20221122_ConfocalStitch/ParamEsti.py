import numpy as np

def pyr_down_time_esti(img_shape, v_thred=1000000):
    n = 0
    while (img_shape[0] * img_shape[1] / 4**n > v_thred):
        n = n + 1
    return n

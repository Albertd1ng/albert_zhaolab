import numpy as np


def judge_strip_cont(strip_pos, xy_v_num, ovl_thre=0.03):
    strip_num = strip_pos.shape[0]
    strip_cont = np.zeros((strip_num, strip_num), dtype='bool')
    for i in range(strip_num):
        for j in range(strip_num):
            if i > j:
                continue
            if i == j:
                continue
            if np.all(np.abs(strip_pos[i, 0:2] - strip_pos[j, 0:2]) < xy_v_num * np.array([1-ovl_thre, 1-ovl_thre])):
                strip_cont[i, j] , strip_cont[j, i] = True, True
    return strip_cont


def judge_strip_cont_one(i, strip_pos, xy_v_num, ovl_thre=0.03):
    strip_num = strip_pos.shape[0]
    strip_cont_vec = np.zeros(strip_num, dtype='bool')
    for j in range(strip_num):
        if i == j:
            continue
        if np.all(np.abs(strip_pos[i, 0:2] - strip_pos[j, 0:2]) < xy_v_num * np.array([1-ovl_thre, 1-ovl_thre])):
            strip_cont_vec[j] = True
    return strip_cont_vec


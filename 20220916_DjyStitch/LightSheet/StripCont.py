import numpy as np


def judge_strip_cont(strip_pos, xy_v_num, ovl_thre=0.1):
    '''
    Judge if two tiles contact with each other.

    :param strip_pos: np.array((strip_num, 3), dtype='int64').
        the xyz position of each strip.
    :param xy_v_num: np.array(2, dtype='int64').
        the voxel quantity in xy dimensions.
    :param ovl_thre: float.
        the two strips can be considered contacted,
        only if the overlap ratio of two images should exceed ovl_thre in each dimension.
    :return:
        strip_cont: np.array((strip_num, strip_num), dtype='bool').
            strip_cont[i,j] = True if i.th strip and j.th strip contact with each other.
    '''
    strip_num = strip_pos.shape[0]
    strip_cont = np.zeros((strip_num, strip_num), dtype='bool')
    for i in range(strip_num):
        for j in range(strip_num):
            if i > j:
                continue
            if i == j:
                strip_cont[i, j] = False
            if np.all(np.abs(strip_pos[i, 0:2] - strip_pos[j, 0:2]) < xy_v_num * np.array([1-ovl_thre, 1-ovl_thre])):
                strip_cont[i, j] , strip_cont[j, i] = True, True
    return strip_cont

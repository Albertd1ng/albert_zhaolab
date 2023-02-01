import numpy as np


def get_ovl_img(img1, img2, border):
    '''
    Get overlapped image of img1 and img2 according to the border of their overlapped area.

    :param img1: numpy.array(3D).
    :param img2: numpy.array(3D).
    :param border: numpy.array((2, 6), dtype='int64').
        the border of overlapped area of two images.
        format: [[xv1_min, xv1_max, yv1_min, yv1_max, zv1_min, zv1_max],
                [xv2_min, xv2_max, yv2_min, yv2_max, zv2_min, zv2_max]]
    :return: ovl1_list, ovl2_list: list(numpy.array(2D)).
        the list of array to store overlapped area of img1 and img2.
    '''
    if border.shape[0]==0:
        return np.array([]), np.array([])
    for i in range(3):
        for j in range(2):
            if border[j, 2 * i] == border[j, 2 * i + 1]:
                return np.array([]), np.array([])
    ovl1_list, ovl2_list = [], []

    ovl1_list.append(img1[border[0, 2]:border[0, 3], border[0, 0], border[0, 4]:border[0, 5]])
    ovl1_list.append(img1[border[0, 2]:border[0, 3], border[0, 1], border[0, 4]:border[0, 5]])
    ovl1_list.append(img1[border[0, 2], border[0, 0]:border[0, 1], border[0, 4]:border[0, 5]])
    ovl1_list.append(img1[border[0, 3], border[0, 0]:border[0, 1], border[0, 4]:border[0, 5]])
    ovl1_list.append(img1[border[0, 2]:border[0, 3], border[0, 0]:border[0, 1], border[0, 4]])
    ovl1_list.append(img1[border[0, 2]:border[0, 3], border[0, 0]:border[0, 1], border[0, 5]])

    ovl2_list.append(img2[border[1, 2]:border[1, 3], border[1, 0], border[1, 4]:border[1, 5]])
    ovl2_list.append(img2[border[1, 2]:border[1, 3], border[1, 1], border[1, 4]:border[1, 5]])
    ovl2_list.append(img2[border[1, 2], border[1, 0]:border[1, 1], border[1, 4]:border[1, 5]])
    ovl2_list.append(img2[border[1, 3], border[1, 0]:border[1, 1], border[1, 4]:border[1, 5]])
    ovl2_list.append(img2[border[1, 2]:border[1, 3], border[1, 0]:border[1, 1], border[1, 4]])
    ovl2_list.append(img2[border[1, 2]:border[1, 3], border[1, 0]:border[1, 1], border[1, 5]])
    return ovl1_list, ovl2_list

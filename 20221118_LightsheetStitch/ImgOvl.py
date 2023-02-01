import numpy as np


def get_ovl_img(img1, img2, border):
    if border.shape[0] == 0:
        return [], []
    for i in range(3):
        for j in range(2):
            if border[j, 2 * i] == border[j, 2 * i + 1]:
                return [], []
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


def get_ovl_img_max(img1, img2, border):
    if border.shape[0] == 0:
        return [], []
    for i in range(3):
        for j in range(2):
            if border[j, 2 * i] == border[j, 2 * i + 1]:
                return [], []
    ovl1_list, ovl2_list = [], []
    img1_ovl = img1[border[0, 2]:border[0, 3], border[0, 0]:border[0, 1], border[0, 4]:border[0, 5]]
    img2_ovl = img2[border[1, 2]:border[1, 3], border[1, 0]:border[1, 1], border[1, 4]:border[1, 5]]
    if (0 in img1_ovl.shape) or (0 in img2_ovl.shape):
        return [], []
    for i in range(3):
        ovl1_list.append(np.max(img1_ovl, axis=i))
        ovl2_list.append(np.max(img2_ovl, axis=i))
    return ovl1_list, ovl2_list

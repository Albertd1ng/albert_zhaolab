import numpy as np
import cv2
import os
from merge import MergeSolution


def choose_img_roi(img, thred=0):
    img_one_ch = img[:, :, 0]
    roi_range = np.zeros((2, 2), dtype='int64')
    lr_roi_index = [i for [i] in np.argwhere(np.max(img_one_ch, axis=0) > thred)]
    ud_roi_index = [i for [i] in np.argwhere(np.max(img_one_ch, axis=1) > thred)]
    if len(lr_roi_index) == 0 or len(ud_roi_index) == 0:
        return np.array([]), np.array([])
    roi_range[0, 0], roi_range[0, 1] = min(lr_roi_index), max(lr_roi_index)
    roi_range[1, 0], roi_range[1, 1] = min(ud_roi_index), max(ud_roi_index)
    return img[roi_range[1, 0]:roi_range[1, 1], roi_range[0, 0]:roi_range[0, 1], :], roi_range


def npy_max_pro(npy_path, npy_name_format, num):
    whole_npy = np.load(os.path.join(npy_path, npy_name_format % 0))
    for i in range(1, num):
        this_npy = np.load(os.path.join(npy_path, npy_name_format % i))
        whole_npy = np.max(np.stack((this_npy, whole_npy), axis=0), axis=0)
    return whole_npy


def npy_max_pro_merge(npy_path, npy_name_format, num):
    img_with_pos_list1 = []
    img_with_pos_list2 = []
    for i in range(num):
        this_npy = np.load(os.path.join(npy_path, npy_name_format % i))
        if i == 0:
            x_r = this_npy.shape[1]
            y_r = this_npy.shape[0]
        this_sec_npy, roi_range = choose_img_roi(this_npy)
        print(i)
        print(roi_range)
        print(this_sec_npy.shape)
        img_with_pos_list1.append(
            [this_sec_npy[:, :, 0].copy(), [[roi_range[0, 0], roi_range[0, 1]], [roi_range[1, 0], roi_range[1, 1]]]])
        img_with_pos_list2.append(
            [this_sec_npy[:, :, 1].copy(), [[roi_range[0, 0], roi_range[0, 1]], [roi_range[1, 0], roi_range[1, 1]]]])
        this_npy = np.array([])
        this_sec_npy = np.array([])
    this_whole_npy = MergeSolution(img_with_pos_list1, [y_r, x_r]).do()
    whole_npy = np.zeros((this_whole_npy.shape[0], this_whole_npy.shape[1], 3), dtype='uint16')
    whole_npy[:, :, 0] = this_whole_npy
    this_whole_npy = MergeSolution(img_with_pos_list2, [y_r, x_r]).do()
    whole_npy[:, :, 1] = this_whole_npy
    return whole_npy


if __name__ == '__main__':
    npy_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221104slabmaxpro'
    npy_name_format = 'slab%.2d_XZ_maxpro.npy'
    num = 43
    # whole_npy = npy_max_pro(npy_path, npy_name_format, num)
    # np.save(os.path.join(npy_path, 'slab_XZ_maxpro.npy'), whole_npy)
    # whole_npy = np.array([])
    # npy_name_format = 'slab%.2d_YZ_maxpro.npy'
    # whole_npy = npy_max_pro(npy_path, npy_name_format, num)
    # np.save(os.path.join(npy_path, 'slab_YZ_maxpro.npy'), whole_npy)
    whole_npy = npy_max_pro_merge(npy_path, npy_name_format, num)
    np.save(os.path.join(npy_path, 'slab_XZ_maxpro_merge.npy'), whole_npy)

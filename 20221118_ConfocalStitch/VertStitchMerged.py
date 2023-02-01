import numpy as np
import cv2
import os
import random

from FileRename import rename_file_Z_stit
from ImgIO import import_img_2D
from ImgProcess import pyr_down_img, adjust_contrast
from LossFunc import loss_func_z_stitch
from ParamEsti import pyr_down_time_esti


def calc_sift_points_match(img1,img2):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kpts1, des1 = sift.detectAndCompute(img1, None)
    kpts2, des2 = sift.detectAndCompute(img2, None)
    kp1, kp2 = np.float32([kp.pt for kp in kpts1]), np.float32([kp.pt for kp in kpts2])
    if kp1.shape[0] == 0 or kp2.shape[0] == 0:
        return np.array([]), np.array([])
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good_matches.append((m[0].queryIdx, m[0].trainIdx))
    pts1, pts2 = np.float32([kp1[i, :] for (i, _) in good_matches]), np.float32([kp2[j, :] for (_, j) in good_matches])
    return pts1, pts2


def calc_xy_shift_by_RANSAC(img1, img2, pts1, pts2, sample_time=1000):
    count = 0
    matches_num = pts1.shape[0]
    RANSAC_num = np.int64(np.max((np.min((4, matches_num * 0.1)),1)))
    loss_min = np.inf
    xy_shift_min = np.array([np.inf, np.inf])
    while (count < sample_time):
        count += 1
        index_list = random.sample(range(matches_num), RANSAC_num)
        xy_shift_all = pts2[index_list, :] - pts1[index_list, :]
        max_shift, min_shift = np.max(xy_shift_all, axis=0), np.min(xy_shift_all, axis=0)
        if any((max_shift - min_shift) > 100):
            continue
        xy_shift = np.int64(np.round(np.mean(xy_shift_all, axis=0)))
        if all(xy_shift == xy_shift_min):
            continue
        ovl1, ovl2 = img1[np.max((0, -xy_shift[1])):, np.max((0, -xy_shift[0])):], img2[np.max((0, xy_shift[1])):,
                                                                                   np.max((0, xy_shift[0])):]
        x_range, y_range = np.min((ovl1.shape[1], ovl2.shape[1])), np.min((ovl1.shape[0], ovl2.shape[0]))
        ovl1, ovl2 = ovl1[0:y_range, 0:x_range], ovl2[0:y_range, 0:x_range]
        this_loss = loss_func_z_stitch(ovl1, ovl2)
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
    return xy_shift_min, loss_min


def calc_xy_shift_by_BF(img1_down, img2_down, img1, img2, xy_shift, pyr_down_times, if_median_blur, blur_kernel_size):
    if if_median_blur:
        img1, img2 = cv2.medianBlur(img1, blur_kernel_size), cv2.medianBlur(img2, blur_kernel_size)
    img1, img2 = adjust_contrast(img1, img2)
    xy_shift_min = np.zeros(2)
    for i in range(pyr_down_times, -1, -1):
        if i == pyr_down_times:
            img1_calc, img2_calc = img1_down, img2_down
        elif i == 0:
            img1_calc, img2_calc = img1, img2
        else:
            img1_calc, img2_calc = pyr_down_img(img1, i), pyr_down_img(img2, i)
        if i == pyr_down_times:
            loss_min = np.inf
            range_calc = 10
        else:
            xy_shift = xy_shift_min * 2
            xy_shift_min = np.zeros(2)
            loss_min = np.inf
            range_calc = 5
        for x in range(-range_calc, range_calc+1):
            for y in range(-range_calc, range_calc+1):
                this_xy_shift = xy_shift + np.array([x, y], dtype='int64')
                ovl1 = img1_calc[np.max((0, -this_xy_shift[1])):, np.max((0, -this_xy_shift[0])):]
                ovl2 = img2_calc[np.max((0, this_xy_shift[1])):, np.max((0, this_xy_shift[0])):]
                x_range = np.min((ovl1.shape[1], ovl2.shape[1], 5000))
                y_range = np.min((ovl1.shape[0], ovl2.shape[0], 5000))
                ovl1, ovl2 = ovl1[0:y_range, 0:x_range], ovl2[0:y_range, 0:x_range]
                this_loss = loss_func_z_stitch(ovl1, ovl2)
                if this_loss < loss_min:
                    loss_min = this_loss
                    xy_shift_min = this_xy_shift
        print('%d.th pyr down times'%i, xy_shift_min, loss_min)
    return xy_shift_min, loss_min


def update_lower_layer_info_merged(xy_shift, axis_range1, dim_elem_num):
    axis_range2 = np.zeros((2, 2), dtype='int64')
    axis_range2[0, 0], axis_range2[1, 0] = axis_range1[0, 0]-xy_shift[0], axis_range1[1, 0]-xy_shift[1]
    axis_range2[0, 1], axis_range2[1, 1] = axis_range2[0, 0]+dim_elem_num[0], axis_range2[1, 0]+dim_elem_num[1]
    return axis_range2


def start_vertical_stit_merged(i, img_name_format, img_path, info_IO_path, file_name_format, img_name,channel_num,
                               channel_ordinal, img_type, img_data_type, overlap_ratio, vert_step, pyr_down_times,
                               if_median_blur, blur_kernel_size, if_rename_file):
    if img_data_type == 'uint8':
        img_mode = cv2.IMREAD_GRAYSCALE
    elif img_data_type == 'uint16':
        img_mode = cv2.IMREAD_UNCHANGED
    img_path1 = os.path.join(img_path,file_name_format % (i))
    img_path2 = os.path.join(img_path,file_name_format % (i + 1))
    dim_elem_num1, dim_elem_num2 = np.zeros(3, dtype='int64'), np.zeros(3, dtype='int64')
    axis_range1, axis_range2 = np.zeros((2, 2),dtype='int64'), np.zeros((2, 2),dtype='int64')
    first_last_index1, first_last_index2 = np.zeros(2, dtype='int64'), np.zeros(2, dtype='int64')
    dim_elem_num1[2] = np.int64(np.floor(len(os.listdir(img_path1))/channel_num))
    dim_elem_num2[2] = np.int64(np.floor(len(os.listdir(img_path2))/channel_num))
    first_last_index1[1], first_last_index2[1] = dim_elem_num1[2]-1, dim_elem_num2[2]-1
    if if_rename_file:
        if i == 0:
            rename_file_Z_stit(img_name_format, img_path1, img_name, dim_elem_num1[2], channel_num, img_type=img_type)
        rename_file_Z_stit(img_name_format, img_path2, img_name, dim_elem_num2[2], channel_num, img_type=img_type)
    img1 = import_img_2D(img_name_format, img_path1, img_name, dim_elem_num1[2]-1, channel_ordinal, img_type=img_type,
                         img_data_type=img_data_type, img_mode=img_mode)
    dim_elem_num1[0], dim_elem_num1[1] = img1.shape[1], img1.shape[0]
    axis_range1[0, 1], axis_range1[1, 1] = dim_elem_num1[0], dim_elem_num1[1]
    if pyr_down_times == -1:
        pyr_down_times = pyr_down_time_esti(dim_elem_num1[0:2])
    if i == 0:
        np.save(os.path.join(info_IO_path, 'dim_elem_num_zstitch_%.4d.npy' % (i)), dim_elem_num1)
        np.save(os.path.join(info_IO_path, 'axis_range_zstitch_%.4d.npy' % (i)), axis_range1)
        np.save(os.path.join(info_IO_path, 'first_last_index_zstitch_%.4d.npy' % (i)), first_last_index1)
    loss_min = np.inf
    index_min = -1
    xy_shift_min = np.zeros(0)
    for j in range(0, np.int64(dim_elem_num2[2] * overlap_ratio), vert_step):
        img2 = import_img_2D(img_name_format, img_path2, img_name, j, channel_ordinal,
                             img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
        if j == 0:
            dim_elem_num2[0], dim_elem_num2[1] = img2.shape[1], img2.shape[0]
            axis_range2[0, 1], axis_range2[1, 1] = dim_elem_num2[0], dim_elem_num2[1]
        img2_down = pyr_down_img(img2, pyr_down_times)
        img1_down = pyr_down_img(img1, pyr_down_times)
        img1_down, img2_down = adjust_contrast(img1_down, img2_down)
        pts1, pts2 = calc_sift_points_match(img1_down, img2_down)
        if pts1.shape[0] == 0:
            continue
        xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift,
                                                  pyr_down_times, if_median_blur, blur_kernel_size)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
            index_min = j

    for j in range(np.max((index_min - vert_step-1, 0)), np.min((index_min + vert_step, dim_elem_num2[2]-1))):
        img2 = import_img_2D(img_name_format, img_path2, img_name, j, channel_ordinal, img_type=img_type,
                             img_data_type=img_data_type, img_mode=img_mode)
        img2_down = pyr_down_img(img2, pyr_down_times)
        img1_down = pyr_down_img(img1, pyr_down_times)
        img1_down, img2_down = adjust_contrast(img1_down, img2_down)
        pts1, pts2 = calc_sift_points_match(img1_down, img2_down)
        if pts1.shape[0]==0:
            continue
        xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift,
                                                  pyr_down_times, if_median_blur, blur_kernel_size)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
            index_min = j
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Finally the matched one is %d.th layer, xy_shift is %s, loss is %.8f' % (
        index_min, str(xy_shift_min), loss_min))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    first_last_index2[0]=index_min
    axis_range2 = update_lower_layer_info_merged(xy_shift_min,np.load(os.path.join(info_IO_path,'axis_range_zstitch_%.4d.npy'%i)),dim_elem_num2)

    print('start saving data')
    np.save(os.path.join(info_IO_path, 'dim_elem_num_zstitch_%.4d.npy' % (i+1)), dim_elem_num2)
    np.save(os.path.join(info_IO_path, 'axis_range_zstitch_%.4d.npy' % (i+1)), axis_range2)
    np.save(os.path.join(info_IO_path, 'first_last_index_zstitch_%.4d.npy' % (i+1)), first_last_index2)
    print('end saving data')

import numpy as np
import cv2
import os
import random
import nd2
from readlif.reader import LifFile

from FileRename import rename_file_Z_stit, pop_other_type_file
from ImgIO import import_img_2D, get_img_2D_from_nd2_vert
from ImgProcess import pyr_down_img, adjust_contrast
from LossFunc import loss_func_z_stitch
from ParamEsti import pyr_down_time_esti
from InfoIO import get_img_nd2_info_vert


def calc_sift_points_match(img1, img2):
    if img1.dtype == 'uint16':
        img1 = np.uint8(np.floor((img1.astype('float32') / 65535 * 255)))
        img2 = np.uint8(np.floor((img2.astype('float32') / 65535 * 255)))
    # cv2.imshow('1',img1)
    # cv2.imshow('2',img2)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
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


def calc_xy_shift_by_BF(img1_down, img2_down, img1, img2, xy_shift, pyr_down_times, if_high_noise):
    if if_high_noise:
        blur_kernel_size = 5
    else:
        blur_kernel_size = 3
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
                # print(xy_shift, this_xy_shift)
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


def start_vertical_stit_merged(i, img_name_format, img_path, info_IO_path, file_name_format, img_name, ch_num,
                               ch_th, img_type, img_data_type, overlap_ratio, pyr_down_times,
                               if_high_noise, if_rename_file):
    if img_data_type == 'uint8':
        img_mode = cv2.IMREAD_GRAYSCALE
    elif img_data_type == 'uint16':
        img_mode = cv2.IMREAD_UNCHANGED
    if img_type in ['nd2', 'lif']:
        img_path1 = os.path.join(img_path,file_name_format % (i)) + '.' + img_type
        img_path2 = os.path.join(img_path,file_name_format % (i + 1)) + '.' + img_type
    elif img_type == 'tif':
        img_path1 = os.path.join(img_path, file_name_format % (i))
        img_path2 = os.path.join(img_path, file_name_format % (i + 1))

    dim_elem_num1, dim_elem_num2 = np.zeros(3, dtype='int64'), np.zeros(3, dtype='int64')
    if img_type == 'nd2':
        whole_img1 = nd2.ND2File(img_path1)
        dim_elem_num1, dim_num1, _ = get_img_nd2_info_vert(whole_img1)
        whole_img1.close()
        whole_img2 = nd2.ND2File(img_path2)
        dim_elem_num2, dim_num2, _ = get_img_nd2_info_vert(whole_img2)
        whole_img2.close()
        whole_img1 = nd2.imread(img_path1)
        img1 = get_img_2D_from_nd2_vert(whole_img1, ch_num, ch_th, dim_elem_num1[2] - 1)
        whole_img2 = nd2.imread(img_path2)
    elif img_type == 'lif':
        pass
    elif img_type == 'tif':
        file_list1 = pop_other_type_file(img_path1, os.listdir(img_path1), img_type)
        file_list2 = pop_other_type_file(img_path2, os.listdir(img_path2), img_type)
        dim_elem_num1[2] = np.int64(np.floor(len(file_list1)/ch_num))
        dim_elem_num2[2] = np.int64(np.floor(len(file_list2)/ch_num))
        if if_rename_file:
            if i == 0:
                rename_file_Z_stit(img_name_format, img_path1, img_name, dim_elem_num1[2], ch_num,
                                   img_type=img_type)
            rename_file_Z_stit(img_name_format, img_path2, img_name, dim_elem_num2[2], ch_num, img_type=img_type)
        img1 = import_img_2D(img_name_format, img_path1, img_name, dim_elem_num1[2] - 1, ch_th,
                             img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
        dim_elem_num1[0], dim_elem_num1[1] = img1.shape[1], img1.shape[0]
    axis_range1, axis_range2 = np.zeros((2, 2), dtype='int64'), np.zeros((2, 2), dtype='int64')
    first_last_index1, first_last_index2 = np.zeros(2, dtype='int64'), np.zeros(2, dtype='int64')
    axis_range1[0, 1], axis_range1[1, 1] = dim_elem_num1[0], dim_elem_num1[1]
    axis_range2[0, 1], axis_range2[1, 1] = dim_elem_num2[0], dim_elem_num2[1]
    first_last_index1[1], first_last_index2[1] = dim_elem_num1[2] - 1, dim_elem_num2[2] - 1

    if pyr_down_times == -1:
        pyr_down_times = pyr_down_time_esti(dim_elem_num1[0:2])
    if i == 0:
        np.save(os.path.join(info_IO_path, 'dim_elem_num_zstitch_%.4d.npy' % (i)), dim_elem_num1)
        np.save(os.path.join(info_IO_path, 'axis_range_zstitch_%.4d.npy' % (i)), axis_range1)
        np.save(os.path.join(info_IO_path, 'first_last_index_zstitch_%.4d.npy' % (i)), first_last_index1)
    loss_min = np.inf
    index_min = -1
    xy_shift_min = np.zeros(0)
    img2_range = np.int64(dim_elem_num2[2] * overlap_ratio)
    if img2_range < 30:
        vert_step = 3
    else:
        vert_step = 5
    for j in range(0, img2_range, vert_step):
        if img_type == 'nd2':
            img2 = get_img_2D_from_nd2_vert(whole_img2, ch_num, ch_th, j)
        elif img_type == 'lif':
            pass
        elif img_type == 'tif':
            img2 = import_img_2D(img_name_format, img_path2, img_name, j, ch_th,
                                 img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
            if j == 0:
                dim_elem_num2[0], dim_elem_num2[1] = img2.shape[1], img2.shape[0]
                axis_range2[0, 1], axis_range2[1, 1] = dim_elem_num2[0], dim_elem_num2[1]
        img2_down = pyr_down_img(img2, pyr_down_times)
        img1_down = pyr_down_img(img1, pyr_down_times)
        img1_down, img2_down = adjust_contrast(img1_down, img2_down, 3000)
        pts1, pts2 = calc_sift_points_match(img1_down.copy(), img2_down.copy())
        if pts1.shape[0] == 0:
            continue
        xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        if not any(np.isinf(xy_shift)):
            xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift,
                                                      pyr_down_times, if_high_noise)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
            index_min = j

    for j in range(np.max((index_min - vert_step-1, 0)), np.min((index_min + vert_step, dim_elem_num2[2]-1))):
        if img_type == 'nd2':
            img2 = get_img_2D_from_nd2_vert(whole_img2, ch_num, ch_th, j)
        elif img_type == 'lif':
            pass
        elif img_type == 'tif':
            img2 = import_img_2D(img_name_format, img_path2, img_name, j, ch_th,
                                 img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
            if j == 0:
                dim_elem_num2[0], dim_elem_num2[1] = img2.shape[1], img2.shape[0]
                axis_range2[0, 1], axis_range2[1, 1] = dim_elem_num2[0], dim_elem_num2[1]
        img2_down = pyr_down_img(img2, pyr_down_times)
        img1_down = pyr_down_img(img1, pyr_down_times)
        img1_down, img2_down = adjust_contrast(img1_down, img2_down, 3000)
        pts1, pts2 = calc_sift_points_match(img1_down.copy(), img2_down.copy())
        if pts1.shape[0] == 0:
            continue
        xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift,
                                                  pyr_down_times, if_high_noise)
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

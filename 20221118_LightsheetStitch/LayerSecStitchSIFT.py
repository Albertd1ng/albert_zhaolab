import numpy as np
import os
import cv2
import random
import ctypes
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_strip_pos_info, get_id_dir_dict_info, get_layer_id_dict_info, get_img_dim_info, save_img_txt_info
from RunningIO import run_print
from LayerStitchSIFT import update_next_layer_info, choose_layer_index, choose_img_roi, calc_sift_points_match
from LayerStitchSIFT import calc_xy_shift_by_RANSAC, calc_xy_shift_by_BF
from ImgProcess import pyr_down_img, adjust_contrast
from LossFunc import loss_func


def layer_sec_stitch_sift(lock, txt_path, data_IO_path, running_IO_path, layer_id_dict, if_layer_stitch, y_range,
                          z_range, step, pyr_down_times):
    while False in if_layer_stitch:
        lock.acquire()
        i = choose_layer_index(if_layer_stitch)
        if i == -1:
            lock.release()
            return
        if_layer_stitch[i] = True
        img1_YZ_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_y%.6d-%.6d_z%.6d-%.6d.npy' % (i - 1, y_range[0], y_range[1], z_range[0], z_range[1])))
        img2_YZ_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_y%.6d-%.6d_z%.6d-%.6d.npy' % (i, y_range[0], y_range[1], z_range[0], z_range[1])))
        img1_pos_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_y%.6d-%.6d_z%.6d-%.6d_pos.npy' % (i - 1, y_range[0], y_range[1], z_range[0], z_range[1])))
        img2_pos_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_y%.6d-%.6d_z%.6d-%.6d_pos.npy' % (i, y_range[0], y_range[1], z_range[0], z_range[1])))
        lock.release()
        xyz_shift_real = np.zeros(3, dtype='int64')
        loss_min = np.inf
        j_m, k_m = -1, -1
        for j in range(0, img1_YZ_arr.shape[2], step):
            for k in range(0, img2_YZ_arr.shape[2], step):
                img1 = img1_YZ_arr[:, :, j].copy()
                img2 = img2_YZ_arr[:, :, k].copy()
                img1, roi_range1 = choose_img_roi(img1, 1)
                img2, roi_range2 = choose_img_roi(img2, 1)
                if img1.shape[0] == 0 or img2.shape[0] == 0:
                    continue
                img1_down = pyr_down_img(img1, pyr_down_times)
                img2_down = pyr_down_img(img2, pyr_down_times)
                pts1, pts2 = calc_sift_points_match(img1_down.copy(), img2_down.copy())
                if pts1.shape[0] == 0:
                    continue
                xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
                if np.isinf(this_loss) or np.any(np.isinf(xy_shift)):
                    continue
                xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift, pyr_down_times)
                if np.isinf(this_loss) or np.any(np.isinf(xy_shift)):
                    continue
                xy_shift[0] = xy_shift[0] + img2_pos_arr[k, 2] - img1_pos_arr[j, 2] + roi_range2[0, 0] - roi_range1[0, 0]  # z
                xy_shift[1] = xy_shift[1] + img2_pos_arr[k, 1] - img1_pos_arr[j, 1] + roi_range2[1, 0] - roi_range1[1, 0]  # y
                if this_loss < loss_min and xy_shift[0] < 500 and xy_shift[1] < 500:
                    loss_min = this_loss
                    j_m, k_m = j, k
                    xyz_shift_real[2] = xy_shift[0]  # z
                    xyz_shift_real[1] = xy_shift[1]  # y
                    xyz_shift_real[0] = img2_pos_arr[k, 0] - img1_pos_arr[j, 0]  # x
        run_print(lock, running_IO_path,
                   '%.2d.th whole_img of %.2d.th layer, with %.2d.th whole_img of %.2d.th layer, zy_shift is %s, loss is %.8f' % (
                   j_m, i-1, k_m, i, str(xy_shift), this_loss))
        for j in range(max(0, j_m-step+1), min(img1_YZ_arr.shape[2], j_m+step)):
            for k in range(max(0, k_m-step+1), min(img2_YZ_arr.shape[2], k_m+step)):
                img1 = img1_YZ_arr[:, :, j].copy()
                img2 = img2_YZ_arr[:, :, k].copy()
                img1, roi_range1 = choose_img_roi(img1, 1)
                img2, roi_range2 = choose_img_roi(img2, 1)
                if img1.shape[0] == 0 or img2.shape[0] == 0:
                    continue
                img1_down = pyr_down_img(img1, pyr_down_times)
                img2_down = pyr_down_img(img2, pyr_down_times)
                pts1, pts2 = calc_sift_points_match(img1_down.copy(), img2_down.copy())
                if pts1.shape[0] == 0:
                    continue
                xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
                if np.isinf(this_loss) or np.any(np.isinf(xy_shift)):
                    continue
                xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift, pyr_down_times)
                if np.isinf(this_loss) or np.any(np.isinf(xy_shift)):
                    continue
                xy_shift[0] = xy_shift[0] + img2_pos_arr[k, 2] - img1_pos_arr[j, 2] + roi_range2[0, 0] - roi_range1[0, 0]  # z
                xy_shift[1] = xy_shift[1] + img2_pos_arr[k, 1] - img1_pos_arr[j, 1] + roi_range2[1, 0] - roi_range1[1, 0]  # y
                run_print(lock, running_IO_path,
                          '%.2d.th whole_img of %.2d.th layer, with %.2d.th whole_img of %.2d.th layer, zy_shift is %s, loss is %.8f' % (
                              j, i - 1, k, i, str(xy_shift), this_loss))
                if this_loss < loss_min and xy_shift[0] < 500 and xy_shift[1] < 500:
                    loss_min = this_loss
                    j_m, k_m = j, k
                    xyz_shift_real[2] = xy_shift[0]  # z
                    xyz_shift_real[1] = xy_shift[1]  # y
                    xyz_shift_real[0] = img2_pos_arr[k, 0] - img1_pos_arr[j, 0]  # x
        lock.acquire()
        try:
            xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'))
        except Exception as e:
            layer_num = len(layer_id_dict)
            xyz_shift_arr = np.zeros((layer_num, 3), dtype='int64')
        xyz_shift_arr[i, :] = xyz_shift_real
        np.save(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'), xyz_shift_arr)
        lock.release()
        run_print(lock, running_IO_path,
                  '################################################################################################\n',
                  '%.2d.th layer xyz_shift data saved, the final xyz_shift is %s\n' % (i, str(xyz_shift_real)),
                  '################################################################################################\n')


def start_layer_sec_stitch_sift(txt_path, save_txt_path, dir_path, data_IO_path, running_IO_path, img_name_format,
                                channel_num, ch_th, img_type, y_range, z_range, step=5, pyr_down_times=1):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)
    if_layer_stitch = Array(ctypes.c_bool, [False for i in range(layer_num)])
    if_layer_stitch[0] = True

    lock = RLock()
    process_num = 10
    run_print(lock, running_IO_path, 'Current processing quantity: %d' % process_num)
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=layer_sec_stitch_sift, args=(lock, txt_path, data_IO_path, running_IO_path,
                                                              layer_id_dict, if_layer_stitch, y_range,z_range, step,
                                                              pyr_down_times))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()

    xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'))
    xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr_20221103.npy'))
    strip_pos = update_next_layer_info(strip_pos, layer_id_dict, xyz_shift_arr)
    save_img_txt_info(save_txt_path, strip_pos, id_dir_dict)

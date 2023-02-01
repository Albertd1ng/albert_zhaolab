import os
import numpy as np
import time
import cv2
import ctypes
import random
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_img_txt_info, get_img_dim_info, save_img_txt_info, save_layer_id_dict_info, save_id_dir_dict_info
from ImgIO import import_img_3D_section
from StripCont import judge_strip_cont
from ImgBorder import get_two_strip_border, get_border_pyr_down
from ImgProcess import pyr_down_img, adjust_contrast
from ImgOvl import get_ovl_img
from LossFunc import loss_func_for_list
from RunningIO import run_print


def choose_stitched_index(if_strip_stitched, if_strip_shelved):
    for i, j in enumerate(zip(if_strip_stitched, if_strip_shelved)):
        if not any(j):
            return i
    return -1


def in_which_layer(i, layer_id_dict):
    for j in layer_id_dict.keys():
        if i in layer_id_dict[j]:
            return j


def choose_refer_index(i, layer_id_dict, if_strip_stitched, strip_cont_vector):
    layer_th = in_which_layer(i, layer_id_dict)
    index_list = []
    for index, j in enumerate(zip(if_strip_stitched, strip_cont_vector)):
        if all(j) and (index in layer_id_dict[layer_th]):
            index_list.append(index)
    min_dis_j = -1
    dis = np.inf
    for j in index_list:
        if abs(j-i) < dis:
            dis = abs(j-i)
            min_dis_j = j
    return min_dis_j


def calc_xyz_shift(img1_sec, img2_sec, img_pos, xy_v_num, z_depth, voxel_range, pyr_down_times):
    for pdt in range(pyr_down_times, -1, -1):
        if pdt == pyr_down_times:
            x_s, y_s, z_s = 0, 0, 0
            x_pd, y_pd, z_pd = np.int64(np.round(np.array(voxel_range, dtype='int64') / (2 ** pdt)))
        else:
            x_s, y_s, z_s = 2 * np.array((x_s, y_s, z_s), dtype='int64')
            x_pd, y_pd, z_pd = 3, 3, 3
            if voxel_range[2] == 0: z_pd = 0
        loss_min = np.inf
        if pdt != 0:
            img1_sec_pd = pyr_down_img(img1_sec, pdt)
            img2_sec_pd = pyr_down_img(img2_sec, pdt)
        else:
            img1_sec_pd = img1_sec.copy()
            img2_sec_pd = img2_sec.copy()
        down_multi = np.array([*img1_sec_pd.shape[1::-1], img1_sec_pd.shape[2]], dtype='float32') / np.array(
            [*xy_v_num, z_depth], dtype='float32')
        x_sr, y_sr, z_sr = x_s, y_s, z_s
        for x in range(x_sr - x_pd, x_sr + x_pd + 1):
            for y in range(y_sr - y_pd, y_sr + y_pd + 1):
                for z in range(z_sr - z_pd, z_sr + z_pd + 1):
                    this_img_pos = img_pos.copy()
                    this_img_pos[0, :] = this_img_pos[0, :] + np.array([x, y, z], dtype='int64') * (2 ** pdt)
                    border = get_border_pyr_down(get_two_strip_border(this_img_pos, xy_v_num, [z_depth, z_depth]),
                                                 down_multi)
                    if border.shape[0] == 0:
                        continue
                    # print(border)
                    ovl1_list, ovl2_list = get_ovl_img(img1_sec_pd, img2_sec_pd, border)
                    # for one_ovl in ovl1_list: print(one_ovl.shape)
                    # for one_ovl in ovl2_list: print(one_ovl.shape)
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list, [x, y, z])
                    if this_loss < loss_min:
                        loss_min = this_loss
                        x_s, y_s, z_s = x, y, z
        # print(pdt, x_s, y_s, z_s, loss_min)
    return x_s, y_s, z_s


def calc_mean_shift_without_outliers(xyz_shift_array):
    xyz_mean = np.mean(xyz_shift_array, axis=0)
    xyz_std = np.std(xyz_shift_array, axis=0)
    chosen_index = []
    for i in range(xyz_shift_array.shape[0]):
        if np.all(np.logical_or(xyz_shift_array[i, :] - xyz_mean < 2 * xyz_std, xyz_std == 0)):
            chosen_index.append(i)
    return np.int64(np.round(np.mean(xyz_shift_array[chosen_index, :], axis=0)))


def update_pos_after_shift(strip_pos, xyz_shift, strip_refer_id):
    def change(refer_one, changed_one):
        strip_pos_update[changed_one, :] = strip_pos_update[changed_one, :] + xyz_shift[refer_one, :]
        follow_changed = np.argwhere(strip_refer_id == changed_one)
        for [follow_changed_one] in follow_changed:
            change(refer_one, follow_changed_one)

    strip_pos_update = strip_pos.copy()
    zero_layer = np.argwhere(strip_refer_id == -1)
    other_layer = np.argwhere(strip_refer_id != -1)
    while other_layer.shape[0] != 0:
        for [zero_layer_one] in zero_layer:
            first_layer = np.argwhere(strip_refer_id == zero_layer_one)
            for [first_layer_one] in first_layer:
                change(first_layer_one, first_layer_one)
                strip_refer_id[first_layer_one] = -1
        zero_layer = np.argwhere(strip_refer_id == -1)
        other_layer = np.argwhere(strip_refer_id != -1)
    return strip_pos_update


def strip_stitch(lock, running_io_path, layer_id_dict, dir_path, img_name_format, id_dir_dict, strip_pos, strip_cont,
                 xy_v_num, z_v_num, ch_th, img_type, img_dtype, if_strip_stitched, if_strip_shelved, strip_refer_id,
                 xyz_shift, voxel_range, pyr_down_times, choose_num=10, z_depth=200):
    this_name = current_process().name
    strip_num = strip_pos.shape[0]
    while False in if_strip_stitched:
        lock.acquire()
        try:
            i = choose_stitched_index(if_strip_stitched, if_strip_shelved)
            if i == -1:
                for k in range(strip_num):
                    if_strip_shelved[k] = False
                lock.release()
                continue
            j = choose_refer_index(i, layer_id_dict, if_strip_stitched, strip_cont[i])
            if j == -1:
                if_strip_shelved[i] = True
                lock.release()
                run_print(lock, running_io_path, '%d.th strip shelved\n' % i)
                continue
            if_strip_stitched[i] = True
            strip_refer_id[i] = j
            lock.release()
        except Exception as e:
            lock.release()
            run_print(lock, running_io_path, '%s has an error\n' % this_name, str(e))
            break
        run_print(lock, running_io_path, '%s is stitching %d.th strip with %d.th strip.' % (this_name, i, j))

        img_path1, img_path2 = os.path.join(dir_path, id_dir_dict[i]), os.path.join(dir_path, id_dir_dict[j])
        border = get_two_strip_border(strip_pos[[i, j], :], xy_v_num, z_v_num[[i, j]])
        if border.shape[0] == 0:
            continue
        z_border_depth = border[0, 5] - border[0, 4]
        one_xyz_shift_array = np.zeros((choose_num, 3), dtype='int64')
        for cn in range(choose_num):
            z_start = random.randint(0, z_border_depth - z_depth)
            img1_sec = import_img_3D_section(img_name_format, img_path1,
                                             [border[0, 4] + z_start, border[0, 4] + z_start + z_depth],
                                             ch_th, xy_v_num, img_type=img_type, img_dtype=img_dtype)
            img2_sec = import_img_3D_section(img_name_format, img_path2,
                                             [border[1, 4] + z_start, border[1, 4] + z_start + z_depth],
                                             ch_th, xy_v_num, img_type=img_type, img_dtype=img_dtype)
            if ((mean1 := np.mean(img1_sec)) < 0.1) or ((mean2 := np.mean(img2_sec)) < 0.1):
                continue
            # print(img1_sec.shape, img2_sec.shape)
            # print('%s is processing two strips sections whose values are %f and %f.'%(this_name, mean1, mean2))
            img_pos = np.zeros((2, 3), dtype='int64')
            img_pos[:, 0:2] = strip_pos[[i, j], 0:2]
            img_pos[:, 2] = strip_pos[[i, j], 2] + border[:, 4] + z_start
            one_xyz_shift_array[cn, :] = calc_xyz_shift(img1_sec, img2_sec, img_pos, xy_v_num, z_depth, voxel_range,
                                                        pyr_down_times)
        xyz_shift_mean = calc_mean_shift_without_outliers(one_xyz_shift_array)
        xyz_shift[3 * i:3 * i + 3] = xyz_shift_mean
        run_print(lock, running_io_path,
                  '%s: current stitch is %d.th strip with %d.th strip,\nxyz_shift is (%d, %d, %d)'
                  % (this_name, i, j, *xyz_shift_mean))


def start_multi_strip_stitches(txt_path, save_txt_path, dir_path, running_io_path, img_name_format, channel_num,
                               ch_th, img_type, voxel_range=[50, 100, 0], pyr_down_times=2, choose_num=10, z_depth=200):
    # initialization of strip information.
    id_dir_dict, strip_pos, layer_id_dict = get_img_txt_info(txt_path)
    save_layer_id_dict_info(os.path.join(os.path.split(save_txt_path)[0],'layer_id_dict.txt'), layer_id_dict)
    save_id_dir_dict_info(os.path.join(os.path.split(save_txt_path)[0],'id_dir_dict.txt') ,id_dir_dict)
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    # strip_pos[:, :2] = np.int64(np.round(strip_pos[:, :2] / 9))
    # strip_pos[:, 2] = np.int64(np.round(strip_pos[:, 2] / 2))
    strip_cont = judge_strip_cont(strip_pos, xy_v_num)
    np.save(os.path.join(os.path.split(save_txt_path)[0], 'strip_cont.npy'), strip_cont)
    strip_num, layer_num = strip_pos.shape[0], len(layer_id_dict)
    # initialization of multiprocessing object.
    # if_strip_stitched = Array(ctypes.c_bool, [False for i in range(strip_num)])
    if_strip_stitched = Array(ctypes.c_bool, [True for i in range(strip_num)])
    for i in layer_id_dict.values():
        if_strip_stitched[i[0]] = True
    if_strip_shelved = Array(ctypes.c_bool, [False for i in range(strip_num)])
    xyz_shift = Array('i', [0 for i in range(3 * strip_num)])
    strip_refer_id = Array('l', [-1 for i in range(strip_num)])
    lock = RLock()

    # process_num = int(round(0.5 * cpu_count()))
    process_num = 10
    run_print(lock, running_io_path, 'Current processing quantity: %d' % process_num)
    # run_print(lock, running_io_path, str(strip_cont[:, 1]), str(xy_v_num), str(z_v_num))
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=strip_stitch, args=(
            lock, running_io_path, layer_id_dict, dir_path, img_name_format, id_dir_dict, strip_pos, strip_cont,
            xy_v_num, z_v_num, ch_th, img_type, img_dtype, if_strip_stitched, if_strip_shelved, strip_refer_id,
            xyz_shift, voxel_range, pyr_down_times, choose_num, z_depth))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    xyz_shift = np.array(xyz_shift, dtype='int64').reshape(-1, 3)
    strip_refer_id = np.array(strip_refer_id, dtype='int64')
    np.save(os.path.join(os.path.split(save_txt_path)[0], 'xyz_shift.npy'), xyz_shift)
    np.save(os.path.join(os.path.split(save_txt_path)[0], 'strip_refer_id.npy'), strip_refer_id)
    run_print(lock, running_io_path, 'start calculating new position!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    strip_pos_update = update_pos_after_shift(strip_pos, xyz_shift, strip_refer_id)
    run_print(lock, running_io_path, 'start saving data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    save_img_txt_info(save_txt_path, strip_pos_update, id_dir_dict)
    run_print(lock, running_io_path, 'end saving data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

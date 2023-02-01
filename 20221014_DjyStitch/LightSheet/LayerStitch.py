import os.path
import numpy as np
import random
import ctypes
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_img_dim_info, save_img_txt_info, get_layer_id_dict_info, get_strip_pos_info, get_id_dir_dict_info
from ImgIO import import_img_3D_section
from StripCont import judge_strip_cont
from ImgBorder import get_two_strip_border, get_border_pyr_down, get_z_border_depth
from ImgOvl import get_ovl_img
from StripStitch import calc_mean_shift_without_outliers, pyr_down_img
from LossFunc import loss_func_for_list
from RunningIO import run_print
from LayerStitchSIFT import update_next_layer_info


def choose_layer_index(if_layer_stitch):
    for i in range(len(if_layer_stitch)):
        if not if_layer_stitch[i]:
            return i
    return -1


def choose_refer_index_cross_layer(refer_id_list, strip_cont_list):
    one_refer_id_list = []
    for i in range(len(strip_cont_list)):
        if strip_cont_list[i] and i in refer_id_list:
            one_refer_id_list.append(i)
    return one_refer_id_list


def calc_xyz_shift_for_list(img1_sec_list, img2_sec_list, img1_pos, img2_pos, xy_v_num, z_depth, voxel_range,
                            pyr_down_times):
    refer_num = img1_pos.shape[0]
    for pdt in range(pyr_down_times, -1, -1):
        if pdt == pyr_down_times:
            x_s, y_s, z_s = 0, 0, 0
            x_pd, y_pd, z_pd = np.int64(np.round(np.array(voxel_range, dtype='int64') / (2 ** pdt)))
        else:
            x_s, y_s, z_s = 2 * np.array((x_s, y_s, z_s), dtype='int64')
            x_pd, y_pd, z_pd = 3, 3, 3
        loss_min = np.inf
        if pdt != 0:
            img1_sec_list_pd = [pyr_down_img(one_img, pdt) for one_img in img1_sec_list]
            img2_sec_list_pd = [pyr_down_img(one_img, pdt) for one_img in img2_sec_list]
        else:
            img1_sec_list_pd = [one_img.copy() for one_img in img1_sec_list]
            img2_sec_list_pd = [one_img.copy() for one_img in img2_sec_list]
        down_multi_list = [
            np.array([*img1_sec_list_pd[i].shape[1::-1], img1_sec_list_pd[i].shape[2]], dtype='float32') / np.array(
                [*xy_v_num, z_depth], dtype='float32') for i in range(refer_num)]
        x_sr, y_sr, z_sr = x_s, y_s, z_s
        for x in range(x_sr - x_pd, x_sr + x_pd + 1):
            for y in range(y_sr - y_pd, y_sr + y_pd + 1):
                for z in range(z_sr - z_pd, z_sr + z_pd + 1):
                    this_img1_pos, this_img2_pos = img1_pos.copy(), img2_pos.copy()
                    this_img1_pos = this_img1_pos + np.array([x, y, z], dtype='int64') * (2**pdt)
                    border_list = [get_border_pyr_down(get_two_strip_border(np.vstack((this_img1_pos[i, :],
                                   this_img2_pos[i, :])), xy_v_num, [z_depth, z_depth]),down_multi_list[i])
                                   for i in range(refer_num)]
                    real_ovl_id_list = np.zeros(refer_num, dtype='bool')
                    for i in range(refer_num):
                        if border_list[i].shape[0] != 0:
                            real_ovl_id_list[i] = True
                    ovl1_list, ovl2_list = [], []
                    for i in range(refer_num):
                        if not real_ovl_id_list[i]:
                            continue
                        this_ovl1_list, this_ovl2_list = get_ovl_img(img1_sec_list_pd[i], img2_sec_list_pd[i], border_list[i])
                        for j,k in zip(this_ovl1_list, this_ovl2_list):
                            if j.shape == k.shape:
                                ovl1_list.append(j)
                                ovl2_list.append(k)
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list, [x, y, z])
                    if this_loss < loss_min:
                        loss_min = this_loss
                        x_s, y_s, z_s = x, y, z
        # print(pdt,'[', x_s, y_s, z_s, ']', loss_min)
    return x_s, y_s, z_s


def update_strip_pos_for_layer(strip_pos, xyz_shift_array, layer_id_dict):
    layer_num = len(layer_id_dict)
    strip_pos_update = strip_pos.copy()
    for i in range(1, layer_num):
        for j in range(layer_id_dict[i]):
            strip_pos_update[j, :] = strip_pos_update[j, :] + xyz_shift_array[i, :]
    return strip_pos_update


def get_multi_strip_border_list(dir_path, strip_pos, id_dir_dict, one_stitched_id, one_refer_id_list, xy_v_num, z_v_num):
    img_path1 = os.path.join(dir_path, id_dir_dict[one_stitched_id])
    img_path2_list = [os.path.join(dir_path, id_dir_dict[one_id]) for one_id in one_refer_id_list]
    border_list = [get_two_strip_border(strip_pos[[one_stitched_id, one_id], :], xy_v_num,
                                        z_v_num[[one_stitched_id, one_id]]) for one_id in one_refer_id_list]
    for i in range(len(border_list)-1, -1, -1):
        if border_list[i].shape[0] == 0:
            border_list.pop(i)
            img_path2_list.pop(i)
    return img_path1, img_path2_list, border_list


def get_multi_img_sec_list(img_name_format, img_path1, img_path2_list, border_list, z_start, z_depth, ch_th,
                           xy_v_num, img_type, img_dtype):
    refer_num = len(img_path2_list)
    img1_sec_list = [import_img_3D_section(img_name_format, img_path1,
                                           [border_list[j][0, 4] + z_start,
                                            border_list[j][0, 4] + z_start + z_depth],
                                           ch_th, xy_v_num, img_type=img_type, img_dtype=img_dtype)
                     for j in range(refer_num)]
    img2_sec_list = [import_img_3D_section(img_name_format, img_path2_list[j],
                                           [border_list[j][1, 4] + z_start,
                                            border_list[j][1, 4] + z_start + z_depth],
                                           ch_th, xy_v_num, img_type=img_type, img_dtype=img_dtype)
                     for j in range(refer_num)]
    for i in range(refer_num-1, 0, refer_num):
        if img1_sec_list[i].shape[0] == 0 or img2_sec_list[i].shape[0] == 0:
            img1_sec_list.pop(i)
            img2_sec_list.pop(i)
    return img1_sec_list, img2_sec_list


def layer_stitch(lock, txt_path, dir_path, running_IO_path, strip_pos, strip_cont, layer_id_dict, id_dir_dict,
                 img_name_format, ch_th, img_type, img_dtype, if_layer_stitch, xy_v_num, z_v_num,
                 voxel_range, pyr_down_times, choose_num, z_depth):
    this_name = current_process().name
    strip_num, layer_num = strip_pos.shape[0], len(layer_id_dict)
    while False in if_layer_stitch:
        lock.acquire()
        i = choose_layer_index(if_layer_stitch)
        if i == -1:
            lock.release()
            break
        if_layer_stitch[i] = True
        lock.release()
        run_print(lock, running_IO_path, '%s: start stitching the %d.th layer with %d.th layer' % (this_name, i, i-1))
        one_xyz_shift_array = np.zeros((choose_num, 3), dtype='int64')
        refer_id_list, stitched_id_list = layer_id_dict[i - 1], layer_id_dict[i]
        count = 0
        exit_count = 0
        while count < choose_num and exit_count < 1000:
            one_stitched_id = stitched_id_list[random.randint(0, len(stitched_id_list)-1)]
            one_refer_id_list = choose_refer_index_cross_layer(refer_id_list, strip_cont[one_stitched_id, :])
            if len(one_refer_id_list) == 0:
                exit_count = exit_count + 1
                continue
            # run_print('%s: Now choosing %d.th strip with %s.th strip for %d.th sample' % (this_name, one_stitched_id, one_refer_id_list, count))
            img_path1, img_path2_list, border_list = get_multi_strip_border_list(dir_path, strip_pos, id_dir_dict, one_stitched_id, one_refer_id_list, xy_v_num, z_v_num)
            if len(img_path2_list) == 0:
                exit_count = exit_count + 1
                continue
            z_border_depth = get_z_border_depth(border_list)
            if z_border_depth < z_depth:
                exit_count = exit_count + 1
                continue
            z_start = random.randint(0, z_border_depth - z_depth)
            img1_sec_list, img2_sec_list = get_multi_img_sec_list(img_name_format, img_path1, img_path2_list,
                                                                  border_list, z_start, z_depth, ch_th, xy_v_num,
                                                                  img_type, img_dtype)
            refer_num = len(img1_sec_list)
            if refer_num == 0:
                exit_count = exit_count + 1
                continue
            img1_pos = np.zeros((refer_num, 3), dtype='int64')
            img2_pos = np.zeros((refer_num, 3), dtype='int64')
            for j in range(refer_num):
                img1_pos[j, 0:2] = strip_pos[one_stitched_id, 0:2]
                img1_pos[j, 2] = strip_pos[one_stitched_id, 2] + border_list[j][0, 4] + z_start
                img2_pos[j, 0:2] = strip_pos[one_refer_id_list[j], 0:2]
                img2_pos[j, 2] = strip_pos[one_refer_id_list[j], 2] + border_list[j][1, 4] + z_start
            one_xyz_shift_array[count, :] = calc_xyz_shift_for_list(img1_sec_list, img2_sec_list, img1_pos, img2_pos,
                                                                    xy_v_num, z_depth, voxel_range, pyr_down_times)
            run_print(lock, running_IO_path, str(one_xyz_shift_array[count, :]))
            count += 1
            exit_count = 0

        xyz_shift_mean = calc_mean_shift_without_outliers(one_xyz_shift_array)
        run_print(lock, running_IO_path,
                  '################################################################################################\n'
                  '%s: end stitching the %d.th layer with %d.th layer, shift is %s\n' % (this_name, i, i-1, str(xyz_shift_mean)),
                  'shift array is %s' % str(one_xyz_shift_array),
                  '################################################################################################\n'
                  )
        lock.acquire()
        try:
            xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'))
        except Exception as e:
            layer_num = len(layer_id_dict)
            xyz_shift_arr = np.zeros((layer_num, 3), dtype='int64')
        xyz_shift_arr[i, :] = xyz_shift_mean
        np.save(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'), xyz_shift_arr)
        lock.release()


def start_multi_layer_stitch(txt_path, save_txt_path, dir_path, running_IO_path, img_name_format, channel_num, ch_th,
                             img_type='tif', voxel_range=[50, 50, 50], pyr_down_times=2, choose_num=20, z_depth=200):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    strip_cont = judge_strip_cont(strip_pos, xy_v_num)
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)
    if_layer_stitch = Array(ctypes.c_bool, [False for i in range(layer_num)])
    if_layer_stitch[0] = True

    lock = RLock()
    process_num = 10
    run_print(lock, running_IO_path, 'Current processing quantity: %d' % process_num)
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=layer_stitch, args=(lock, txt_path, dir_path, running_IO_path, strip_pos, strip_cont,
                        layer_id_dict, id_dir_dict, img_name_format, ch_th, img_type, img_dtype, if_layer_stitch,
                        xy_v_num, z_v_num, voxel_range, pyr_down_times, choose_num, z_depth))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()

    xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'))
    strip_pos = update_next_layer_info(strip_pos, layer_id_dict, xyz_shift_arr)
    save_img_txt_info(save_txt_path, strip_pos, id_dir_dict)

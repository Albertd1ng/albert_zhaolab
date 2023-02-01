import os.path

import numpy as np
import random

from InfoIO import get_img_txt_info, get_img_dim_info, save_img_txt_info, get_layer_id_dict_info
from ImgIO import import_img_2D_one, import_img_3D_section
from StripCont import judge_strip_cont
from ImgBorder import get_two_strip_border, get_border_pyr_down, get_z_border_depth
from ImgOvl import get_ovl_img
from StripStitch import calc_mean_shift_without_outliers, pyr_down_img
from LossFunc import loss_func_for_list


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
                    ovl1_list, ovl2_list = [], []
                    for i in range(refer_num):
                        this_ovl1_list, this_ovl2_list = get_ovl_img(img1_sec_list_pd[i], img2_sec_list_pd[i], border_list[i])
                        for j,k in zip(this_ovl1_list, this_ovl2_list):
                            ovl1_list.append(j)
                            ovl2_list.append(k)
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list, [x, y, z])
                    if this_loss < loss_min:
                        loss_min = this_loss
                        x_s, y_s, z_s = x, y, z
        print(pdt,'[', x_s, y_s, z_s, ']', loss_min)
    return x_s, y_s, z_s


def update_strip_pos_for_layer(strip_pos, xyz_shift_array, layer_id_dict):
    layer_num = len(layer_id_dict)
    strip_pos_update = strip_pos.copy()
    for i in range(1, layer_num):
        for j in range(layer_id_dict[i]):
            strip_pos_update[j, :] = strip_pos_update[j, :] + xyz_shift_array[i, :]
    return strip_pos_update


def layer_stitch(txt_path, save_txt_path, dir_path, img_name_format, channel_num, ch_th, img_type,
                 voxel_range=[50, 100, 100], pyr_down_times=2, choose_num=20, z_depth=400):
    id_dir_dict, strip_pos, _ = get_img_txt_info(txt_path)
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    strip_cont = judge_strip_cont(strip_pos, xy_v_num)
    strip_num, layer_num = strip_pos.shape[0], len(layer_id_dict)
    xyz_shift_array = np.zeros((layer_num, 3), dtype='int64')
    for i in range(1, layer_num):
        print('start stitching the %d.th layer with %d.th layer' % (i, i-1))
        one_xyz_shift_array = np.zeros((choose_num, 3), dtype='int64')
        refer_id_list, stitched_id_list = layer_id_dict[i - 1], layer_id_dict[i]
        count = 0
        while count < choose_num:
            one_stitched_id = stitched_id_list[random.randint(0, len(stitched_id_list))]
            one_refer_id_list = choose_refer_index_cross_layer(refer_id_list, strip_cont[one_stitched_id, :])
            if (refer_num := len(one_refer_id_list)) == 0:
                continue
            print('Now choosing %d.th strip with %s.th strip for %d.th sample' %
                  (one_stitched_id, one_refer_id_list, count))
            img_path1 = os.path.join(dir_path, id_dir_dict[one_stitched_id])
            img_path2_list = [os.path.join(dir_path, id_dir_dict[one_id]) for one_id in one_refer_id_list]
            border_list = [get_two_strip_border(strip_pos[[one_stitched_id, one_id], :], xy_v_num,
                                                z_v_num[[one_stitched_id, one_id]]) for one_id in one_refer_id_list]
            z_border_depth = get_z_border_depth(border_list)
            z_start = random.randint(0, z_border_depth - z_depth)
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
            img1_pos = np.zeros((refer_num, 3), dtype='int64')
            img2_pos = np.zeros((refer_num, 3), dtype='int64')
            for j in range(refer_num):
                img1_pos[j, 0:2] = strip_pos[one_stitched_id, 0:2]
                img1_pos[j, 2] = strip_pos[one_stitched_id, 2] + border_list[j][0, 4] + z_start
                img2_pos[j, 0:2] = strip_pos[one_refer_id_list[j], 0:2]
                img2_pos[j, 2] = strip_pos[one_refer_id_list[j], 2] + border_list[j][1, 4] + z_start
            one_xyz_shift_array[count, :] = calc_xyz_shift_for_list(img1_sec_list, img2_sec_list, img1_pos, img2_pos,
                                                                    xy_v_num, z_depth, voxel_range, pyr_down_times)
            count += 1
        xyz_shift_mean = calc_mean_shift_without_outliers(one_xyz_shift_array)
        xyz_shift_array[i, :] = xyz_shift_mean
    strip_pos_update = update_strip_pos_for_layer(strip_pos, xyz_shift_array, layer_id_dict)
    save_img_txt_info(save_txt_path, strip_pos_update, id_dir_dict)

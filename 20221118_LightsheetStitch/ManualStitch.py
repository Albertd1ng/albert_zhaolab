import numpy as np
import os
import cv2

from InfoIO import get_strip_pos_info, get_id_dir_dict_info, get_layer_id_dict_info, save_img_txt_info, get_img_dim_info
from ImgIO import import_img_whole_2D_XY


def update_next_layer_info(strip_pos, layer_id_dict, xyz_shift_arr):
    layer_num = len(layer_id_dict)
    for i in range(layer_num):
        id_list = layer_id_dict[i]
        strip_pos[id_list, :] = strip_pos[id_list, :] + xyz_shift_arr[i, :]
        for j in range(i + 1, layer_num):
            next_id_list = layer_id_dict[j]
            strip_pos[next_id_list, :] = strip_pos[next_id_list, :] + xyz_shift_arr[i, :]
    return strip_pos


def layer_manual_stitch(l_th, txt_path, save_txt_path, xyz_shift_vec):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    layer_num = len(layer_id_dict)
    xyz_shift_arr = np.zeros((layer_num, 3), dtype='int64')
    xyz_shift_arr[l_th, :] = np.array(xyz_shift_vec, dtype='int64')
    strip_pos = update_next_layer_info(strip_pos, layer_id_dict, xyz_shift_arr)
    save_img_txt_info(save_txt_path, strip_pos, id_dir_dict)


def strip_manual_stitch(s_th, txt_path, save_txt_path, xyz_shift_vec, if_change_followed=False):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    strip_num = strip_pos.shape[0]
    if if_change_followed:
        strip_pos[s_th:, :] = strip_pos[s_th:, :] + np.array(xyz_shift_vec, dtype='int64')
    else:
        strip_pos[s_th, :] = strip_pos[s_th, :] + np.array(xyz_shift_vec, dtype='int64')
    save_img_txt_info(save_txt_path, strip_pos, id_dir_dict)


def man_stitch_and_show(s_th, txt_path, save_txt_path, dir_path, img_save_path, xyz_shift_vec, img_name_format, z_pos, channel_num, ch_th,
                        change_type='layer', if_change_followed=False, img_type='tif', if_bright_edge=False):
    if change_type == 'layer':
        layer_manual_stitch(s_th, txt_path, save_txt_path, xyz_shift_vec)
    elif change_type == 'strip':
        strip_manual_stitch(s_th, txt_path, save_txt_path, xyz_shift_vec, if_change_followed=if_change_followed)
    strip_pos = get_strip_pos_info(save_txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format, img_type=img_type)
    this_img = import_img_whole_2D_XY(img_name_format, dir_path, id_dir_dict, z_pos, ch_th, strip_pos, xy_v_num,
                                      z_v_num, img_type, img_dtype, if_bright_edge=if_bright_edge)
    img_name = 'z%.6d_ch%.2d.%s' % (z_pos, ch_th, img_type)
    cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    print(img_name, 'saved')


if __name__ == '__main__':
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_yz_stitch_20221101.txt'
    save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_yz_stitch_20221101.txt'
    dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/src_frames_ds331'
    img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221101outputwhole'
    img_name_format = r'%s_%s'
    channel_num = 2
    ch_th = 0
    img_type='tif'

    z_pos = 8000
    if_bright_edge = False

    change_type = 'strip'
    if_chang_followed = False
    s_th = []#165-1773
    xyz_shift_vec = [0, 0, 0]

    man_stitch_and_show(s_th, txt_path, save_txt_path, dir_path, img_save_path, xyz_shift_vec, img_name_format, z_pos,
                        channel_num, ch_th, change_type=change_type, if_change_followed=if_chang_followed,
                        img_type=img_type, if_bright_edge=if_bright_edge)
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_yz_stitch_20221101.txt'


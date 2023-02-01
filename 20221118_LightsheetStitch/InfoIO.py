import os
import re
import numpy as np
import pandas as pd

from ImgIO import import_img_2D_one


def get_xyz_from_str(xyz_str):
    new_xyz_str = re.sub('(\(|\)| )', '', xyz_str)
    # print(new_xyz_str)
    xyz_str_list = new_xyz_str.split(',')
    x, y, z = int(round(float(xyz_str_list[0]))), int(round(float(xyz_str_list[1]))), int(round(float(xyz_str_list[2])))
    return [x, y, z]


def get_img_txt_info(txt_path):
    dir_list = []
    x_pos_list, y_pos_list, z_pos_list = [], [], []
    with open(txt_path, 'r') as f:
        while one_line := f.readline():
            one_line_list = one_line.split(';')
            if len(one_line_list) != 3:
                continue
            xyz_list = get_xyz_from_str(one_line_list[2])
            dir_list.append(one_line_list[0])
            x_pos_list.append(xyz_list[0]), y_pos_list.append(xyz_list[1]), z_pos_list.append(xyz_list[2])
    dir_pos_df = pd.DataFrame({'DirName': dir_list, 'X': x_pos_list, 'Y': y_pos_list, 'Z': z_pos_list})
    # print(dir_pos_df)
    dir_pos_df.sort_values(by=['X', 'Y'], ascending=True, inplace=True)
    # print(dir_pos_df)
    strip_num = len(dir_list)

    id_dir_dict = {}
    layer_id_dict = {}
    strip_pos = np.zeros((strip_num, 3), dtype='int64')

    this_layer_id = []
    this_x_pos = np.inf
    this_layer_num = -1
    for i in range(strip_num):
        id_dir_dict[i] = dir_pos_df.iloc[i, 0]
        strip_pos[i, :] = dir_pos_df.iloc[i, 1:]
        if strip_pos[i, 0] == this_x_pos:
            this_layer_id.append(i)
        else:
            if this_layer_num != -1:
                layer_id_dict[this_layer_num] = this_layer_id
            this_x_pos = strip_pos[i, 0]
            this_layer_id = [i]
            this_layer_num += 1
    layer_id_dict[this_layer_num] = this_layer_id

    return id_dir_dict, strip_pos, layer_id_dict


def get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format, img_type='tif'):
    strip_num = len(id_dir_dict)
    xy_v_num = np.zeros(2, dtype='int64')
    z_v_num = np.zeros(strip_num, dtype='int64')
    for i in range(strip_num):
        img_path = os.path.join(dir_path, id_dir_dict[i])
        z_v_num[i] = np.int64(round(len(os.listdir(img_path)) / channel_num))
        if i == 0:
            one_img = import_img_2D_one(img_name_format, img_path, 0, 0, img_type=img_type)
            xy_v_num[0], xy_v_num[1] = one_img.shape[1], one_img.shape[0]
            img_dtype = one_img.dtype
    return xy_v_num, z_v_num, img_dtype


def get_strip_pos_info(txt_path):
    x_pos_list, y_pos_list, z_pos_list = [], [], []
    with open(txt_path, 'r') as f:
        while one_line := f.readline():
            one_line_list = one_line.split(';')
            if len(one_line_list) != 3:
                continue
            xyz_list = get_xyz_from_str(one_line_list[2])
            x_pos_list.append(xyz_list[0]), y_pos_list.append(xyz_list[1]), z_pos_list.append(xyz_list[2])
    strip_num = len(x_pos_list)
    strip_pos = np.zeros((strip_num, 3), dtype='int64')
    for i in range(strip_num):
        strip_pos[i, :] = [x_pos_list[i], y_pos_list[i], z_pos_list[i]]
    return strip_pos


def get_id_dir_dict_info(txt_path):
    id_dir_dict = {}
    with open(txt_path, 'r') as f:
        while one_line := f.readline():
            one_line_list = one_line.split(';')
            if len(one_line_list) == 2:
                # print(int(one_line_list[0]), one_line_list[1])
                id_dir_dict[int(one_line_list[0])] = one_line_list[1].replace('\n','')
    return id_dir_dict


def get_layer_id_dict_info(txt_path):
    layer_id_dict = {}
    with open(txt_path, 'r') as f:
        while one_line := f.readline():
            one_line_list = one_line.split(';')
            if len(one_line_list) == 2:
                # print(int(one_line_list[0]), one_line_list[1])
                layer_id_dict[int(one_line_list[0])] = eval(one_line_list[1])
    return layer_id_dict


def save_img_txt_info(txt_path, strip_pos, id_dir_dict):
    with open(txt_path, 'w') as f:
        for i in range(strip_pos.shape[0]):
            f.write('%s; ; (%d, %d, %d)\n' % (id_dir_dict[i], strip_pos[i, 0], strip_pos[i, 1], strip_pos[i, 2]))


def save_layer_id_dict_info(save_path, layer_id_dict):
    with open(save_path, 'w') as f:
        for i in layer_id_dict.keys():
            f.write('%d;%s\n' % (i, layer_id_dict[i]))


def save_id_dir_dict_info(save_path, id_dir_dict):
    with open(save_path, 'w') as f:
        for i in id_dir_dict.keys():
            f.write('%d;%s\n' % (i, id_dir_dict[i]))


def trans_pos_info(txt_path, save_txt_path, xyz_times):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    for i in range(3):
        strip_pos[:, i] = strip_pos[:, i] * xyz_times[i]
    save_img_txt_info(save_txt_path, strip_pos, id_dir_dict)
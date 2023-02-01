import cv2
import numpy as np
import os
import nd2
from readlif.reader import LifFile

from AxisRange import calc_max_axis_range_vert_merged


def get_img_from_nd2(whole_img, i, ch_num, ch_th):
    if ch_num == 1:
        old_img = whole_img[i, :, :, :].copy()
    else:
        old_img = whole_img[i, ch_th, :, :, :].copy()
    new_img = np.zeros((old_img.shape[1], old_img.shape[2], old_img.shape[0]), dtype=old_img.dtype)
    for j in range(old_img.shape[0]):
        new_img[:, :, j] = old_img[j, :, :]
    return new_img


def get_img_2D_from_nd2(whole_img, i, ch_num, ch_th, z_th):
    if ch_num == 1:
        img = whole_img[i, z_th, :, :].copy()
    else:
        img = whole_img[i, ch_th, z_th, :, :].copy()
    return img


def get_img_from_lif(whole_img, i, ch_th, dim_elem_num):
    if whole_img.info['settings']['BitSize'] == '8':
        img_dtype = 'uint8'
    elif whole_img.info['settings']['BitSize'] == '16':
        img_dtype = 'uint16'
    img = np.zeros((dim_elem_num[1], dim_elem_num[0], dim_elem_num[2]), dtype=img_dtype)
    for j in range(dim_elem_num[2]):
        img[:, :, j] = np.array(whole_img.get_frame(z=j, t=ch_th, m=i), dtype=img_dtype)
    return img


def get_img_2D_from_lif(whole_img, i, ch_th, z_th, dim_elem_num):
    if whole_img.info['settings']['BitSize'] == '8':
        img_dtype = 'uint8'
    elif whole_img.info['settings']['BitSize'] == '16':
        img_dtype = 'uint16'
    img = np.array(whole_img.get_frame(z=z_th, t=ch_th, m=i), dtype=img_dtype)
    return img


def import_img_one_tile(img_name_format, img_path, img_name, i, ch_th, dim_elem_num,
                        img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    voxel_array = np.zeros(tuple(dim_elem_num), dtype=img_data_type)
    for j in range(dim_elem_num[2]):
        one_img_name = os.path.join(img_path, img_name_format % (img_name, i, j, ch_th, img_type))
        voxel_array[:, :, i] = cv2.imread(one_img_name, img_mode)
    return voxel_array


def import_img_2D(img_name_format, img_path, img_name, z_th, channel_ordinal,
                  img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    one_img_name = os.path.join(img_path, img_name_format % (img_name, z_th, channel_ordinal, img_type))
    one_img = cv2.imread(one_img_name, img_mode)
    return one_img


def import_img_2D_tile(img_name_format, img_path, img_name, ordinal, z_th, ch_th,
                       img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    one_img_name = os.path.join(img_path, img_name_format % (img_path, img_name, ordinal, z_th, ch_th, img_type))
    return cv2.imread(one_img_name, img_mode)


def export_img_hori_stit(layer_num, info_IO_path, img_path, img_save_path, img_name_format, img_name, ch_num,
                         img_type, img_data_type, img_save_type='tif'):
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    dim_elem_num = np.load(os.path.join(info_IO_path, 'dim_elem_num_%.4d.npy' % layer_num))
    dim_len = np.load(os.path.join(info_IO_path, 'dim_len_%.4d.npy' % layer_num))
    voxel_len = np.load(os.path.join(info_IO_path, 'voxel_len.npy'))
    tile_pos = np.load(os.path.join(info_IO_path, 'tile_pos_stitch_%.4d.npy' % layer_num))
    axis_range = np.load(os.path.join(info_IO_path, 'axis_range_stitch_%.4d.npy' % layer_num))
    first_last_index = np.load(os.path.join(info_IO_path, 'first_last_index_%.4d.npy' % layer_num))
    voxel_num = np.int64(np.round((axis_range[:, 1] - axis_range[:, 0]) / voxel_len))
    tile_num = tile_pos.shape[0]
    if img_type == 'nd2':
        whole_img = nd2.imread(img_path)
    elif img_type == 'lif':
        whole_img = LifFile(img_path)
        whole_img = whole_img.get_image(0)
    img_num = 0
    for j in range(first_last_index[0], first_last_index[1]):
        for ch_th in range(ch_num):
            this_img = np.zeros((voxel_num[1::-1]), dtype=img_data_type)
            this_z = axis_range[2, 0] + voxel_len[2] * j
            for k in range(tile_num):
                if tile_pos[k, 2] + dim_len[2] < this_z or tile_pos[k, 2] > this_z:
                    continue
                z_th = np.int64(np.round((this_z - tile_pos[k, 2]) / voxel_len[2]))
                if z_th >= dim_elem_num[2] or z_th < 0:
                    continue
                x_th = np.int64(np.round((tile_pos[k, 0] - axis_range[0, 0]) / voxel_len[0]))
                y_th = np.int64(np.round((tile_pos[k, 1] - axis_range[1, 0]) / voxel_len[1]))
                if x_th < 0 or x_th + dim_elem_num[0] > voxel_num[0] or y_th < 0 or y_th + dim_elem_num[1] > \
                        voxel_num[1]:
                    continue
                if img_type == 'nd2':
                    img_2D = get_img_2D_from_nd2(whole_img, k, ch_num, ch_th, z_th)
                elif img_type == 'lif':
                    img_2D = get_img_2D_from_lif(whole_img, k, ch_th, z_th, dim_elem_num)
                elif img_type == 'tif':
                    img_2D = import_img_2D_tile(img_name_format, img_path, img_name, k, z_th, ch_th,
                                                img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
                this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
            cv2.imwrite(os.path.join(img_save_path, '%s_z%.4d_ch%.2d.%s' % (img_name, img_num, ch_th, img_save_type)),
                        this_img)
            img_num += 1


def export_img_vert_stit_merged(layer_num, info_IO_path, file_path, file_name_format, img_save_path, img_name_format,
                                img_name, channel_num, img_type, img_data_type, img_num=0):
    if img_data_type == 'uint8':
        img_mode = cv2.IMREAD_GRAYSCALE
    elif img_data_type == 'uint16':
        img_mode = cv2.IMREAD_UNCHANGED
    xy_axis_range, xy_voxel_num = calc_max_axis_range_vert_merged(layer_num,info_IO_path)
    for i in range(layer_num):
        img_path = os.path.join(file_path, file_name_format % (i))
        dim_elem_num = np.load(os.path.join(info_IO_path, 'dim_elem_num_zstitch_%.4d.npy' % (i)))
        axis_range = np.load(os.path.join(info_IO_path, 'axis_range_zstitch_%.4d.npy' % (i)))
        first_last_index = np.load(os.path.join(info_IO_path, 'first_last_index_zstitch_%.4d.npy' % (i)))
        for j in range(first_last_index[0], first_last_index[1]):
            for c in range(channel_num):
                this_img = np.zeros(xy_voxel_num[1::-1], dtype=img_data_type)
                x_th, y_th = axis_range[0, 0] - xy_axis_range[0, 0], axis_range[1, 0] - xy_axis_range[1, 0]
                img_2D = import_img_2D(img_name_format, img_path, img_name, j, c, img_type=img_type,
                                       img_data_type=img_data_type, img_mode=img_mode)
                this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
                os.path.join(img_save_path, img_name_format % (img_name, img_num, c, img_type), this_img)
            img_num += 1

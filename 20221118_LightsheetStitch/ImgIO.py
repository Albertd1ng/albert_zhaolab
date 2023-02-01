import os
import cv2
import numpy as np
import math
from merge import MergeSolution


def import_img_2D_one(img_name_format, img_path, z_th, ch_th, img_type='tif'):
    this_img_name = os.path.join(img_path, ''.join((img_name_format % (z_th, ch_th), '.', img_type)))
    return cv2.imread(this_img_name, cv2.IMREAD_UNCHANGED)


def import_img_3D_section(img_name_format, img_path, z_range, ch_th, xy_v_num, img_type='tif', img_dtype='uint16'):
    strip_section = np.zeros((xy_v_num[1], xy_v_num[0], z_range[1]-z_range[0]), dtype=img_dtype)
    for z in range(z_range[0],z_range[1]):
        if z < 0:
            continue
        this_img_name = os.path.join(img_path, ''.join((img_name_format % (z, ch_th), '.', img_type)))
        this_img = cv2.imread(this_img_name, cv2.IMREAD_UNCHANGED)
        if this_img is None:
            continue
        strip_section[:,:,z-z_range[0]] = this_img
    return strip_section


def import_img_3D_sec_by_z(img_name_format, img_path, z_pos, z_start, z_depth, ch_th, xy_v_num, img_type='tif', img_dtype='uint16'):
    strip_section = np.zeros((xy_v_num[1], xy_v_num[0], z_depth), dtype=img_dtype)
    for z in range(z_depth):
        z_th = z_start - z_pos + z
        this_img_name = os.path.join(img_path, ''.join((img_name_format % (z, ch_th), '.', img_type)))
        this_img = cv2.imread(this_img_name, cv2.IMREAD_UNCHANGED)
        if this_img is None:
            continue
        strip_section[:, :, z] = this_img
    return strip_section


def import_img_slab_2D_max_YZ(img_name_format, dir_path, id_dir_dict, id_list, ch_th, strip_pos, xy_v_num, z_v_num,
                              direct='lr', sam_times=10, x_step=30, img_type='tif', img_dtype='uint16'):
    strip_num = len(id_list)
    x_min, x_max = np.min(strip_pos[id_list, 0]), np.max(strip_pos[id_list, 0]) + xy_v_num[0]
    y_min, y_max = np.min(strip_pos[id_list, 1]), np.max(strip_pos[id_list, 1]) + xy_v_num[1]
    z_min, z_max = np.min(strip_pos[id_list, 2]), np.max(strip_pos[id_list ,2] + z_v_num[id_list])
    img_YZ_3D_array = np.zeros((y_max-y_min, z_max-z_min, sam_times), dtype=img_dtype)
    img_pos_array = np.zeros((sam_times, 3), dtype='int64')
    img_pos_array[:, 1] = y_min
    img_pos_array[:, 2] = z_min
    for i in range(sam_times):
        if direct == 'lr':
            img_pos_array[i, 0] = x_min + i * x_step
        elif direct == 'rl':
            img_pos_array[i, 0] = x_max - (i+1) * x_step
    for i in range(strip_num):
        this_id = id_list[i]
        this_x, this_y = strip_pos[this_id, 0], strip_pos[this_id, 1]
        y_th, y_th_m = this_y - y_min, this_y - y_min + xy_v_num[1]
        for j in range(z_v_num[this_id]):
            this_z = strip_pos[this_id, 2] + j
            z_th = this_z - z_min
            this_img = import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[this_id]), j, ch_th, img_type=img_type)
            for k in range(sam_times):
                x_th, x_th_m = img_pos_array[k, 0] - this_x, img_pos_array[k, 0] + x_step - this_x
                if x_th < 0:
                    x_th = 0
                if x_th_m > xy_v_num[0]:
                    x_th_m = xy_v_num[0]
                if x_th >= xy_v_num[0]:
                    continue
                if x_th_m <= 0:
                    continue
                img_YZ_3D_array[y_th:y_th_m, z_th, k] = np.max(np.vstack((np.max(this_img[:, x_th:x_th_m], axis=1), img_YZ_3D_array[y_th:y_th_m, z_th, k])), axis=0)
    return img_YZ_3D_array, img_pos_array


def import_slab_2D_max_XZ_YZ(img_name_format, dir_path, id_dir_dict, id_list, ch_num, strip_pos, xy_v_num, z_v_num,
                                img_type='tif', img_dtype='uint16'):
    strip_num = len(id_list)
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0]
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1]
    z_min, z_max = np.min(strip_pos[:, 2]), np.max(strip_pos[:, 2] + z_v_num)
    img_XZ_3D_arr = np.zeros((x_max - x_min, z_max - z_min, 3), dtype=img_dtype)
    img_YZ_3D_arr = np.zeros((y_max - y_min, z_max - z_min, 3), dtype=img_dtype)
    for i in range(strip_num):
        this_id = id_list[i]
        this_x, this_y = strip_pos[this_id, 0], strip_pos[this_id, 1]
        x_th, x_th_m = this_x - x_min, this_x - x_min + xy_v_num[0]
        y_th, y_th_m = this_y - y_min, this_y - y_min + xy_v_num[1]
        for j in range(z_v_num[this_id]):
            this_z = strip_pos[this_id, 2] + j
            z_th = this_z - z_min
            for k in range(ch_num):
                this_img = import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[this_id]), j, k, img_type=img_type)
                img_XZ_3D_arr[x_th:x_th_m, z_th, k] = np.max(np.vstack((np.max(this_img, axis=0), img_XZ_3D_arr[x_th:x_th_m, z_th, k])), axis=0)
                img_YZ_3D_arr[y_th:y_th_m, z_th, k] = np.max(np.vstack((np.max(this_img, axis=1), img_YZ_3D_arr[y_th:y_th_m, z_th, k])), axis=0)
    return img_XZ_3D_arr, img_YZ_3D_arr


def import_img_slab_sec_2D_YZ(img_name_format, dir_path, id_dir_dict, id_list, ch_th, strip_pos, xy_v_num, z_v_num,
                              y_range, z_range, img_type='tif', img_dtype='uint16'):
    strip_num = len(id_list)
    x_min, x_max = np.min(strip_pos[id_list, 0]), np.max(strip_pos[id_list, 0]) + xy_v_num[0]
    sam_times = x_max - x_min
    y_min, y_max = y_range
    z_min, z_max = z_range
    img_YZ_3D_arr = np.zeros((y_max-y_min, z_max-z_min, sam_times), dtype=img_dtype)
    img_pos_arr = np.zeros((sam_times, 3), dtype='int64')
    img_pos_arr[:, 1] = y_min
    img_pos_arr[:, 2] = z_min
    for i in range(sam_times):
        img_pos_arr[i, 0] = x_min + i
    for i in range(strip_num):
        this_id = id_list[i]
        this_x, this_y = strip_pos[this_id, 0], strip_pos[this_id, 1]
        x_th, x_th_m = this_x - x_min, this_x - x_min + xy_v_num[0]
        y_th, y_th_m = this_y - y_min, this_y - y_min + xy_v_num[1]
        y_th_i, y_th_m_i = 0, xy_v_num[1]
        if y_th >= y_max-y_min or y_th_m <= 0:
            continue
        if y_th < 0:
            y_th = 0
            y_th_i = y_th_m_i - (y_th_m - y_th)
        if y_th_m > y_max-y_min:
            y_th_m = y_max-y_min
            y_th_m_i = y_th_i + y_th_m - y_th
        for j in range(z_v_num[this_id]):
            this_z = strip_pos[this_id, 2] + j
            z_th = this_z - z_min
            if z_th < 0 or z_th >= z_max-z_min:
                continue
            this_img = import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[this_id]), j, ch_th, img_type=img_type)
            img_YZ_3D_arr[y_th:y_th_m, z_th, x_th:x_th_m] = this_img[y_th_i:y_th_m_i, :]
    return img_YZ_3D_arr, img_pos_arr


def import_img_whole_2D_max_YZ(img_name_format, dir_path, id_dir_dict, ch_th, strip_pos, xy_v_num, z_v_num, x_start,
                               max_pro_num=30, img_type='tif', img_dtype='uint16'):
    strip_num = strip_pos.shape[0]
    id_list = []
    for i in range(strip_num):
        if strip_pos[i, 0] > x_start+max_pro_num-1 or strip_pos[i, 0]+xy_v_num[0]-1 < x_start:
            continue
        else:
            id_list.append(i)
    strip_num = len(id_list)
    y_min, y_max = np.min(strip_pos[id_list, 1]), np.max(strip_pos[id_list, 1]) + xy_v_num[1]
    z_min, z_max = np.min(strip_pos[id_list, 2]), np.max(strip_pos[id_list, 2] + z_v_num[id_list])
    img_YZ_arr = np.zeors((y_max-y_min, z_max-z_min), dtype=img_dtype)
    for i in id_list:
        y_th, y_th_m = strip_pos[i, 1] - y_min, strip_pos[i, 1] - y_min + xy_v_num[1]
        x_th, x_th_m = x_start - strip_pos[i, 0], x_start - strip_pos[i, 0] + max_pro_num
        if x_th <0:
            x_th = 0
        if x_th_m > xy_v_num[0]:
            x_th_m = xy_v_num[0]
        for j in range(z_v_num[i]):
            this_z = strip_pos[i, 2] + j
            z_th = this_z - z_min
            this_img = import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[i]), j, ch_th, img_type=img_type)
            img_YZ_arr[y_th:y_th_m, z_th] = np.max(np.vstack((np.max(this_img[:, x_th:x_th_m], axis=1),img_YZ_arr[y_th:y_th_m, z_th])), axis=0)
    return img_YZ_arr


def import_img_whole_2D_XY(img_name_format, dir_path, id_dir_dict, z_pos, ch_th, strip_pos, xy_v_num, z_v_num,
                           img_type='tif', img_dtype='uint16', if_bright_edge = False):
    if img_dtype == 'uint8':
        max_num = 255
    elif img_dtype == 'uint16':
        max_num = 40000
    strip_num = strip_pos.shape[0]
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    this_img_whole = np.zeros((y_max-y_min, x_max-x_min), dtype=img_dtype)
    for i in range(strip_num):
        z_th = z_pos - strip_pos[i, 2]
        x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
        if z_th < 0 or z_th >= z_v_num[i]:
            continue
        this_img_whole[y_th:y_th+xy_v_num[1], x_th:x_th+xy_v_num[0]] = \
            import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[i]), z_th, ch_th, img_type=img_type)
    if if_bright_edge:
        for i in range(strip_num):
            x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
            this_img_whole[y_th, x_th:x_th+xy_v_num[0]] = max_num
            this_img_whole[y_th+xy_v_num[1]-1, x_th:x_th+xy_v_num[0]] = max_num
            this_img_whole[y_th:y_th+xy_v_num[1], x_th] = max_num
            this_img_whole[y_th:y_th+xy_v_num[1], x_th+xy_v_num[0]-1] = max_num
    return this_img_whole


def import_img_2D_XY_by_id(img_name_format, dir_path, id_dir_dict, id_list, z_pos, ch_th, strip_pos, xy_v_num, z_v_num,
                           img_type='tif', img_dtype='uint16', if_adjust_contrast=True, if_bright_edge=False):
    x_min, x_max = np.min(strip_pos[[id_list], 0]), np.max(strip_pos[[id_list], 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[[id_list], 1]), np.max(strip_pos[[id_list], 1]) + xy_v_num[1] + 1
    this_img = np.zeros((y_max - y_min, x_max - x_min), dtype=img_dtype)
    for i in id_list:
        z_th = z_pos - strip_pos[i, 2]
        if z_th < 0 or z_th >= z_v_num[i]:
            continue
        x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
        this_img[y_th:y_th + xy_v_num[1], x_th:x_th + xy_v_num[0]] = \
            import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[i]), z_th, ch_th, img_type=img_type)
    if if_adjust_contrast:
        this_img = np.uint16(np.clip((this_img.astype('float64')*6), 0, 65535))
    if if_bright_edge:
        if img_dtype == 'uint16':
            max_num = 65535
        elif img_dtype == 'uint8':
            max_num = 255
        # bright_num = 0
        for i in id_list:
            # bright_num = bright_num + 1
            # if bright_num >=10:
            #     break
            x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
            x_th_m, y_th_m = x_th + xy_v_num[0], y_th + xy_v_num[1]
            this_img[y_th:y_th_m, x_th] = max_num
            this_img[y_th:y_th_m, x_th_m-1] = max_num
            this_img[y_th, x_th:x_th_m] = max_num
            this_img[y_th_m-1, x_th:x_th_m] = max_num
    return this_img


def import_img_whole_2D_XY_max(img_name_format, dir_path, id_dir_dict, z_pos, z_depth, ch_th, strip_pos, xy_v_num,
                               z_v_num, img_type='tif', img_dtype='uint16'):
    strip_num = strip_pos.shape[0]
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    img_whole_max = np.zeros((y_max - y_min, x_max - x_min), dtype=img_dtype)
    for z in range(z_pos, z_pos+z_depth):
        this_img_whole = np.zeros((y_max - y_min, x_max - x_min), dtype=img_dtype)
        for i in range(strip_num):
            z_th = z - strip_pos[i, 2]
            x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
            if z_th < 0 or z_th >= z_v_num[i]:
                continue
            this_img = import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[i]), z_th, ch_th, img_type=img_type)
            this_img_whole[y_th:y_th + xy_v_num[1], x_th:x_th + xy_v_num[0]] = \
                np.max(np.stack((this_img, this_img_whole[y_th:y_th + xy_v_num[1], x_th:x_th + xy_v_num[0]]), axis=2), axis=2)
        img_whole_max = np.max(np.stack((img_whole_max, this_img_whole), axis=2), axis=2)
    return img_whole_max


def import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z_pos, ch_th, strip_pos, xy_v_num,
                                 z_v_num, img_type='tif', img_dtype='uint16'):
    strip_num = strip_pos.shape[0]
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    img_with_pos_list = []
    for i in range(strip_num):
        z_th = z_pos - strip_pos[i, 2]
        x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
        if z_th < 0 or z_th >= z_v_num[i]:
            continue
        this_img = import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[i]), z_th, ch_th, img_type=img_type)
        img_with_pos_list.append([this_img, [[x_th, x_th+xy_v_num[0]], [y_th,y_th+xy_v_num[1]]]])
    return MergeSolution(img_with_pos_list, [y_max-y_min, x_max-x_min]).do()


def import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z_pos, z_depth, ch_th, strip_pos, xy_v_num,
                                     z_v_num, img_type='tif', img_dtype='uint16', order='bm'):
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    if order == 'bm':
        img_whole_max = np.zeros((y_max - y_min, x_max - x_min), dtype=img_dtype)
        for z in range(z_pos, z_pos+z_depth):
            this_img_whole = import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z, ch_th, strip_pos,
                                                          xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
            img_whole_max = np.max(np.stack((img_whole_max, this_img_whole), axis=2), axis=2)
        return img_whole_max
    elif order == 'mb':
        img_with_pos_list=[]
        strip_num = strip_pos.shape[0]
        for i in range(strip_num):
            x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
            this_img = np.zeros((xy_v_num[1], xy_v_num[0]), dtype='int64')
            for z in range(z_pos, z_pos+z_depth):
                z_th = z - strip_pos[i, 2]
                if z_th < 0 or z_th >= z_v_num[i]:
                    continue
                this_img = np.max(np.stack((this_img, import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[i]), z_th, ch_th, img_type=img_type)), axis=2), axis=2)
            img_with_pos_list.append([this_img, [[x_th, x_th+xy_v_num[0]], [y_th,y_th+xy_v_num[1]]]])
        return MergeSolution(img_with_pos_list, [y_max-y_min, x_max-x_min]).do()

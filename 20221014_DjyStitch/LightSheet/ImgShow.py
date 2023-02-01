import numpy as np
import os
import cv2
import ctypes
from merge import MergeSolution
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from ImgIO import import_img_whole_2D_XY, import_img_2D_XY_by_id, import_img_whole_2D_XY_max
from ImgIO import import_slab_2D_max_XZ_YZ, import_img_2D_one
from InfoIO import get_layer_id_dict_info, get_img_dim_info, get_strip_pos_info, get_id_dir_dict_info
from LayerStitchSIFT import choose_layer_index

"""
def import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z_pos, ch_th, strip_pos, xy_v_num,
                                 z_v_num, img_type='tif', img_dtype='uint16'):
    strip_num = strip_pos.shape[0]
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    img_with_pos_list = []
    for i in range(strip_num):
        if i<=185:
            dir_path_real = dir_path[0]
        else:
            dir_path_real = dir_path[1]
        z_th = z_pos - strip_pos[i, 2]
        x_th, y_th = strip_pos[i, 0] - x_min, strip_pos[i, 1] - y_min
        if z_th < 0 or z_th >= z_v_num[i]:
            continue
        this_img = import_img_2D_one(img_name_format, os.path.join(dir_path_real, id_dir_dict[i]), z_th, ch_th, img_type=img_type)
        img_with_pos_list.append([this_img, [[x_th, x_th+xy_v_num[0]], [y_th,y_th+xy_v_num[1]]]])
    return MergeSolution(img_with_pos_list, [y_max-y_min, x_max-x_min]).do()


def import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z_pos, z_depth, ch_th, strip_pos, xy_v_num,
                                     z_v_num, img_type='tif', img_dtype='uint16'):
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    img_whole_max = np.zeros((y_max - y_min, x_max - x_min), dtype=img_dtype)
    for z in range(z_pos, z_pos+z_depth):
        this_img_whole = import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z, ch_th, strip_pos,
                                                      xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
        img_whole_max = np.max(np.stack((img_whole_max, this_img_whole), axis=2), axis=2)
    return img_whole_max"""


def import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z_pos, z_depth, ch_th, strip_pos, xy_v_num,
                                     z_v_num, img_type='tif', img_dtype='uint16'):
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    y_min, y_max = np.min(strip_pos[:, 1]), np.max(strip_pos[:, 1]) + xy_v_num[1] + 1
    strip_num = strip_pos.shape[0]
    img_with_pos_list = []
    for i in range(strip_num):
        if i <= 185:
            dir_path_real = dir_path[0]
        else:
            dir_path_real = dir_path[1]
        this_x, this_y = strip_pos[i, 0], strip_pos[i, 1]
        x_th, x_th_m = this_x - x_min, this_x - x_min + xy_v_num[0]
        y_th, y_th_m = this_y - y_min, this_y - y_min + xy_v_num[1]
        this_img_max = np.zeros((xy_v_num[1], xy_v_num[0]), dtype=img_dtype)
        for j in range(z_v_num[i]):
            this_z = strip_pos[i, 2] + j
            if not z_pos <= this_z < z_pos + z_depth:
                continue
            this_img = import_img_2D_one(img_name_format, os.path.join(dir_path_real, id_dir_dict[i]), j, ch_th, img_type=img_type)
            this_img_max = np.max(np.stack((this_img, this_img_max), axis=0),axis=0)
        img_with_pos_list.append([this_img_max, [[x_th, x_th_m], [y_th, y_th_m]]])
    return MergeSolution(img_with_pos_list, [y_max - y_min, x_max - x_min]).do()


def one_XZ_YZ_import(lock, dir_path, img_save_path, id_dir_dict, layer_id_dict, strip_pos, img_name_format, channel_num,
                     xy_v_num, z_v_num, if_layer_output, img_type='tif', img_dtype='uint16'):
    while False in if_layer_output:
        lock.acquire()
        i = choose_layer_index(if_layer_output)
        if i==-1:
            lock.release()
            break
        if_layer_output[i] = True
        lock.release()
        if i <= 9:
            dir_path_real = dir_path[0]
        else:
            dir_path_real = dir_path[1]
        img_XZ, img_YZ = import_slab_2D_max_XZ_YZ(img_name_format, dir_path_real, id_dir_dict, layer_id_dict[i], channel_num,
                                                  strip_pos, xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
        np.save(os.path.join(img_save_path, 'slab%.2d_XZ_maxpro.npy' % i), img_XZ)
        np.save(os.path.join(img_save_path, 'slab%.2d_YZ_maxpro.npy' % i), img_YZ)
        # img_XZ, img_YZ = np.array([]), np.array([])


def start_multi_XZ_YZ_import(txt_path, dir_path, img_save_path, channel_num, img_name_format, img_type='tif'):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    img_dtype = 'uint16'
    xy_v_num = np.array([1024, 2048], dtype='int64')
    z_v_num = np.load(os.path.join(os.path.split(txt_path)[0], 'z_v_num.npy'))
    z_v_num[[165, 166, 167, 168, 169, 170, 171, 172, 173]] = 8000
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)
    if_layer_output = Array(ctypes.c_bool, [False for i in range(layer_num)])
    # dir_path_list=[]
    # for i in range(0,229):
    #     dir_path_list.append(os.path.join(dir_path[0], id_dir_dict[i]))
    # for i in range(229,strip_num):
    #     dir_path_list.append(os.path.join(dir_path[1], id_dir_dict[i]))

    lock = RLock()
    process_num = 10
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=one_XZ_YZ_import, args=(lock, dir_path, img_save_path, id_dir_dict, layer_id_dict,
                                                         strip_pos, img_name_format, channel_num, xy_v_num, z_v_num,
                                                         if_layer_output, img_type, img_dtype))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()


def one_XY_import(lock, dir_path, img_save_path, id_dir_dict, strip_pos, img_name_format, output_z_list,
                  z_depth, xy_v_num, z_v_num, if_layer_output, img_type='tif', img_dtype='uint16'):
    while False in if_layer_output:
        lock.acquire()
        i = choose_layer_index(if_layer_output)
        if i == -1:
            lock.release()
            break
        if_layer_output[i] = True
        lock.release()
        this_img = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, output_z_list[i], z_depth,
                                                0, strip_pos, xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
        whole_img = np.zeros((this_img.shape[0], this_img.shape[1], 3), dtype='uint16')
        whole_img[:, :, 0] = this_img
        this_img = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, output_z_list[i], z_depth,
                                                1, strip_pos, xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
        whole_img[:, :, 1] = this_img
        np.save(os.path.join(img_save_path, 'z%.6d_depth%.4d_merged.npy' % (output_z_list[i], z_depth)), whole_img)


def start_multi_XY_import(txt_path, dir_path, img_save_path, channel_num, img_name_format, img_type='tif', z_depth=500):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    img_dtype = 'uint16'
    xy_v_num = np.array([1024, 2048], dtype='int64')
    z_v_num = np.load(os.path.join(os.path.split(txt_path)[0], 'z_v_num.npy'))
    z_v_num[[165, 166, 167, 168, 169, 170, 171, 172, 173]] = 8000
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)

    output_z_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    output_num = len(output_z_list)
    if_layer_output = Array(ctypes.c_bool, [False for i in range(output_num)])

    lock = RLock()
    process_num = 7
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=one_XY_import, args=(lock, dir_path, img_save_path, id_dir_dict, strip_pos,
                                                      img_name_format, output_z_list, z_depth, xy_v_num, z_v_num,
                                                      if_layer_output, img_type, img_dtype))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()


def import_img_whole_sec_2D_XZ_max(img_name_format, dir_path, id_dir_dict, y_start, depth, ch_th, strip_pos, xy_v_num,
                                   z_v_num, img_type='tif', img_dtype='uint16'):
    strip_num = strip_pos.shape[0]
    x_min, x_max = np.min(strip_pos[:, 0]), np.max(strip_pos[:, 0]) + xy_v_num[0] + 1
    z_min, z_max = np.min(strip_pos[:, 2]), np.max(strip_pos[:, 2] + z_v_num) + 1
    y_min, y_max = y_start, y_start + depth
    whole_img = np.zeros((x_max-x_min, z_max-z_min), dtype=img_dtype)
    for i in range(strip_num):
        if i <= 185:
            dir_path_real = dir_path[0]
        else:
            dir_path_real = dir_path[1]
        this_x, this_y = strip_pos[i, 0], strip_pos[i, 1]
        y_th, y_th_m = y_min - this_y, y_min - this_y + depth
        if y_th_m <= 0 or y_th >= xy_v_num[1]:
            continue
        if y_th < 0:
            y_th = 0
        if y_th_m > xy_v_num[1]:
            y_th_m = xy_v_num[1]
        x_th, x_th_m = this_x - x_min, this_x - x_min + xy_v_num[0]
        for j in range(z_v_num[i]):
            this_z = strip_pos[i, 2] + j
            z_th = this_z - z_min
            this_img = import_img_2D_one(img_name_format, os.path.join(dir_path_real, id_dir_dict[i]), j, ch_th, img_type=img_type)
            whole_img[x_th:x_th_m, z_th] = np.max(np.vstack((np.max(this_img[y_th:y_th_m, :], axis=0), whole_img[x_th:x_th_m, z_th])), axis=0)
    return whole_img


def one_sec_XZ_import(txt_path, dir_path, img_save_path, img_name_format, y_start, depth, channel_num, img_tpye='tif'):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    img_dtype = 'uint16'
    xy_v_num = np.array([1024, 2048], dtype='int64')
    z_v_num = np.load(os.path.join(os.path.split(txt_path)[0], 'z_v_num.npy'))
    z_v_num[[165, 166, 167, 168, 169, 170, 171, 172, 173]] = 8000
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)

    this_img = import_img_whole_sec_2D_XZ_max(img_name_format, dir_path, id_dir_dict, y_start, depth, 0, strip_pos,
                                              xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
    whole_img = np.zeros((this_img.shape[0], this_img.shape[1], 3), dtype='uint16')
    whole_img[:, :, 0] = this_img
    this_img = import_img_whole_sec_2D_XZ_max(img_name_format, dir_path, id_dir_dict, y_start, depth,  1, strip_pos,
                                              xy_v_num, z_v_num, img_type=img_type, img_dtype=img_dtype)
    whole_img[:, :, 1] = this_img
    np.save(os.path.join(img_save_path, 'y%.6d_depth%.4d.npy'%(y_start, depth)), whole_img)


if __name__ == '__main__':
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_20221124.txt'
    dir_path1 = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/220908/frames'
    dir_path2 = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/220923/frames'
    img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221124'
    img_name_format = r'%s_%s'
    channel_num = 2
    img_type = 'tif'
    # start_multi_XZ_YZ_import(txt_path, [dir_path1, dir_path2], img_save_path, channel_num, img_name_format, img_type=img_type)

    # img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221124'
    # z_depth = 500
    # start_multi_XY_import(txt_path, [dir_path1, dir_path2], img_save_path, channel_num, img_name_format, img_type=img_type, z_depth=z_depth)

    img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221124'
    y_start = 10000
    y_depth = 10
    one_sec_XZ_import(txt_path, [dir_path1, dir_path2], img_save_path, img_name_format, y_start, y_depth, channel_num, img_tpye='tif')




    # strip_pos = get_strip_pos_info(txt_path)
    # id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    # layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    # xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
    #                                                 img_type=img_type)
    # z_v_num[[165, 166, 167, 168, 169, 170, 171, 172, 173]] = 8000  # 165-173

    # for z_pos in range(3000, 3001):
    #     for ch_th in range(1):
    #         this_img = import_img_whole_2D_XY(img_name_format, dir_path, id_dir_dict, z_pos, ch_th, strip_pos, xy_v_num,
    #                                           z_v_num, img_type=img_type, img_dtype=img_dtype, if_bright_edge=False)
    #         img_name = 'z%.6d_ch%.2d.%s'%(z_pos, ch_th, img_type)
    #         cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #         print(img_name, 'saved')

    # for l_th in layer_id_dict.keys():
    #     for z_pos in [500, 3000]:
    #         for ch_th in range(2):
    #             this_img = import_img_2D_XY_by_id(img_name_format, dir_path, id_dir_dict, layer_id_dict[l_th],
    #                                               z_pos, ch_th, strip_pos, xy_v_num, z_v_num,
    #                                               img_type='tif', img_dtype='uint16', if_bright_edge=True)
    #             img_name = 'slab%.3d_z%.6d_ch%.2d.%s' % (l_th, z_pos, ch_th, img_type)
    #             cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #             print(img_name, 'saved')

    # id_list = layer_id_dict[18]
    # z_min = np.min(strip_pos[[id_list], 2])
    # for z_pos in range(1500, 1550, 5):
    #     this_img1 = import_img_2D_XY_by_id(img_name_format, dir_path, id_dir_dict, id_list, z_pos, 0, strip_pos,
    #                                        xy_v_num, z_v_num, img_type='tif', img_dtype='uint16',
    #                                        if_adjust_contrast=False, if_bright_edge=False)
    #     this_img2 = import_img_2D_XY_by_id(img_name_format, dir_path, id_dir_dict, id_list, z_pos, 1, strip_pos,
    #                                        xy_v_num, z_v_num, img_type='tif', img_dtype='uint16',
    #                                        if_adjust_contrast=False, if_bright_edge=False)
    #     this_img = np.zeros((this_img1.shape[0], this_img1.shape[1], 3), dtype='uint16')
    #     this_img[:, :, 0] = this_img1
    #     this_img[:, :, 1] = this_img2
    #     img_name = 'slab%.3d_z%.6d.%s' % (9, z_pos, img_type)
    #     cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #     print(img_name, 'saved')

    # layer_list = [17, 18]
    # for l_th in layer_list:
    #     id_list = layer_id_dict[l_th]
    #     for z_pos in range(1500, 1550, 5):
    #         this_img = import_img_2D_XY_by_id(img_name_format, dir_path, id_dir_dict, id_list, z_pos, 0, strip_pos,
    #                                           xy_v_num, z_v_num, img_type='tif', img_dtype='uint16',
    #                                           if_adjust_contrast=False, if_bright_edge=True)
    #         img_name = 'slab%.3d_z%.6d_ch%.2d.%s' % (l_th, z_pos, 0, img_type)
    #         cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #         print(img_name, 'saved')

    # z = 3000
    # ch_th = 0
    # z_depth=30
    # this_img = import_img_whole_2D_XY_max(img_name_format, dir_path, id_dir_dict, z, z_depth, ch_th, strip_pos, xy_v_num, z_v_num)
    # img_name = 'z%.6d_depth%.3d_ch%.2d.tif' % (z, z_depth, ch_th)
    # cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    # print(img_name, 'saved')

    # ch_th = 1
    # for z_depth in [30,50]:
    #     this_img = import_img_whole_2D_XY_max(img_name_format, dir_path, id_dir_dict, z, z_depth, ch_th, strip_pos,
    #                                           xy_v_num, z_v_num)
    #     img_name = 'z%.6d_depth%.3d_ch%.2d.tif' % (z, z_depth, ch_th)
    #     cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #     print(img_name, 'saved')


    # this_img = import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z, ch_th, strip_pos, xy_v_num, z_v_num)
    # img_name = 'z%.6d_ch%.2d_merged.tif' % (z, ch_th)
    # cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    # print(img_name, 'saved')

    # z_depth = 5
    # this_img = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z, z_depth, ch_th, strip_pos, xy_v_num, z_v_num)
    # img_name = 'z%.6d_depth%.3d_ch%.2d_merged.tif' % (z, z_depth, ch_th)
    # cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    # print(img_name, 'saved')

    # 先只用最大值映射
    # z = 3000
    # img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221029maxpro'
    # for z_depth in [5, 10, 30, 50, 100, 200, 500]:
    #     this_img1 = import_img_whole_2D_XY_max(img_name_format, dir_path, id_dir_dict, z, z_depth, 0, strip_pos, xy_v_num, z_v_num)
    #     this_img2 = import_img_whole_2D_XY_max(img_name_format, dir_path, id_dir_dict, z, z_depth, 1, strip_pos, xy_v_num, z_v_num)
    #     img_name = 'z%.6d_depth%.4d.tif' % (z, z_depth)
    #     this_img = np.zeros((this_img1.shape[0], this_img1.shape[1], 3), dtype='uint16')
    #     this_img[:, :, 0] = this_img1
    #     this_img[:, :, 1] = this_img2
    #     cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #     print(img_name, 'saved')

    # 先用最大值映射，再用blending
    # z = 3000
    # img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221029maxproblend'
    # for z_depth in [1, 10, 30, 50, 100, 200, 500]:
    #     this_img1 = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z, z_depth, 0, strip_pos, xy_v_num, z_v_num, order='mb')
    #     this_img2 = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z, z_depth, 1, strip_pos, xy_v_num, z_v_num, order='mb')
    #     img_name = 'z%.6d_depth%.4d_merged_order_mb.tif' % (z, z_depth)
    #     this_img = np.zeros((this_img1.shape[0], this_img1.shape[1], 3), dtype='uint16')
    #     this_img[:, :, 0] = this_img1
    #     this_img[:, :, 1] = this_img2
    #     cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #     print(img_name, 'saved')

    # 先用blending再用最大值映射
    # z = 8000
    # img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221101outputwhole'
    # for z_depth in [1]:
    #     this_img1 = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z, z_depth, 0, strip_pos,
    #                                                  xy_v_num, z_v_num, order='bm')
    #     this_img2 = import_img_whole_2D_XY_max_merge(img_name_format, dir_path, id_dir_dict, z, z_depth, 1, strip_pos,
    #                                                  xy_v_num, z_v_num, order='bm')
    #     img_name = 'z%.6d_depth%.4d_merged_order_bm.tif' % (z, z_depth)
    #     this_img = np.zeros((this_img1.shape[0], this_img1.shape[1], 3), dtype='uint16')
    #     this_img[:, :, 0] = this_img1
    #     this_img[:, :, 1] = this_img2
    #     cv2.imwrite(os.path.join(img_save_path, img_name), this_img)
    #     print(img_name, 'saved')


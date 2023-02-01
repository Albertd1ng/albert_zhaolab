import os.path
import time
import cv2
import numpy as np
import random
import ctypes
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_img_dim_info, save_img_txt_info, get_layer_id_dict_info, get_strip_pos_info, get_id_dir_dict_info
from ImgIO import import_img_2D_one, import_img_3D_sec_by_z
from StripCont import judge_strip_cont, judge_strip_cont_one
from ImgBorder import get_two_strip_sec_border, get_border_pyr_down
from ImgOvl import get_ovl_img, get_ovl_img_max
from LossFunc import loss_func_for_list
from RunningIO import run_print
from ImgProcess import adjust_contrast, pyr_down_img


def choose_start_index(dir_path, id_dir_dict, z_start, img_name_format, ch_th, img_type='tif',
                       thred=10, choose_range=[0.3, 0.7]):
    strip_num = len(id_dir_dict)
    m, a =0, 0
    while m < thred:
        a = random.randint(int(choose_range[0] * strip_num), int(choose_range[1] * strip_num))
        m = np.mean(import_img_2D_one(img_name_format, os.path.join(dir_path, id_dir_dict[a]), z_start, ch_th, img_type=img_type))
    return a


def choose_stitched_index(if_strip_stitched, if_strip_being_stitched, if_strip_shelved):
    for i, j in enumerate(zip(if_strip_stitched, if_strip_being_stitched, if_strip_shelved)):
        if not any(j):
            return i
    return -1


def choose_refer_index(i, if_strip_stitched, strip_cont_vec):
    index_j = []
    for j in range(len(if_strip_stitched)):
        if strip_cont_vec[j] and if_strip_stitched[j]:
            index_j.append(j)
    if len(index_j) == 0:
        return -1
    index_j = np.array(index_j, dtype='int64')
    dis_ij = np.abs(index_j - i)
    min_dis_index = np.argwhere(dis_ij == np.min(dis_ij))
    j = index_j[min_dis_index[0, 0]]
    return j


def choose_refer_index_list(if_strip_stitched, strip_cont_vec):
    index_j = []
    for j in range(len(if_strip_stitched)):
        if strip_cont_vec[j] and if_strip_stitched[j]:
            index_j.append(j)
    return index_j


def calc_xyz_shift_for_list(img1_sec, img2_sec_list, img1_pos, img2_pos, xy_v_num, z_depth, voxel_range, pyr_down_times, alpha):
    refer_num = img2_pos.shape[0]
    for pdt in range(pyr_down_times, -1, -1):
        if pdt == pyr_down_times:
            x_s, y_s, z_s = 0, 0, 0
            x_pd, y_pd, z_pd = np.int64(np.round(np.array(voxel_range, dtype='int64') / (2 ** pdt)))
        else:
            x_s, y_s, z_s = 2 * np.array((x_s, y_s, z_s), dtype='int64')
            x_pd, y_pd, z_pd = 3, 3, 3
        loss_min = np.inf
        if pdt != 0:
            img1_sec_pd = pyr_down_img(img1_sec, pdt)
            img2_sec_list_pd = [pyr_down_img(one_img, pdt) for one_img in img2_sec_list]
        else:
            img1_sec_pd = img1_sec.copy()
            img2_sec_list_pd = [one_img.copy() for one_img in img2_sec_list]
        down_multi = np.array([*img1_sec_pd.shape[1::-1], img1_sec_pd.shape[2]], dtype='float32') / np.array(
            [*xy_v_num, z_depth], dtype='float32')
        x_sr, y_sr, z_sr = x_s, y_s, z_s
        for x in range(x_sr - x_pd, x_sr + x_pd + 1):
            for y in range(y_sr - y_pd, y_sr + y_pd + 1):
                for z in range(z_sr - z_pd, z_sr + z_pd + 1):
                    this_img1_pos, this_img2_pos = img1_pos.copy(), img2_pos.copy()
                    this_img1_pos = this_img1_pos + np.array([x, y, z], dtype='int64') * (2 ** pdt)
                    border_list = [get_border_pyr_down(
                        get_two_strip_sec_border(np.vstack((this_img1_pos, this_img2_pos[i, :])), xy_v_num, z_depth),
                        down_multi) for i in range(refer_num)]
                    shift_ovl_id_list = np.zeros(refer_num, dtype='bool')
                    for i in range(refer_num):
                        if border_list[i].shape[0] != 0:
                            shift_ovl_id_list[i] = True
                    ovl1_list, ovl2_list = [], []
                    for i in range(refer_num):
                        if not shift_ovl_id_list[i]:
                            continue
                        this_ovl1_list, this_ovl2_list = get_ovl_img_max(img1_sec_pd, img2_sec_list_pd[i], border_list[i])
                        for j,k in zip(this_ovl1_list, this_ovl2_list):
                            if j.shape == k.shape:
                                ovl1_list.append(j)
                                ovl2_list.append(k)
                    # if pdt == 0 and x == x_sr and y == y_sr and z == z_sr:
                    #     for l in range(len(ovl1_list)):
                    #         cv2.imwrite(r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221027test/ovl01_%.4d.tif' % l,
                    #                     ovl1_list[l])
                    #         cv2.imwrite(r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221027test/ovl02_%.4d.tif' % l,
                    #                     ovl2_list[l])
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list, [x, y, z], alpha=alpha)
                    if this_loss < loss_min:
                        loss_min = this_loss
                        x_s, y_s, z_s = x, y, z
    return [x_s, y_s, z_s], loss_min


def strip_sec_stitch(lock, dir_path, running_IO_path, id_dir_dict, strip_pos, strip_pos_sec, strip_cont, z_start,
                     img_name_format, ch_th, img_type, img_dtype, xy_v_num, z_v_num, if_strip_stitched, if_strip_being_stitched,
                     if_strip_shelved, voxel_range, pyr_down_times, z_depth, alpha):
    this_name = current_process().name
    strip_num = strip_pos.shape[0]
    while False in if_strip_stitched:
        lock.acquire()
        i = choose_stitched_index(if_strip_stitched, if_strip_being_stitched, if_strip_shelved)
        if i == -1:
            for k in range(strip_num):
                if_strip_shelved[k] = False
            time.sleep(1)
            lock.release()
            continue
        j = choose_refer_index(i, if_strip_stitched, strip_cont[i, :])
        if j == -1:
            if_strip_shelved[i] = True
            lock.release()
            continue
        if_strip_being_stitched[i] = True
        run_print(lock, running_IO_path, str(strip_pos_sec[3 * i:3 * i + 3]), str(strip_pos_sec[3 * j:3 * j + 3]))
        strip_pos_sec[3 * i:3 * i + 3] = strip_pos_sec[3 * j:3 * j + 3] + strip_pos[i, :] - strip_pos[j, :]
        run_print(lock, running_IO_path, str(strip_pos_sec[3 * i:3 * i + 3]), str(strip_pos_sec[3 * j:3 * j + 3]))
        lock.release()

        strip_cont_vec = judge_strip_cont_one(i, np.array(strip_pos_sec).reshape(-1,3), xy_v_num)
        j_list = choose_refer_index_list(if_strip_stitched, strip_cont_vec)
        img_path1 = os.path.join(dir_path, id_dir_dict[i])
        img_path2_list = [os.path.join(dir_path, id_dir_dict[j]) for j in j_list]
        border_list = [get_two_strip_sec_border(np.vstack((strip_pos_sec[3 * i:3 * i + 3], strip_pos_sec[3 * j:3 * j + 3])),
                                                xy_v_num, z_depth) for j in j_list]
        for k in range(len(border_list) - 1, -1, -1):
            if border_list[k].shape[0] == 0:
                border_list.pop(k)
                img_path2_list.pop(k)
                j_list.pop(k)
        if (refer_num := len(img_path2_list)) == 0:
            lock.acquire()
            if_strip_stitched[i] = True
            if_strip_being_stitched[i] = False
            lock.release()
            run_print(lock, running_IO_path, 'Warning: %s: %d.th strip end stitching for no overlapped section' % (this_name, i))
            continue
        img1_sec = import_img_3D_sec_by_z(img_name_format, img_path1, strip_pos[i,2], z_start, z_depth, ch_th, xy_v_num,
                                          img_type=img_type, img_dtype=img_dtype)
        if np.mean(img1_sec) < 0.1:
            lock.acquire()
            if_strip_stitched[i] = True
            if_strip_being_stitched[i] = False
            lock.release()
            run_print(lock, running_IO_path, 'Warning: %s: %d.th strip end stitching for low mean value' % (this_name, i))
            continue
        img2_sec_list = [import_img_3D_sec_by_z(img_name_format, img_path2_list[j], strip_pos[j_list[j], 2], z_start,
                                                z_depth, ch_th, xy_v_num, img_type=img_type, img_dtype=img_dtype)
                         for j in range(len(j_list))]
        run_print(lock, running_IO_path,
                  '%s: start stitching the %d.th strip with %s strips, refer strip is %s.th strip' % (this_name, i, str(j_list), j))
        # run_print(lock, running_IO_path, str(np.mean(img1_sec)), str([np.mean(img2_sec) for img2_sec in img2_sec_list]))

        img1_pos = np.array(strip_pos_sec[3 * i:3 * i + 3], dtype='int64')
        img2_pos = np.zeros((refer_num, 3), dtype='int64')
        for j in range(refer_num):
            img2_pos[j, :] = strip_pos_sec[3 * j_list[j]:3 * j_list[j] + 3]
        xyz_shift, loss_min = calc_xyz_shift_for_list(img1_sec, img2_sec_list, img1_pos, img2_pos, xy_v_num, z_depth, voxel_range, pyr_down_times, alpha)
        lock.acquire()
        run_print(lock, running_IO_path, str(strip_pos_sec[3 * i:3 * i + 3]), str(xyz_shift))
        strip_pos_sec[3 * i:3 * i + 3] = strip_pos_sec[3 * i:3 * i + 3] + np.array(xyz_shift, dtype='int64')
        run_print(lock, running_IO_path, str(strip_pos_sec[3 * i:3 * i + 3]), str(xyz_shift))
        if_strip_stitched[i] = True
        if_strip_being_stitched[i] = False
        run_print(lock, running_IO_path,
                  '%s: end stitching the %d.th strip with %s strips, xyz shift is %s, loss is %.6f' % (this_name, i, str(j_list), str(xyz_shift), loss_min))
        strip_pos_sec_IO = np.array(strip_pos_sec, dtype='int64').reshape(-1, 3)
        if_strip_stitched_IO = np.array(if_strip_stitched, dtype='bool')
        np.save(os.path.join(os.path.split(running_IO_path)[0], 'strip_pos_sec_z%.6d.npy' % z_start), strip_pos_sec_IO)
        np.save(os.path.join(os.path.split(running_IO_path)[0], 'if_strip_stitched_z%.6d.npy' % z_start), if_strip_stitched_IO)
        lock.release()


def update_strip_pos_sec(strip_pos, strip_pos_sec, z_start):
    strip_pos_update = strip_pos_sec.copy()
    strip_pos_update[:, 2] = strip_pos[:, 2] + strip_pos_sec[:, 2] - z_start
    return strip_pos_update


def start_multi_strip_sec_stitch(txt_path, dir_path, running_IO_path, z_start, img_name_format, channel_num, ch_th,
                                 img_type='tif', voxel_range=[50,50,50], pyr_down_times=2, z_depth=100, alpha=0):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    strip_cont = judge_strip_cont(strip_pos, xy_v_num)
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)
    lock = RLock()
    if_strip_stitched = Array(ctypes.c_bool, [False for i in range(strip_num)])
    start_id = choose_start_index(dir_path, id_dir_dict, z_start, img_name_format, ch_th, img_type, thred=1000)
    if_strip_stitched[start_id] = True
    if_strip_being_stitched = Array(ctypes.c_bool, [False for i in range(strip_num)])
    run_print(lock, running_IO_path, 'start strip is %d.th' % start_id)
    if_strip_shelved = Array(ctypes.c_bool, [False for i in range(strip_num)])
    strip_pos_sec = Array('l', [0 for i in range(strip_num * 3)])
    for i in range(strip_num):
        strip_pos_sec[3 * i] = strip_pos[i, 0]
        strip_pos_sec[3 * i + 1] = strip_pos[i, 1]
        strip_pos_sec[3 * i + 2] = z_start

    process_num = 1
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=strip_sec_stitch, args=(lock, dir_path, running_IO_path, id_dir_dict, strip_pos,
                    strip_pos_sec, strip_cont, z_start, img_name_format, ch_th, img_type, img_dtype, xy_v_num, z_v_num,
                    if_strip_stitched, if_strip_being_stitched, if_strip_shelved, voxel_range, pyr_down_times, z_depth, alpha))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()

    run_print(lock, running_IO_path, 'start calculating new position!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    strip_pos_sec = np.array(strip_pos_sec, dtype='int64').reshape(-1, 3)
    strip_pos_update = update_strip_pos_sec(strip_pos, strip_pos_sec, z_start)
    run_print(lock, running_IO_path, 'start saving data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    save_img_txt_info(os.path.join(os.path.split(txt_path)[0], 'pos_down_z%.6d_sec_stitch.txt' % z_start),
                      strip_pos_update, id_dir_dict)
    run_print(lock, running_IO_path, 'end saving data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


if __name__ == '__main__':
    """
    start_multi_strip_sec_stitch(txt_path, dir_path, running_IO_path, z_start, img_name_format, channel_num, ch_th,
                                 img_type='tif', voxel_range=[50,50,50], pyr_down_times=2, z_depth=100)
    """
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_yz_stitch_20221024.txt'
    dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/src_frames_ds331'
    running_IO_path = r'/home/zhaohu_lab/dingjiayi/20221027.txt'
    img_name_format = r'%s_%s'
    img_type = 'tif'
    channel_num = 2
    ch_th = 0
    voxel_range = [20, 20, 20]
    pyr_down_times = 1
    z_depth = 200
    z_start = 3000  # [0 - 15000]
    alpha = 0.0005
    start_multi_strip_sec_stitch(txt_path, dir_path, running_IO_path, z_start, img_name_format,
                                 channel_num, ch_th, img_type=img_type, voxel_range=voxel_range,
                                 pyr_down_times=pyr_down_times, z_depth=z_depth, alpha=alpha)

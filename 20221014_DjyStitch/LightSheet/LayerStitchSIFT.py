import numpy as np
import os
import cv2
import random
import ctypes
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_strip_pos_info, get_id_dir_dict_info, get_layer_id_dict_info, get_img_dim_info, save_img_txt_info
from RunningIO import run_print
from ImgProcess import pyr_down_img, adjust_contrast
from LossFunc import loss_func


def choose_layer_index(if_layer_stitch):
    for i in range(len(if_layer_stitch)):
        if not if_layer_stitch[i]:
            return i
    return -1


def choose_img_roi(img, thred=0):
    roi_range = np.zeros((2, 2), dtype='int64')
    lr_roi_index = [i for [i] in np.argwhere(np.max(img, axis=0) > thred)]
    ud_roi_index = [i for [i] in np.argwhere(np.max(img, axis=1) > thred)]
    if len(lr_roi_index) == 0 or len(ud_roi_index) == 0:
        return np.array([]), np.array([])
    roi_range[0, 0], roi_range[0, 1] = min(lr_roi_index), max(lr_roi_index)
    roi_range[1, 0], roi_range[1, 1] = min(ud_roi_index), max(ud_roi_index)
    return img[roi_range[1, 0]:roi_range[1, 1], roi_range[0, 0]:roi_range[0, 1]], roi_range


def calc_sift_points_match(img1,img2):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    if img1.dtype == 'uint16':
        img1 = np.uint8(np.clip(np.round(img1 / 256), 0, 255))
        img2 = np.uint8(np.clip(np.round(img2 / 256), 0, 255))
    img1, img2 = adjust_contrast(img1, img2, 10)
    kpts1, des1 = sift.detectAndCompute(img1, None)
    kpts2, des2 = sift.detectAndCompute(img2, None)
    kp1, kp2 = np.float32([kp.pt for kp in kpts1]), np.float32([kp.pt for kp in kpts2])
    # print(kp1.shape, kp2.shape)
    if kp1.shape[0] == 0 or kp2.shape[0] == 0:
        return np.array([]), np.array([])
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good_matches.append((m[0].queryIdx, m[0].trainIdx))
    pts1, pts2 = np.float32([kp1[i, :] for (i, _) in good_matches]), np.float32([kp2[j, :] for (_, j) in good_matches])
    return pts1, pts2


def calc_xy_shift_by_RANSAC(img1, img2, pts1, pts2, sample_time=1000):
    count = 0
    matches_num = pts1.shape[0]
    RANSAC_num = np.int32(np.max((np.min((4, matches_num * 0.1)),1)))
    loss_min = np.inf
    xy_shift_min = np.array([np.inf, np.inf])
    while (count < sample_time):
        count += 1
        index_list = random.sample(range(matches_num), RANSAC_num)
        xy_shift_all = pts2[index_list, :] - pts1[index_list, :]
        max_shift, min_shift = np.max(xy_shift_all, axis=0), np.min(xy_shift_all, axis=0)
        if any((max_shift - min_shift) > 100):
            continue
        xy_shift = np.int32(np.round(np.mean(xy_shift_all, axis=0)))  # XY
        if all(xy_shift == xy_shift_min):
            continue
        ovl1, ovl2 = img1[np.max((0, -xy_shift[1])):, np.max((0, -xy_shift[0])):], img2[np.max((0, xy_shift[1])):,
                                                                                   np.max((0, xy_shift[0])):]
        x_range, y_range = np.min((ovl1.shape[1], ovl2.shape[1])), np.min((ovl1.shape[0], ovl2.shape[0]))
        ovl1, ovl2 = ovl1[0:y_range, 0:x_range], ovl2[0:y_range, 0:x_range]
        this_loss = loss_func(ovl1, ovl2)
        # print(xy_shift,this_loss)
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
    # print(xy_shift_min,loss_min)
    return xy_shift_min, loss_min


def calc_xy_shift_by_BF(img1_down, img2_down, img1, img2, xy_shift, pyr_down_times,
                        if_median_blur=True, blur_kernel_size=3):
    if if_median_blur:
        img1,img2=cv2.medianBlur(img1,blur_kernel_size),cv2.medianBlur(img2,blur_kernel_size)
    # img1, img2 = adjust_contrast(img1, img2)
    xy_shift_min = np.zeros(2)
    for i in range(pyr_down_times,-1,-1):
        if i == pyr_down_times:
            img1_calc,img2_calc = img1_down,img2_down
        elif i == 0:
            img1_calc,img2_calc = img1,img2
        else:
            img1_calc,img2_calc = pyr_down_img(img1,i),pyr_down_img(img2,i)
        if i == pyr_down_times:
            loss_min = np.inf
            range_calc = 10
        else:
            xy_shift=xy_shift_min*2
            xy_shift_min = np.zeros(2)
            loss_min=np.inf
            range_calc = 5
        for x in range(-range_calc, range_calc+1):
            for y in range(-range_calc, range_calc+1):
                this_xy_shift = xy_shift + np.array([x, y], dtype='int32')
                ovl1 = img1_calc[np.max((0, -this_xy_shift[1])):, np.max((0, -this_xy_shift[0])):]
                ovl2 = img2_calc[np.max((0, this_xy_shift[1])):, np.max((0, this_xy_shift[0])):]
                x_range, y_range = np.min((ovl1.shape[1], ovl2.shape[1])), np.min((ovl1.shape[0], ovl2.shape[0]))
                ovl1, ovl2 = ovl1[0:y_range, 0:x_range], ovl2[0:y_range, 0:x_range]
                this_loss = loss_func(ovl1, ovl2)
                if this_loss < loss_min:
                    loss_min = this_loss
                    xy_shift_min = this_xy_shift
        # print('%d.th pyr down times'%i, xy_shift_min, loss_min)
    return xy_shift_min, loss_min


def update_next_layer_info(strip_pos, layer_id_dict, xyz_shift_arr):
    layer_num = len(layer_id_dict)
    for i in range(layer_num):
        id_list = layer_id_dict[i]
        strip_pos[id_list, :] = strip_pos[id_list, :] - xyz_shift_arr[i, :]
        for j in range(i + 1, layer_num):
            next_id_list = layer_id_dict[j]
            strip_pos[next_id_list, :] = strip_pos[next_id_list, :] - xyz_shift_arr[i, :]
    return strip_pos


def layer_stitch_sift(lock, txt_path, data_IO_path, running_IO_path, layer_id_dict, if_layer_stitch, sam_times, x_step,
                      pyr_down_times):
    while(False in if_layer_stitch):
        lock.acquire()
        i = choose_layer_index(if_layer_stitch)
        if i == -1:
            lock.release()
            return
        if_layer_stitch[i] = True
        lock.release()
        img1_YZ_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s.npy' % (i, sam_times, x_step, 'rl')))
        img2_YZ_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s.npy' % (i + 1, sam_times, x_step, 'lr')))
        img1_pos_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s_pos.npy' % (i, sam_times, x_step, 'rl')))
        img2_pos_arr = np.load(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s_pos.npy' % (i + 1, sam_times, x_step, 'lr')))
        xyz_shift_real = np.zeros(3, dtype='int64')
        loss_min = np.inf
        for j in range(sam_times):
            for k in range(sam_times):
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
                xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)  #notice: xy_shift here means zy_shift in reality
                run_print(lock, running_IO_path,
                          '%.2d.th pyr_img of %.2d.th layer, with %.2d.th pyr_img of %.2d.th layer, zy_shift is %s, loss is %.8f' % (j, i, k, i + 1, str(xy_shift), this_loss))
                xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(), img2.copy(), xy_shift, pyr_down_times)
                xy_shift[0] = xy_shift[0] + img2_pos_arr[k, 2] - img1_pos_arr[j, 2] + roi_range2[0, 0] - roi_range1[0, 0] # z
                xy_shift[1] = xy_shift[1] + img2_pos_arr[k, 1] - img1_pos_arr[j, 1] + roi_range2[1, 0] - roi_range1[1, 0] # y
                run_print(lock, running_IO_path,
                          '%.2d.th whole_img of %.2d.th layer, with %.2d.th whole_img of %.2d.th layer, zy_shift is %s, loss is %.8f' % (j, i, k, i + 1, str(xy_shift), this_loss))
                if this_loss < loss_min and xy_shift[0] < 2000 and xy_shift[1] < 2000:
                    loss_min = this_loss
                    xyz_shift_real[2] = xy_shift[0]  # z
                    xyz_shift_real[1] = xy_shift[1]  # y
                    xyz_shift_real[0] = img2_pos_arr[k, 0] - img1_pos_arr[j, 0]  # x
        lock.acquire()
        try:
            xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'))
        except Exception as e:
            layer_num = len(layer_id_dict)
            xyz_shift_arr = np.zeros((layer_num, 3), dtype='int64')
        xyz_shift_arr[i + 1, :] = xyz_shift_real
        np.save(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'), xyz_shift_arr)
        lock.release()
        run_print(lock, running_IO_path,
                  '################################################################################################\n',
                  '%.2d.th layer xyz_shift data saved, the final xyz_shift is %s\n' % (i + 1, str(xyz_shift_real)),
                  '################################################################################################\n')


def start_layer_stitch_sift(txt_path, save_txt_path, dir_path, data_IO_path, running_IO_path, img_name_format,
                            channel_num, ch_th, img_type, sam_times=10, x_step=30, pyr_down_times=4):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    layer_num, strip_num = len(layer_id_dict), len(id_dir_dict)

    if_layer_stitch = Array(ctypes.c_bool, [False for i in range(layer_num)])
    if_layer_stitch[layer_num - 1] = True

    lock = RLock()
    process_num = 10
    run_print(lock, running_IO_path, 'Current processing quantity: %d' % process_num)
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=layer_stitch_sift, args=(lock, txt_path, data_IO_path, running_IO_path, layer_id_dict,
                                                          if_layer_stitch, sam_times, x_step, pyr_down_times))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()

    xyz_shift_arr = np.load(os.path.join(os.path.split(txt_path)[0], 'xyz_shift_arr.npy'))
    strip_pos = update_next_layer_info(strip_pos, layer_id_dict, xyz_shift_arr)
    save_img_txt_info(save_txt_path, strip_pos, id_dir_dict)

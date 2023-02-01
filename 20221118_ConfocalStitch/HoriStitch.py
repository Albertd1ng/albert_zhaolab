import numpy as np
import cv2
import nd2
import os
from multiprocessing import Value, Array, RLock, Pool
from readlif.reader import LifFile

from InfoIO import get_img_xml_info, get_img_nd2_info, get_img_lif_info
from FileRename import rename_file
from TileCont import judge_tile_cont
from ImgIO import get_img_from_nd2, get_img_from_lif, import_img_one_tile
from ParamEsti import pyr_down_time_esti
from ImgProcess import pyr_down_img
from ImgBorder import get_2img_border, get_border_pyr_down
from ImgOvl import get_ovl_img
from LossFunc import loss_func_for_list
from AxisRange import calc_axis_range, find_first_last_index


def get_stitch_result(res_list, tile_num, voxel_len):
    tile_shift_arr = np.zeros((tile_num, tile_num, 3), dtype='float64')
    tile_shift_loss = np.inf * np.ones((tile_num, tile_num), dtype='float64')
    for k in res_list:
        i, j = k[0], k[1]
        res = k[2].get()
        tile_shift_arr[i, j, :] = np.array(res[0], dtype='float64') * voxel_len
        tile_shift_arr[j, i, :] = -tile_shift_arr[i, j, :]
        tile_shift_loss[i, j] = res[1]
        tile_shift_loss[j, i] = tile_shift_loss[i, j]
    return tile_shift_arr, tile_shift_loss


def update_strip_pos_by_MST(tile_pos, tile_shift_arr, tile_shift_loss):
    tile_num = tile_pos.shape[0]
    if_tile_stitched = np.zeros(tile_num, dtype='bool')
    if_tile_stitched[0] = True
    tile_refer_id = -np.ones(tile_num, dtype='int64')
    while False in if_tile_stitched:
        loss_min = np.inf
        stitch_id = -1
        refer_id = -1
        for i in range(tile_num):
            if if_tile_stitched[i]:
                continue
            for j in range(tile_num):
                if not if_tile_stitched[j]:
                    continue
                if tile_shift_loss[i, j] < loss_min:
                    loss_min = tile_shift_loss
                    stitch_id = i
                    refer_id = j
        if stitch_id == -1:
            break
        if_tile_stitched[stitch_id] = True
        tile_refer_id[stitch_id] = refer_id

    def change_with_child(father_id, shift_vec):
        tile_pos_update[father_id, :] = tile_pos_update[father_id, :] + shift_vec
        for [child_id] in np.agrwhere(tile_refer_id == father_id):
            change_with_child(child_id, shift_vec)
    tile_pos_update = tile_pos.copy()
    while np.any(tile_refer_id != -1):
        for i in range(tile_num):
            for [j] in np.argwhere(tile_refer_id == -1):
                for [k] in np.argwhere(tile_refer_id == j):
                    change_with_child(k, tile_shift_arr[k, j, :])
                    tile_refer_id[k] = -1
    return tile_pos_update


def one_stitch(img1, img2, tile_pos, dim_elem_num, dim_len, voxel_len, voxel_range,
               if_sparce, pyr_down_times, blur_kernel_size):
    img1 = cv2.medianBlur(img1, blur_kernel_size)
    img2 = cv2.medianBlur(img2, blur_kernel_size)
    for pdt in range(pyr_down_times, -1, -1):
        if pdt == pyr_down_times:
            x_s, y_s, z_s = 0, 0, 0
            x_pd, y_pd, z_pd = np.int64(np.round(np.array(voxel_range, dtype='int64') / (2 ** pdt)))
        else:
            x_s, y_s, z_s = 2 * np.array((x_s, y_s, z_s), dtype='int64')
            x_pd, y_pd, z_pd = 3, 3, 3
        loss_min = np.inf
        if pdt != 0:
            img1_pd = pyr_down_img(img1, pdt)
            img2_pd = pyr_down_img(img2, pdt)
            down_multi = np.array([img1_pd.shape[1], img1_pd.shape[0], img1_pd.shape[2]], dtype='float32') / np.array(
                [img1.shape[1], img1.shape[0], img1.shape[2]], dtype='float32')
        else:
            img1_pd = img1.copy()
            img2_pd = img2.copy()
        x_sr, y_sr, z_sr = x_s, y_s, z_s
        for x in range(x_sr - x_pd, x_sr + x_pd + 1):
            for y in range(y_sr - y_pd, y_sr + y_pd + 1):
                for z in range(z_sr - z_pd, z_sr + z_pd + 1):
                    this_tile_pos = tile_pos.copy()
                    this_tile_pos[0, :] = this_tile_pos[0, :] + voxel_len * np.array([x, y, z], dtype='float64') * (
                                2 ** pdt)
                    border = get_2img_border(dim_elem_num, dim_len, voxel_len, this_tile_pos)
                    if pdt != 0:
                        border = get_border_pyr_down(border, down_multi)
                    if 0 in border.shape:
                        continue
                    ovl1_list, ovl2_list = get_ovl_img(img1_pd, img2_pd, border, if_sparce)
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list)
                    if this_loss < loss_min:
                        loss_min = this_loss
                        x_s, y_s, z_s = x, y, z
                    #
                    # if x==x_sr and y==y_sr and z==z_sr and pdt==0:
                    #     for ovl1,ovl2 in zip(ovl1_list, ovl2_list):
                    #         cv2.imshow('1', np.clip(ovl1*7,0,255))
                    #         cv2.imshow('2', np.clip(ovl2*7,0,255))
                    #         cv2.waitKey(0)
                    #         cv2.destroyAllWindows()
                    #
    print([x_s, y_s, z_s], loss_min)
    return [x_s, y_s, z_s], loss_min


def start_multi_stitch(layer_num, info_IO_path, img_file_type, img_path, img_name_format, img_name,
                       ch_num, ch_th, img_data_type, move_ratio, if_sparce, if_high_noise, if_rename_file, pro_num=10):
    # info input
    if img_file_type == 'nd2':
        whole_img = nd2.ND2File(img_path)
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos, dim_num, img_data_type = get_img_nd2_info(whole_img)
        whole_img.close()
        whole_img = nd2.imread(img_path)
    elif img_file_type == 'lif':
        whole_img = LifFile(img_path)
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_lif_info(whole_img)
        whole_img = whole_img.get_image(0)
    elif img_file_type == 'tif':
        info_file_path = os.path.join(img_path, "MetaData", img_name+'.xml')
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_xml_info(info_file_path)
        if if_rename_file:
            rename_file(img_name_format, img_path, img_name, dim_elem_num[2], ch_num, img_file_type)
        if img_data_type == 'uint8':
            img_mode = cv2.IMREAD_GRAYSCALE
        elif img_data_type == 'uint16':
            img_mode = cv2.IMREAD_UNCHANGED
        else:
            print('Input Error')
            return
    else:
        print('Input Error')
        return

    # tile contact
    tile_contact = judge_tile_cont(dim_len, tile_pos)

    # info save
    np.save(os.path.join(info_IO_path, 'dim_elem_num_%.4d.npy' % layer_num), dim_elem_num)
    np.save(os.path.join(info_IO_path, 'dim_len_%.4d.npy' % layer_num), dim_len)
    np.save(os.path.join(info_IO_path, 'voxel_len.npy'), voxel_len)
    np.save(os.path.join(info_IO_path, 'tile_num_%.4d.npy' % layer_num), tile_num)
    np.save(os.path.join(info_IO_path, 'tile_pos_%.4d.npy' % layer_num), tile_pos)
    np.save(os.path.join(info_IO_path, 'tile_contact_%.4d.npy' % layer_num), tile_contact)

    # info prepare
    voxel_range = np.int64(np.ceil(dim_elem_num * move_ratio))
    pyr_down_times = pyr_down_time_esti(dim_elem_num[:2], v_thred=200*200)
    if if_high_noise:
        blur_kernel_size = 5
    else:
        blur_kernel_size = 3

    # start multi stitch
    pool = Pool(processes=pro_num)
    res_list = []
    for i in range(tile_num):
        for j in range(tile_num):
            if i >= j:
                continue
            if tile_contact[i, j]:
                if img_file_type == 'nd2':
                    img1 = get_img_from_nd2(whole_img, i, ch_num, ch_th)
                    img2 = get_img_from_nd2(whole_img, j, ch_num, ch_th)
                elif img_file_type == 'lif':
                    img1 = get_img_from_lif(whole_img, i, ch_th, dim_elem_num)
                    img2 = get_img_from_lif(whole_img, j, ch_th, dim_elem_num)
                elif img_file_type == 'tif':
                    img1 = import_img_one_tile(img_name_format, img_path, img_name, i, ch_th, dim_elem_num,
                                               img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                    img2 = import_img_one_tile(img_name_format, img_path, img_name, j, ch_th, dim_elem_num,
                                               img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                res_list.append([i, j, pool.apply_async(one_stitch, args=(
                    img1, img2, tile_pos[[i, j], :], dim_elem_num, dim_len, voxel_len, voxel_range,
                    if_sparce, pyr_down_times, blur_kernel_size))])
    pool.close()
    pool.join()
    whole_img = np.array([])

    # update pos
    # tile_shift_arr[i, j], i相对于j的位移值
    tile_shift_arr, tile_shift_loss = get_stitch_result(res_list, tile_num, voxel_len)
    tile_pos_stitch = update_strip_pos_by_MST(tile_pos, tile_shift_arr, tile_shift_loss)

    # calculate new info
    axis_range, voxel_num = calc_axis_range(tile_pos_stitch, dim_elem_num, voxel_len)
    first_last_index = find_first_last_index(tile_pos_stitch, dim_elem_num, axis_range, voxel_len, voxel_num)

    # save data
    np.save(os.path.join(info_IO_path, 'tile_pos_stitch_%.4d.npy' % layer_num), tile_pos_stitch)
    np.save(os.path.join(info_IO_path, 'axis_range_stitch_%.4d.npy' % layer_num), axis_range)
    np.save(os.path.join(info_IO_path, 'first_last_index_%.4d.npy' % layer_num), first_last_index)

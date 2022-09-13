from InfoIO import get_img_npy_basic_info,get_img_info_hori_stit,get_img_info_vert_stit,save_img_info_vert_stit,get_img_xml_info_Z_stit
from ImgIO import import_whole_img
from ImgProcess import pyr_down_img,adjust_contrast
from LossFunc import loss_func_z_stitch

from lxml import etree
import numpy as np
import time
import cv2
import os
import random

def calc_sift_points_match(img1,img2):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kpts1, des1 = sift.detectAndCompute(img1, None)
    kpts2, des2 = sift.detectAndCompute(img2, None)
    kp1, kp2 = np.float32([kp.pt for kp in kpts1]), np.float32([kp.pt for kp in kpts2])
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            pts1, pts2 = np.float32([kp1[i, :] for (i, _) in good_matches]), np.float32([kp2[j, :] for (_, j) in good_matches])
    return pts1,pts2

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
        this_loss = loss_func_z_stitch(ovl1, ovl2)
        # print(xy_shift,this_loss)
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
    # print(xy_shift_min,loss_min)
    return xy_shift_min, loss_min


def calc_xy_shift_by_BF(img1_down, img2_down, img1, img2, xy_shift,pyr_down_times,if_median_blur,blur_kernel_size):
    if if_median_blur:
        img1,img2=cv2.medianBlur(img1,blur_kernel_size),cv2.medianBlur(img2,blur_kernel_size)
    img1, img2 = adjust_contrast(img1, img2)
    xy_shift_min = np.zeros(2)
    for i in range(pyr_down_times,-1,-1):
        if i==pyr_down_times:
            img1_calc,img2_calc=img1_down,img2_down
        elif i==0:
            img1_calc,img2_calc=img1,img2
        else:
            img1_calc,img2_calc=pyr_down_img(img1,i),pyr_down_img(img1,i)
        if i==pyr_down_times:
            loss_min=np.inf
            range_calc = 10
        else:
            xy_shift=xy_shift_min*2
            xy_shift_min = np.zeros(0)
            loss_min=np.inf
            range_calc = 5
        for x in range(-range_calc, range_calc+1):
            for y in range(-range_calc, range_calc+1):
                this_xy_shift = xy_shift + np.array([x, y], dtype='int32')
                ovl1 = img1_calc[np.max((0, -this_xy_shift[1])):, np.max((0, -this_xy_shift[0])):]
                ovl2 = img2_calc[np.max((0, this_xy_shift[1])):, np.max((0, this_xy_shift[0])):]
                x_range, y_range = np.min((ovl1.shape[1], ovl2.shape[1],5000)), np.min((ovl1.shape[0], ovl2.shape[0],5000))
                ovl1, ovl2 = ovl1[0:y_range, 0:x_range], ovl2[0:y_range, 0:x_range]
                this_loss = loss_func_z_stitch(ovl1, ovl2)
                if this_loss < loss_min:
                    loss_min = this_loss
                    xy_shift_min = this_xy_shift
        print('i pyr down times', xy_shift_min, loss_min)
    return xy_shift_min, loss_min


def update_lower_layer_info(xy_shift, tile_pos, axis_range1, axis_range2, dim_len, voxel_len):
    tile_pos[:, 0] = tile_pos[:, 0] - axis_range2[0, 0] + axis_range1[0, 0] - xy_shift[0] * voxel_len[0]
    tile_pos[:, 1] = tile_pos[:, 1] - axis_range2[1, 0] + axis_range1[1, 0] - xy_shift[1] * voxel_len[1]
    axis_range2[0, 0], axis_range2[0, 1] = np.min(tile_pos[:, 0]), np.max(tile_pos[:, 0]) + dim_len[0]
    axis_range2[1, 0], axis_range2[1, 1] = np.min(tile_pos[:, 1]), np.max(tile_pos[:, 1]) + dim_len[1]
    voxel_num = np.uint64(np.round((axis_range2[:, 1] - axis_range2[:, 0]) / voxel_len))
    return tile_pos, axis_range2, voxel_num


def start_vertical_stitch(i, img_path_format,img_path, info_IO_path, file_name_format, img_name, channel_ordinal, img_type,
                          img_data_type,overlap_ratio,vert_step,pyr_down_times,
                          if_median_blur,blur_kernel_size):
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    if i == 0:
        tile_pos1,axis_range1,index1 = get_img_info_hori_stit(i,info_IO_path)
        save_img_info_vert_stit(i,info_IO_path,tile_pos1,axis_range1,index1)
    img_path1, img_path2 = os.path.join(img_path,file_name_format % (i)),os.path.join(img_path,file_name_format % (i + 1))
    dim_elem_num1, dim_len1, voxel_len = get_img_npy_basic_info(i, info_IO_path)
    dim_elem_num2, dim_len2, voxel_len = get_img_npy_basic_info(i + 1, info_IO_path)
    tile_pos1, axis_range1, index1 = get_img_info_vert_stit(i,info_IO_path)
    tile_pos2, axis_range2, index2 = get_img_info_hori_stit(i+1,info_IO_path)
    voxel_num1, voxel_num2 = np.uint64(np.round((axis_range1[:, 1] - axis_range1[:, 0]) / voxel_len)), np.uint64(
        np.round((axis_range2[:, 1] - axis_range2[:, 0]) / voxel_len))
    index1, index2 = index1[1], index2[0]
    tile_num1, tile_num2 = tile_pos1.shape[0], tile_pos2.shape[0]
    img1 = import_whole_img(index1, img_path_format, img_path1,img_name,channel_ordinal,axis_range1,dim_elem_num1,voxel_num1,
                            voxel_len,tile_pos1,img_type=img_type,img_data_type=img_data_type,img_mode=img_mode)

    loss_min = np.inf
    index_min = -1
    xy_shift_min = np.zeros(0)
    for j in range(index2, np.int32(np.round(voxel_num2[2] * overlap_ratio) + index2), vert_step):
        img2 = import_whole_img(j, img_path_format,img_path2, img_name, channel_ordinal,axis_range2, dim_elem_num2, voxel_num2,
                                voxel_len, tile_pos2,img_type=img_type,img_data_type=img_data_type,img_mode=img_mode)
        img2_down = pyr_down_img(img2, pyr_down_times)
        img1_down = pyr_down_img(img1, pyr_down_times)
        img1_down, img2_down = adjust_contrast(img1_down, img2_down)
        pts1,pts2=calc_sift_points_match(img1_down, img2_down)
        if pts1.shape[0]==0:
            continue
        xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(),img2.copy(),xy_shift,pyr_down_times,
                                                  if_median_blur,blur_kernel_size)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
            index_min = j

    for j in range(np.max((index_min - vert_step-1, index2)), index_min + vert_step):
        img2 = import_whole_img(j, img_path_format, img_path2, img_name, channel_ordinal, axis_range2, dim_elem_num2,voxel_num2,
                                voxel_len, tile_pos2, img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
        img2_down = pyr_down_img(img2, pyr_down_times)
        img1_down = pyr_down_img(img1, pyr_down_times)
        img1_down, img2_down = adjust_contrast(img1_down, img2_down)
        pts1, pts2 = calc_sift_points_match(img1_down, img2_down)
        if pts1.shape[0]==0:
            continue
        xy_shift, this_loss = calc_xy_shift_by_RANSAC(img1_down, img2_down, pts1, pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        xy_shift, this_loss = calc_xy_shift_by_BF(img1_down, img2_down, img1.copy(),img2.copy(),xy_shift,pyr_down_times,
                                                  if_median_blur,blur_kernel_size)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f' % (j, str(xy_shift), this_loss))
        if this_loss < loss_min:
            loss_min = this_loss
            xy_shift_min = xy_shift
            index_min = j
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Finally the matched one is %d.th layer, xy_shift is %s, loss is %.8f' % (
    index_min, str(xy_shift_min), loss_min))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    index2 = np.load(info_IO_path + r'\first_last_index_%.4d.npy' % (i + 1))
    index2[0] = index_min
    tile_pos2, axis_range2, voxel_num2 = update_lower_layer_info(xy_shift, tile_pos2, axis_range1, axis_range2,
                                                                 dim_len2, voxel_len)
    print('start saving data')
    save_img_info_vert_stit(i+1,info_IO_path,tile_pos2,axis_range2,index2)
    print('end saving data')
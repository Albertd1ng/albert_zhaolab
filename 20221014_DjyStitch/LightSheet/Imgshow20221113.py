import numpy as np
import os
import cv2
import ctypes
from merge import MergeSolution

from ImgIO import import_img_whole_2D_XY_merge
from InfoIO import get_layer_id_dict_info, get_img_dim_info, get_strip_pos_info, get_id_dir_dict_info
from LayerStitchSIFT import choose_layer_index


if __name__ == '__main__':
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_xyz_stitch_z003000.txt'
    dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/src_frames_ds331'
    img_save_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/20221113/z007000_z010000'
    img_name_format = r'%s_%s'
    channel_num = 2
    img_type = 'tif'

    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    z_v_num[[165, 166, 167, 168, 169, 170, 171, 172, 173]] = 8000  # 165-173
    z_min, z_max = np.min(strip_pos[:, 2]), np.max(strip_pos[:, 2] + z_v_num)

    # for z in range(z_min, int(round(z_max/4))):
    for z in range(0, 1000):
        # if z % 2 == 1:
        #     continue
        # this_z = int(z / 2)
        img1 = import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z, 0, strip_pos, xy_v_num, z_v_num)
        cv2.imwrite(os.path.join(img_save_path, '%.6d_%.2d.tif' % (z, 0)), img1)
        img2 = import_img_whole_2D_XY_merge(img_name_format, dir_path, id_dir_dict, z, 1, strip_pos, xy_v_num, z_v_num)
        cv2.imwrite(os.path.join(img_save_path, '%.6d_%.2d.tif' % (z, 1)), img2)

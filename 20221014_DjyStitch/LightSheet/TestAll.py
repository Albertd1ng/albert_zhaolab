import os
import numpy as np
import time
import cv2
import ctypes
import random
import re
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_img_txt_info, get_img_dim_info, save_img_txt_info, save_layer_id_dict_info, get_id_dir_dict_info
from InfoIO import get_strip_pos_info,get_layer_id_dict_info
from ImgIO import import_img_3D_section
from StripCont import judge_strip_cont
from ImgBorder import get_two_strip_border, get_border_pyr_down
from ImgProcess import pyr_down_img, adjust_contrast
from ImgOvl import get_ovl_img
from LossFunc import loss_func_for_list
from StripStitch import in_which_layer


def updown_z(dir_path, img_path, mid_img_name_format, img_name_format, channel_num=2):
    img_name_list=os.listdir(os.path.join(dir_path,img_path))
    z_num = round(len(img_name_list) / channel_num)
    for one_name in img_name_list:
        one_name_list = re.split('_', re.sub('.tif', '', one_name))
        this_z, this_ch = int(one_name_list[0]), int(one_name_list[1])
        new_this_z = z_num - this_z - 1
        old_name = os.path.join(dir_path, img_path, one_name)
        new_name = os.path.join(dir_path, img_path, mid_img_name_format % (new_this_z, this_ch))
        os.rename(old_name, new_name)
    time.sleep(10)
    img_name_list = os.listdir(os.path.join(dir_path, img_path))
    for one_name in img_name_list:
        one_name_list = re.split('_', re.sub('.tif', '', one_name))
        this_z, this_ch = int(one_name_list[0]), int(one_name_list[1])
        old_name = os.path.join(dir_path, img_path, one_name)
        new_name = os.path.join(dir_path, img_path, img_name_format % (this_z, this_ch))
        os.rename(old_name, new_name)


if  __name__=='__main__':
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_stitch.txt'
    dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/src_frames_ds331'
    img_name_format = r'%s_%s'
    img_type = 'tif'
    channel_num = 2

    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    np.save(os.path.join(os.path.split(txt_path)[0], 'xy_v_num.npy'), xy_v_num)
    np.save(os.path.join(os.path.split(txt_path)[0], 'z_v_num.npy'), z_v_num)
    

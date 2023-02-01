import os
import numpy as np
import time
import cv2
import ctypes
import random
from multiprocessing import Value, Array, Process, RLock, cpu_count, current_process

from InfoIO import get_img_txt_info, get_img_dim_info, save_img_txt_info, save_layer_id_dict_info
from ImgIO import import_img_3D_section
from StripCont import judge_strip_cont
from ImgBorder import get_two_strip_border, get_border_pyr_down
from ImgProcess import pyr_down_img, adjust_contrast
from ImgOvl import get_ovl_img
from LossFunc import loss_func_for_list


txt_path = r'C:\Users\dingj\Desktop\Software\pos.txt'
save_txt_path = r'C:\Users\dingj\Desktop\Software\pos_stitch.txt'
id_dir_dict, strip_pos, layer_id_dict = get_img_txt_info(txt_path)
save_layer_id_dict_info(os.path.join(os.path.split(save_txt_path)[0],'layer_id_dict.txt'), layer_id_dict)
xy_v_num = np.array([1024, 2048], dtype='int64')
strip_cont = judge_strip_cont(strip_pos, xy_v_num)
for i in range(strip_cont.shape[0]):
    print(np.any(strip_cont[i,:]))
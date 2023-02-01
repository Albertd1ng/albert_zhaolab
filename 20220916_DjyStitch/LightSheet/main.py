import os
import numpy as np

from StripStitch import start_multi_strip_stitches
from LayerStitch import layer_stitch


if __name__ == '__main__':
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down.txt'
    save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_stitch.txt'
    dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/src_frames_ds331'
    img_name_format = r'%s_%s'
    channel_num = 2
    ch_th = 0
    img_type = 'tif'
    voxel_range = [30, 60, 0]
    pyr_down_times = 1
    choose_num = 10
    z_depth = 100
    start_multi_strip_stitches(txt_path, save_txt_path, dir_path, img_name_format, channel_num, ch_th, img_type,
                               voxel_range, pyr_down_times, choose_num, z_depth)

    # ----------------------------------------------------------------------------------------------------------
    # txt_path = r'/home/zhaohu_lab/dingjiayi/pos_stitch.txt'
    # save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_xyz_stitch.txt'
    # dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/220908/frames'
    # img_name_format = r'%s_%s'
    # channel_num = 2
    # ch_th = 0
    # img_type = 'tif'
    # voxel_range = [100, 100, 100]
    # pyr_down_times = 2
    # choose_num = 20
    # z_depth = 200
    # layer_stitch(txt_path, save_txt_path, dir_path, img_name_format, channel_num, ch_th, img_type,
    #              voxel_range, pyr_down_times, choose_num, z_depth)

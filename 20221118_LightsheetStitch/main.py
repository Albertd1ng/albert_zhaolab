import os
import numpy as np

from StripStitch import start_multi_strip_stitches
from LayerStitchData import layer_stitch_data_IO, layer_sec_stitch_data_IO
from LayerStitchSIFT import start_layer_stitch_sift
from LayerStitch import start_multi_layer_stitch
from LayerSecStitchSIFT import start_layer_sec_stitch_sift
from InfoIO import trans_pos_info

if __name__ == '__main__':
    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down.txt'
    save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_stitch.txt'
    dir_path = r'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/src_frames_ds331'
    running_IO_path = r'/home/zhaohu_lab/dingjiayi/20221024.txt'
    img_name_format = r'%s_%s'
    channel_num = 2
    ch_th = 0
    img_type = 'tif'
    voxel_range = [50, 100, 0]
    pyr_down_times = 1
    choose_num = 10
    z_depth = 100
    start_multi_strip_stitches(txt_path, save_txt_path, dir_path, running_IO_path, img_name_format, channel_num,
                               ch_th, img_type, voxel_range, pyr_down_times, choose_num, z_depth)

    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_stitch.txt'
    data_IO_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/img331_yz_npy'
    sam_times = 10
    x_step = 30
    layer_stitch_data_IO(txt_path, data_IO_path, dir_path, img_name_format, channel_num, ch_th, img_type=img_type,
                         sam_times=sam_times, x_step=x_step)

    pyr_down_times = 4
    save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_yz_stitch.txt'
    running_IO_path = r'/home/zhaohu_lab/dingjiayi/20221023.txt'
    start_layer_stitch_sift(txt_path, save_txt_path, dir_path, data_IO_path, running_IO_path, img_name_format,
                            channel_num, ch_th, img_type, sam_times=sam_times, x_step=x_step, pyr_down_times=pyr_down_times)

    txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_yz_stitch_20221101.txt'
    data_IO_path = r'/GPFS/zhaohu_lab_permanent/dingjiayi/img331_xyz_npy'
    y_range = [3000, 7000]
    z_range = [3000, 3500]
    layer_sec_stitch_data_IO(txt_path, data_IO_path, dir_path, img_name_format, channel_num, ch_th, img_type=img_type,
                             y_range=y_range, z_range=z_range)

    save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_down_xyz_stitch_z%.6d.txt' % 3000
    running_IO_path = r'/home/zhaohu_lab/dingjiayi/20221101.txt'
    step = 5
    pyr_down_times = 1
    start_layer_sec_stitch_sift(txt_path, save_txt_path, dir_path, data_IO_path, running_IO_path, img_name_format,
                                channel_num, ch_th, img_type=img_type, y_range=y_range, z_range=z_range, step=step,
                                pyr_down_times=pyr_down_times)

    txt_path = save_txt_path
    save_txt_path = r'/home/zhaohu_lab/dingjiayi/pos_20221103.txt'
    trans_pos_info(txt_path, save_txt_path, [3, 3, 1])

import numpy as np
import os

from InfoIO import get_strip_pos_info, get_id_dir_dict_info, get_layer_id_dict_info, get_img_dim_info
from ImgIO import import_img_slab_2D_max_YZ, import_img_slab_sec_2D_YZ


def layer_stitch_data_IO(txt_path, data_IO_path, dir_path, img_name_format, channel_num, ch_th, img_type,
                         sam_times=10, x_step=30):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format,
                                                    img_type=img_type)
    layer_num = len(layer_id_dict)
    for i in range(layer_num-1):
        img1_YZ_arr, img1_pos_arr = import_img_slab_2D_max_YZ(img_name_format, dir_path, id_dir_dict,
                                                                layer_id_dict[i], ch_th, strip_pos, xy_v_num, z_v_num,
                                                                direct='rl', sam_times=sam_times, x_step=x_step,
                                                                img_type=img_type, img_dtype=img_dtype)
        np.save(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s.npy' % (i, sam_times, x_step, 'rl')), img1_YZ_arr)
        np.save(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s_pos.npy' % (i, sam_times, x_step, 'rl')), img1_pos_arr)
        print('slab%.2d_sam%.2d_step%.2d_%s.npy, saved' % (i, sam_times, x_step, 'rl'))
        img2_YZ_arr, img2_pos_arr = import_img_slab_2D_max_YZ(img_name_format, dir_path, id_dir_dict,
                                                                layer_id_dict[i+1], ch_th, strip_pos, xy_v_num,z_v_num,
                                                                direct='lr', sam_times=sam_times, x_step=x_step,
                                                                img_type=img_type, img_dtype=img_dtype)
        np.save(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s.npy' % (i + 1, sam_times, x_step, 'lr')), img2_YZ_arr)
        np.save(os.path.join(data_IO_path, 'slab%.2d_sam%.2d_step%.2d_%s_pos.npy' % (i + 1, sam_times, x_step, 'lr')), img2_pos_arr)
        print('slab%.2d_sam%.2d_step%.2d_%s.npy, saved' % (i + 1, sam_times, x_step, 'lr'))


def layer_sec_stitch_data_IO(txt_path, data_IO_path, dir_path, img_name_format, channel_num, ch_th, img_type, y_range, z_range):
    strip_pos = get_strip_pos_info(txt_path)
    id_dir_dict = get_id_dir_dict_info(os.path.join(os.path.split(txt_path)[0], 'id_dir_dict.txt'))
    layer_id_dict = get_layer_id_dict_info(os.path.join(os.path.split(txt_path)[0], 'layer_id_dict.txt'))
    xy_v_num, z_v_num, img_dtype = get_img_dim_info(dir_path, id_dir_dict, channel_num, img_name_format, img_type=img_type)
    layer_num = len(layer_id_dict)
    for i in range(layer_num):
        img_YZ_arr, img_pos_arr = import_img_slab_sec_2D_YZ(img_name_format, dir_path, id_dir_dict, layer_id_dict[i],
                                                            ch_th, strip_pos, xy_v_num, z_v_num, y_range=y_range,
                                                            z_range=z_range, img_type=img_type, img_dtype=img_dtype)
        np.save(os.path.join(data_IO_path, 'slab%.2d_y%.6d-%.6d_z%.6d-%.6d.npy' % (i,
                y_range[0], y_range[1], z_range[0], z_range[1])), img_YZ_arr)
        np.save(os.path.join(data_IO_path, 'slab%.2d_y%.6d-%.6d_z%.6d-%.6d_pos.npy' % (i,
                y_range[0], y_range[1], z_range[0], z_range[1])), img_pos_arr)

import numpy as np
import os

from InfoIO import get_img_npy_basic_info,get_img_info_hori_stit
from LossFunc import loss_func

# info_IO_path=r'D:\Albert\Data\ZhaoLab\Image\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G\MetaData'
# a=np.load(os.path.join(info_IO_path,'tile_pos_stitch_0000.npy'))
# dim_elem_num, dim_len, voxel_len = get_img_npy_basic_info(0, info_IO_path)
# tile_pos, axis_range, first_last_index = get_img_info_hori_stit(0, info_IO_path)
# voxel_num = np.int64(np.round((axis_range[:, 1] - axis_range[:, 0]) / voxel_len))
# print(voxel_num)

print(loss_func(np.array([]),np.array([]),[0,0,0],0))
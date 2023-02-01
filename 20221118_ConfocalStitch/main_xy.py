import os
from HoriStitch import start_multi_stitch
from ImgIO import export_img_hori_stit

# 必填项
# 文件数据类型，[nd2, lif, tif...]
img_file_type = 'tif'
# 文件路径，如果是tif等图片格式是文件夹，如果是nd2、lif等特殊格式则具体到文件名
img_path = r'D:\Albert\Data\ZhaoLab\Image\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
# 输出文件路径
img_save_path = r'D:\Albert\Data\ZhaoLab\Image\S'
# 通道数
ch_num = 1
# 用第几个通道拼接
ch_th = 0
# 图片bit数，['uint8', 'uint16']
img_data_type = 'uint8'

# 选填项
# 中间文件路径，默认是拼接文件路径的上一级
info_IO_path = os.path.split(img_path)[0]
# 图片计算范围
move_ratio = [0.1, 0.1, 0.05]
# 第几层数据，默认为0，多层拼接时选填
layer_num = 0
# 是否为稀疏数据
if_sparce = False
# 是否为高噪点数据
if_high_noise = False
# 最大进程数，根据内存大小和文件大小决定
pro_num = 5

# 如果是tif图片格式数据则选填下面选项
# 文件命名格式
img_name_format = r'%s_t%.4d_z%.4d_ch%.2d.%s'  # %(img_name, tile_th, z, ch_th, img_type)
# 图片名
img_name = 'Region'
# 是否重命名文件
if_rename_file = True


if __name__ == '__main__':
    start_multi_stitch(layer_num, info_IO_path, img_file_type, img_path, img_name_format, img_name, ch_num, ch_th,
                       img_data_type, move_ratio, if_sparce, if_high_noise, if_rename_file, pro_num=pro_num)
    export_img_hori_stit(layer_num, info_IO_path, img_path, img_save_path, img_name_format, img_name, ch_num,
                         img_file_type, img_data_type, img_save_type='tif')

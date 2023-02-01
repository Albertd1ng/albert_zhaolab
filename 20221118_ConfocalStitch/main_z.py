from VertStitchMerged import start_vertical_stit_merged
from ImgIO import export_img_vert_stit_merged

# 文件路径
file_path = r'C:\Users\User\Desktop\lmz_stitch'
# 图片文档命名格式
file_name_format = r'%.4d'
# 拼接后的图片保存路径
img_save_path = r'C:\Users\User\Desktop\lmz_stitch\S'
# 需要拼接的层数
layer_num = 2
# 通道数
channel_num = 2
# 基于第几个通道进行拼接，假如有3个通道，编号为0，1，2
channel_ordinal = 1

# 可选参数，一般情况下默认即可
# 图片类型
img_file_type = 'tif'  # {'tif', 'tiff',...}
# 图片数据类型
img_data_type = 'uint8'  # {'uint8', 'uint16'}
# 计算时考虑的重叠比率
overlap_ratio = 0.2  # z
# 计算时的像素步长
step = 5
# 是否重命名文件，如果选否，认真填写图片命名格式img_name_format和图片名
if_rename_file = False
# 图片命名格式
img_name_format = '%s_z%.4d_ch%.2d.%s'  # %(img_name,z,ch_th,img_type)
# 图片名
img_name = 'Region'
# 中间文件储存路径
info_IO_path = file_path
# 拼接时是否要进行中值降噪
if_median_blur = True
# 中值降噪的卷积核大小
blur_kernel_size = 3
# 垂直拼接时XY维度的降采样次数，取值为2的幂次方，建议降采样至图片大小小于800*800. 也可选-1，自动估计
pyr_down_times = -1


if __name__ == '__main__':
    for i in range(0, layer_num - 1):
        start_vertical_stit_merged(i, img_name_format, file_path, info_IO_path, file_name_format, img_name, channel_num,
                                   channel_ordinal, img_file_type, img_data_type, overlap_ratio=overlap_ratio,
                                   vert_step=step, pyr_down_times=pyr_down_times, if_median_blur=if_median_blur,
                                   blur_kernel_size=blur_kernel_size, if_rename_file=if_rename_file)
    export_img_vert_stit_merged(layer_num, info_IO_path, file_path, file_name_format, img_save_path, img_name_format,
                                img_name, channel_num, img_file_type, img_data_type, 0)

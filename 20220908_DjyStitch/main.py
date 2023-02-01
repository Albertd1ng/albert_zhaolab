from HoriStitch import start_multi_stitchers
from VertStitch import start_vertical_stitch,start_vertical_stit_merged
from FileRename import rename_file,rename_file_tile_id,rename_file_Z_stit
from ImgIO import export_img_vert_stit,export_img_hori_stit,export_img_vert_stit_merged

import os
import numpy as np
from multiprocessing import Process

#python ./20220908_DjyStitch/main.py
#选择拼接模式
stitch_mode='XY' #{‘XYZ’， ‘XY’， ‘Z'}

##################################################
if stitch_mode=='XYZ':
    #必要参数
    #文件路径
    file_path=r'D:\Albert\Data\ZhaoLab\Imaging\新建文件夹'
    #图片文档命名格式
    file_name_format=r'S%.4d'
    #拼接后的图片保存路径
    img_save_path = r'D:\Albert\Data\ZhaoLab\Imaging\新建文件夹\S'
    #需要拼接的层数
    layer_num=5
    #通道数
    channel_num=3
    #基于第几个通道进行拼接，假如有3个通道，编号为0，1，2
    channel_ordinal=0

    #可选参数
    #一般情况下默认即可
    # 图片类型
    img_file_type = 'tif'  # {'tif', 'tiff',...}
    # 图片数据类型
    img_data_type = 'uint8'  # {'uint8', 'uint16'}
    # 图像维度和位置信息文件类型。要放在图片给文档的MetaData文件夹中，与图片名一致
    info_file_type = 'xml'  # {'xml', 'txt',...}
    # 计算时考虑的重叠比率
    overlap_ratio_hori = [0.2, 0.2, 0.05]  # [x,y,z]
    overlap_ratio_vert = 0.3  # z
    # 计算时的像素步长
    hori_step = 5
    vert_step = 5
    #是否重命名文件，如果选否，认真填写图片命名格式img_name_format和图片名
    if_rename_file = True
    # 图片命名格式
    img_name_format='%s\%s_t%.4d_z%.4d_ch%.2d.%s'#%(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    # 图片名
    img_name = 'Region'
    #中间文件储存路径
    info_IO_path=file_path
    #α超参数，防止被拼接的图像超出实际正常位置过远，低噪点数据建议取10e-7，高噪点数据建议取10e-5，也可以取0
    alpha=10e-7
    #拼接时是否要进行中值降噪
    if_median_blur=True
    #中值降噪的卷积核大小
    blur_kernel_size=3
    #设定拼接损失函数阈值，如果损失值超出阈值，则该块图像重新拼接,低噪点数据建议取0.2-0.5，高噪点数据建议取0.5-1.5
    bad_tile_threshold=0.5
    threshold_step=0.1 #一般取bad_tile_threshold的20%
    #一个块的最大计算次数
    max_calc_times=2
    #垂直拼接时XY维度的降采样次数，取值为2的幂次方，建议降采样至图片大小小于800*800
    pyr_down_times=4

##################################################
elif stitch_mode=='XY':
    # 图片路径
    file_path = r'D:\Albert\Data\ZhaoLab\Image\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
    # 拼接后的图片保存路径
    img_save_path = r'D:\Albert\Data\ZhaoLab\Image\stitched'
    # 通道数
    channel_num = 1
    # 基于第几个通道进行拼接，假如有3个通道，编号为0，1，2
    channel_ordinal = 0

    # 可选参数
    # 一般情况下默认即可
    # 图片类型
    img_file_type = 'tif'  # {'tif', 'tiff',...}
    # 图片数据类型
    img_data_type = 'uint8'  # {'uint8', 'uint16'}
    # 计算时考虑的重叠比率
    overlap_ratio = [0.08, 0.08, 0.06]  # [x,y,z]
    # 计算时的像素步长
    step = 3
    # 图像维度和位置信息文件类型。要放在图片给文档的MetaData文件夹中，与图片名一致
    info_file_type = 'xml'  # {'xml', 'txt',...}
    # 是否重命名文件，如果选否，认真填写图片命名格式img_name_format和图片名
    if_rename_file = True
    # 图片命名格式
    img_name_format = '%s\%s_t%.4d_z%.4d_ch%.2d.%s'  # %(img_path,img_name,tile_ordinal,z_dimension_ordinal,channel_ordinal,img_type)
    # 图片名
    img_name = 'Region'
    # 中间文件储存路径
    info_IO_path = file_path+r'\MetaData'
    # α超参数，防止被拼接的图像超出实际正常位置过远，低噪点数据建议取10e-7，高噪点数据建议取10e-5，也可以取0
    alpha = 10e-7
    # 拼接时是否要进行中值降噪
    if_median_blur = True
    # 中值降噪的卷积核大小
    blur_kernel_size = 3
    # 设定拼接损失函数阈值，如果损失值超出阈值，则该块图像重新拼接,低噪点数据建议取0.2-0.5，高噪点数据建议取0.5-1.5
    bad_tile_threshold = 0.3
    threshold_step = 0.1  # 一般取bad_tile_threshold的20%
    # 一个块的最大计算次数
    max_calc_times = 2

##################################################
elif stitch_mode == 'Z':
    # 文件路径
    file_path = r'D:\Albert\Data\20220909_Image'
    # 图片文档命名格式
    file_name_format = r'S%.4d'
    # 拼接后的图片保存路径
    img_save_path = r'D:\Albert\Data\S'
    # 需要拼接的层数
    layer_num = 10
    # 通道数
    channel_num = 3
    # 基于第几个通道进行拼接，假如有3个通道，编号为0，1，2
    channel_ordinal = 0

    # 可选参数
    # 一般情况下默认即可
    # 图片类型
    img_file_type = 'tif'  # {'tif', 'tiff',...}
    # 图片数据类型
    img_data_type = 'uint8'  # {'uint8', 'uint16'}
    # 计算时考虑的重叠比率
    overlap_ratio = 0.3  # z
    # 计算时的像素步长
    step = 5
    # 是否重命名文件，如果选否，认真填写图片命名格式img_name_format和图片名
    if_rename_file = True
    # 图片命名格式
    img_name_format = '%s\%s_z%.4d_ch%.2d.%s'  # %(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    # 图片名
    img_name = 'Region'
    # 中间文件储存路径
    info_IO_path = file_path
    # 拼接时是否要进行中值降噪
    if_median_blur = True
    # 中值降噪的卷积核大小
    blur_kernel_size = 3
    # 垂直拼接时XY维度的降采样次数，取值为2的幂次方，建议降采样至图片大小小于800*800
    pyr_down_times = 4

if __name__=='__main__':
    if stitch_mode=='XYZ':
        for i in range(0,layer_num+2):
            if 0<=i-2<layer_num-1:
                vert_proc=Process(target=start_vertical_stitch,args=(
                    i - 2, img_name_format, img_path, info_IO_path, file_name_format, img_name,
                    channel_ordinal, img_file_type, img_data_type, overlap_ratio_vert, vert_step,
                    pyr_down_times, if_median_blur, blur_kernel_size
                ))
                vert_proc.start()
            else:
                vert_proc=None
            if 0<=i<layer_num:
                img_path = os.path.join(file_path, file_name_format % (i))
                info_file_path = os.path.join(img_path, 'MetaData', img_name + '.' + info_file_type)
                start_multi_stitchers(i,info_file_type,info_file_path,file_path,img_file_type,img_path,img_name_format,
                                  img_name,channel_num,channel_ordinal,img_data_type,overlap_ratio_hori,hori_step,alpha,
                                  if_median_blur,blur_kernel_size,bad_tile_threshold,threshold_step,max_calc_times,
                                  if_rename_file)
            if vert_proc!=None:
                vert_proc.join()
        export_img_vert_stit(layer_num, info_IO_path, file_path, file_name_format, img_save_path, img_name_format, img_name, channel_num,
                             img_file_type, img_data_type, 0)

    ####################################################################################################
    if stitch_mode=='XY':
        info_IO_path = os.path.join(file_path, 'MetaData')
        info_file_path = os.path.join(info_IO_path, img_name + '.' + info_file_type)
        start_multi_stitchers(0, info_file_type, info_file_path, info_IO_path, img_file_type, file_path, img_name_format,
                              img_name, channel_num, channel_ordinal, img_data_type, overlap_ratio, step, alpha,
                              if_median_blur, blur_kernel_size, bad_tile_threshold, threshold_step, max_calc_times,
                              if_rename_file)
        export_img_hori_stit(0, info_IO_path, file_path, img_save_path, img_name_format, img_name, channel_num,
                             img_file_type, img_data_type, 0, export_type='whole')

    ####################################################################################################
    if stitch_mode=='Z':
        for i in range(0,layer_num-1):
            start_vertical_stit_merged(i,img_name_format,img_path,info_IO_path,file_name_format,img_name,channel_ordinal,
                                       img_file_type,img_data_type,overlap_ratio=overlap_ratio,vert_step=step,
                                       pyr_down_times=pyr_down_times,if_median_blur=if_median_blur,blur_kernel_size=blur_kernel_size)
        export_img_vert_stit_merged(layer_num,info_IO_path,file_path,file_name_format,img_save_path,img_name_format,
                                    img_name,channel_num,img_file_type,img_data_type,0)







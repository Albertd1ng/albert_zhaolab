import os
import re


def rename_file(img_path_format, img_path, img_name, z_num, channel_num, img_type='tif'):
    r'''
    Rename file under the image path.

    Parameters
    ----------
    img_path_format - % format string
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_t%.4d_z%.4d_ch%.2d.%s'%(img_path,img_name,ordinal,z_dimension_ordinal,channel_ordinal,img_type)
    img_path - str
    img_name - str
    z_num - int
        z dimension ordinal
    channel_num - int
        the channel number of image.
    img_type - str
        'tif' and so on.

    Examples
    ----------
    >>>img_path_format=r'%s\%s_t%.4d_z%.4d_ch%.2d.%s'
    >>>img_path=r'D:\Data\Imaging\20220219'
    >>>img_name='Region'
    >>>z_num=200
    >>>channel_num=3
    >>>rename_file(img_path_format,img_path,img_name,z_num,channel_num,img_type='tif')
    '''
    file_list = os.listdir(img_path)
    for i, one_file in enumerate(file_list):
        if not os.path.isfile(os.path.join(img_path, one_file)):
            file_list.pop(i)
    file_num = len(file_list)
    tile_num = int(round(file_num / z_num / channel_num))
    this_file_num = 0
    for i in range(tile_num):
        for z in range(z_num):
            for c in range(channel_num):
                old_name = os.path.join(img_path, file_list[this_file_num])
                new_name = img_path_format % (img_path, img_name, i, z, c, img_type)
                os.rename(old_name, new_name)
                this_file_num += 1


def rename_file_Z_stit(img_path_format, img_path, img_name, z_num, channel_num, img_type='tif'):
    r'''
    Rename file under the image path for subsequent z-dimension stitch.

    Parameters
    ----------
    img_path_format - % format string
        the image path string format, %(img path, img name, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_z%.4d_ch%.2d.%s'%(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    img_path - str
    img_name - str
    z_num - int
        z dimension ordinal
    channel_num - int
        the channel number of image.
    img_type - str
        'tif' and so on.

    Examples
    ----------
    >>>img_path_format=r'%s\%s_z%.4d_ch%.2d.%s'
    >>>img_path=r'D:\Data\Imaging\20220219'
    >>>img_name='Region'
    >>>z_num=200
    >>>channel_num=3
    >>>rename_file_Z_stit(img_path_format,img_path,img_name,z_num,channel_num,img_type='tif')
    '''
    file_list = os.listdir(img_path)
    for i, one_file in enumerate(file_list):
        if not os.path.isfile(os.path.join(img_path, one_file)):
            file_list.pop(i)
    file_num = len(file_list)
    this_file_num = 0
    for z in range(z_num):
        for c in range(channel_num):
            old_name = os.path.join(img_path, file_list[this_file_num])
            new_name = img_path_format % (img_path, img_name, z, c, img_type)
            os.rename(old_name, new_name)
            this_file_num += 1


def rename_file_tile_id(img_path, tile_id_dict):
    r'''
    Substitute tile id in image file name with int type ordinal

    Parameters
    ----------
    img_path - str
    tile_id_dict - dict, {str : int}, (tile_num)
        store the tile_id and related ordinal number, such as {'AA01':0, 'AA02':1,...}

    Exmaple
    ----------
    >>>rename_file_tile_id(r'D:\Data\Imaging\20220219',{'AA01':0, 'AA02':1,...})
    '''
    file_list = os.listdir(img_path)
    for i, one_file in enumerate(file_list):
        if not os.path.isfile(img_path + '\\' + one_file):
            file_list.pop(i)
    for one_file in file_list:
        one_id_match = re.match(r'[A-Z][A-Z][0-9][0-9]', one_file)
        if one_id_match != None:
            one_id = one_id_match.group()
            new_name = os.path.join(img_path, re.sub(one_id, str(tile_id_dict[one_id]), one_file))
            old_name = os.path.join(img_path, one_file)
            os.rename(old_name, new_name)
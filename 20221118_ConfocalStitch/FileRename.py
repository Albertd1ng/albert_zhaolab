import os
import math


def rename_file(img_name_format, img_path, img_name, z_num, channel_num, img_type='tif'):
    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(img_path, file_list, img_type)
    file_num = len(file_list)
    tile_num = int(math.floor(file_num / z_num / channel_num))
    this_file_num = 0
    old_name_list = []
    new_name_list = []
    for i in range(tile_num):
        for z in range(z_num):
            for c in range(channel_num):
                old_name = os.path.join(img_path, file_list[this_file_num])
                new_name = os.path.join(img_path, img_name_format % (img_name, i, z, c, img_type))
                old_name_list.append(old_name)
                new_name_list.append(new_name)
                this_file_num += 1
    try:
        for i in range(len(old_name_list)):
            os.rename(old_name_list[i], new_name_list[i])
    except Exception as e:
        print(e)
        for i in range(len(old_name_list)):
            if os.path.exists(new_name_list[i]):
                os.rename(new_name_list[i], old_name_list[i])


def rename_file_Z_stit(img_name_format, img_path, img_name, z_num, channel_num, img_type='tif'):
    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(img_path, file_list, img_type)
    file_num = len(file_list)
    this_file_num = 0
    old_name_list = []
    new_name_list = []
    for z in range(z_num):
        for c in range(channel_num):
            old_name = os.path.join(img_path, file_list[this_file_num])
            new_name = os.path.join(img_path, img_name_format % (img_name, z, c, img_type))
            old_name_list.append(old_name)
            new_name_list.append(new_name)
            this_file_num += 1
    try:
        for i in range(len(old_name_list)):
            os.rename(old_name_list[i], new_name_list[i])
    except Exception as e:
        print(e)
        for i in range(len(old_name_list)):
            if os.path.exists(new_name_list[i]):
                os.rename(new_name_list[i], old_name_list[i])


def pop_other_type_file(img_path, file_list, file_type):
    for i, one_file in enumerate(file_list):
        if file_type not in one_file:
            file_list.pop(i)
            continue
        if not os.path.isfile(os.path.join(img_path, one_file)):
            file_list.pop(i)
            continue
    return file_list

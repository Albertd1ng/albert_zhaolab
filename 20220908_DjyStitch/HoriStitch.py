from ImgIO import import_img
from ImgBorder import get_border_list,get_2img_border_after_shift,get_ovl_img
from LossFunc import loss_func_for_list
from TileCont import choose_refe_tile,choose_refe_tile_for_stit,judge_tile_cont_of_one
from InfoIO import get_img_xml_info,get_img_txt_info,get_save_tile_cont_info,save_img_info,save_img_info_hori_stit
from AxisRange import calc_axis_range,find_first_last_index
from FileRename import rename_file,rename_file_tile_id

import numpy as np
import time
import cv2
import multiprocessing
from multiprocessing import Value,Array,Process,RLock
import ctypes
import random

def calc_xyz_shift(i,index_j,border_list,path_format,img_path,img_name,channel_ordinal,dim_elem_num,
                   img_type,img_data_type,img_mode,voxel_range,step,
                   alpha=0,if_median_blur=True,blur_kernel_size=3):
    r'''
    This function gets the best overlapping position for two partly overlapping images.

    Parameters
    ----------
    i - int
        the ordinal number of being stitched tile.
    index_j - list, int
        the ordinal number of tiles contact with i.th tile.
    border_list - list for array((2,6),dtype='uint32')
        all the border indexes between i.th and j.th tile for j in index_j.
        border specific form:
            [[x1_min,x1_max,y1_min,y1_max,z1_min,z1_max],
            [x2_min,x2_max,y2_min,y2_max,z2_min,z2_max]]
    path_format
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_z%.4d_ch%.2d.%s'%(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    img_path - str
    img_name - str
    channel_ordinal - int
        the channel number of image.
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    img_type - str
        'tif' and so on.
    img_data_type - str
        'uint8' or 'uint16'.
    img_mode - cv2 object
        usually cv2.IMREAD_GRAYSCALE for img_data_type='uint8', cv2.IMREAD_UNCHANGED for img_data_type='uint16'.
        if choose cv2.IMREAD_GRAYSCALE, the RGB image will be put in to one gray color, and data type of array will be uint8.
        if choose cv2.IMREAD_UNCHANGED, the RGB image remain unchanged, and the data type of array remain unchanged as uint8 or uint16.
    voxel_range - list/vector, int, (3)
        xyz shift range, as voxel number.
    step - int
        xyz shift step, as voxel number.
    alpha - float
        one hyperparameter to avoid images shift too far from actual correct position.
    if_median_blur - bool
        if True, median blur image to decrease noise points.
    blur_kernel_size - int

    Returns
    ----------
    xyz_shift_min - list/vector, int, (3)
        the best xyz shift to get the minimum loss.
        specific form:
            [xv_shift,yv_shift,zv_shift]
    loss-min - float
        the minimum loss when stitched
    '''
    #import being stitched image and contact images
    #print(voxel_range)
    if if_median_blur:
        img1 = np.float32(cv2.medianBlur(
            import_img(path_format,img_path, img_name, i, channel_ordinal, dim_elem_num,
                       img_type=img_type,img_data_type=img_data_type,img_mode=img_mode),
            blur_kernel_size))
    else:
        img1 = np.float32(import_img(path_format,img_path, img_name, i, channel_ordinal, dim_elem_num,
                       img_type=img_type,img_data_type=img_data_type,img_mode=img_mode))
    m1=np.mean(img1)
    if m1<0.001:
        return [0,0,0],0
    img2_list=[]
    for j in index_j:
        if if_median_blur:
            img2_list.append(np.float32(cv2.medianBlur(
                import_img(path_format,img_path, img_name, j, channel_ordinal, dim_elem_num,
                           img_type=img_type,img_data_type=img_data_type,img_mode=img_mode),
                blur_kernel_size)))
        else:
            img2_list.append(np.float32(import_img(path_format,img_path, img_name, j, channel_ordinal, dim_elem_num,
                           img_type=img_type,img_data_type=img_data_type,img_mode=img_mode)))
    ovl_num=len(border_list)
    #start stitch process
    xv_shift,yv_shift,zv_shift=0,0,0
    loss_min=np.inf
    for x in range(-voxel_range[0],voxel_range[0]+1,step):
        for y in range(-voxel_range[1],voxel_range[1]+1,step):
            for z in range(-voxel_range[2],voxel_range[2]+1,step):
                border_list_shift=[]
                ovl1_list=[]
                ovl2_list=[]
                for i in range(ovl_num):
                    #print([x,y,z])
                    #print(border_list[i])
                    border_list_shift.append(get_2img_border_after_shift(dim_elem_num,border_list[i],[x,y,z]))
                    #print(border_list_shift[i])
                for i in range(ovl_num):
                    ovl1,ovl2=get_ovl_img(img1,img2_list[i],border_list_shift[i])
                    ovl1_list.append(ovl1)
                    ovl2_list.append(ovl2)
                this_loss=loss_func_for_list(ovl1_list,ovl2_list,[x,y,z],alpha=alpha)
                #print((x,y,z),this_loss)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    xv_shift1, yv_shift1, zv_shift1 = xv_shift, yv_shift, zv_shift
    for x in range(xv_shift1-step+1,xv_shift1+step):
        for y in range(yv_shift1-step+1,yv_shift1+step):
            for z in range(zv_shift1-step+1,zv_shift1+step):
                border_list_shift=[]
                ovl1_list=[]
                ovl2_list=[]
                for i in range(ovl_num):
                    border_list_shift.append(get_2img_border_after_shift(dim_elem_num,border_list[i],[x,y,z]))
                for i in range(ovl_num):
                    ovl1,ovl2=get_ovl_img(img1,img2_list[i],border_list_shift[i])
                    ovl1_list.append(ovl1)
                    ovl2_list.append(ovl2)
                this_loss=loss_func_for_list(ovl1_list,ovl2_list,[x,y,z],alpha=alpha)
                #print((x,y,z),this_loss)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    #print(xv_shift,yv_shift,zv_shift,loss_min)
    return [xv_shift,yv_shift,zv_shift],loss_min

def choose_start_img(path_format,tile_num,img_path,img_name,channel_ordinal,dim_elem_num,
                     img_type,img_data_type,img_mode=cv2.IMREAD_GRAYSCALE,img_thre=10.0,choose_range=[0.3,0.7]):
    r'''
    Choose the first image and set it been stitched

    Parameters
    ----------
    path_format
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_z%.4d_ch%.2d.%s'%(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    tile_num - int
        the quantity of tiles.
    img_path - str
    img_name - str
    channel_ordinal - int
        the channel number of image.
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    img_type - str
        'tif' and so on.
    img_data_type - str
        'uint8' or 'uint16'.
    img_mode - cv2 object
        usually cv2.IMREAD_GRAYSCALE for img_data_type='uint8', cv2.IMREAD_UNCHANGED for img_data_type='uint16'.
        if choose cv2.IMREAD_GRAYSCALE, the RGB image will be put in to one gray color, and data type of array will be uint8.
        if choose cv2.IMREAD_UNCHANGED, the RGB image remain unchanged, and the data type of array remain unchanged as uint8 or uint16.
    img_thre - float
        image tile whose mean signal exceed img_thre can be chosen.
    choose_range - list, float, (2)
        value from 0 to 1, choose the tile in the range of [choose_range[0]*tile_num),int(choose_range[1]*tile_num]

    Returns
    ----------
    a - int
        chosen tile ordinal number to be set been stitched.
    '''
    m,a=0,0
    while(m<img_thre):
        a=random.randint(int(choose_range[0]*tile_num),int(choose_range[1]*tile_num))
        m=np.mean(import_img(path_format,img_path,img_name,a,channel_ordinal,dim_elem_num,
                             img_type=img_type,img_data_type=img_data_type,img_mode=img_mode))
    print('start_tile is %d.th tile, average is %.8f'%(a,m))
    return a

def run_sift_stitcher(lock,path_format,img_path,img_name,channel_ordinal,img_type,img_data_type,img_mode,
                      dim_elem_num,dim_len,voxel_len,tile_num,tile_contact,tile_pos,tile_pos_stitch,
                      if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_shelved_bad,if_tile_being_stitched,
                      voxel_range,step,alpha=0,if_median_blur=True,blur_kernel_size=3,bad_tile_threshold=1.0,
                      threshold_step=0.2,max_calc_times=2):
    r'''
    Parameters
    ----------
    lock - multiprocessing.RLock()
    path_format - % format string
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_z%.4d_ch%.2d.%s'%(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    img_path - str
    img_name - str
    ordinal - int
        the tile ordinal, when the layer contains multiple tiles.
    channel_ordinal - int
        the channel number of image.
    dim_elem_num - vector, uint32, (3)
        the quantity of voxels for each dimension.
    img_type - str
        'tif' and so on.
    img_data_type - str
        'uint8' or 'uint16'.
    img_mode - cv2 object
        usually cv2.IMREAD_GRAYSCALE for img_data_type='uint8', cv2.IMREAD_UNCHANGED for img_data_type='uint16'.
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    tile_num - int
        the quantity of tiles.
    tile_contact - array, bool, (tile_num,tile_num)
        the tile contact array for all tiles. tile_contact[i,j] means whether i.th tile and j.th tile have overlapped region.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile.
    tile_pos_stitch - vector/array/multiprocessing.Array, float64, (tile_num*3)
        the XYZ position of all tiles during stitch process.
    if_tile_stitched - multiprocessing.Array, bool, (tile_num)
        True for tile been stitched, False for not yet.
    if_tile_shelved - multiprocessing.Array, bool, (tile_num)
        True for tile been shelved for it has no contact tile or it's considered a bad tile.
    if_tile_bad - multiprocessing.Array, int, (tile_num)
        0 for a good tile.
        int>0 for tile been considered a bad tile for its loss exceeds the bad_tile_threshold.
    if_tile_being_stitched - multiprocessing.Array, bool, (tile_num)
        True for tile being stitched.
    voxel_range - list/vector, int, (3)
        xyz shift range, as voxel number.
    step - int
        xyz shift step, as voxel number.
    alpha - float
        one hyperparameter to avoid images shift too far from actual correct position.
    if_median_blur - bool
        if True, median blur image to decrease noise points.
    blur_kernel_size - int
    bad_tile_threshold - float
        if loss of one tile exceed bad_tile_threshold, it's considered a bad tile.
    threshold_step - float
    max_calc_times - int
    '''
    stitch_num=0
    this_name=multiprocessing.current_process().name
    while(False in if_tile_stitched):
        lock.acquire()
        #choose tile that will be stitched
        usable_tile_index=[index for index,value in enumerate(
            zip(if_tile_stitched,if_tile_shelved,if_tile_shelved_bad,if_tile_being_stitched)) if not any(value)]
        if len(usable_tile_index)==0:
            for i in range(tile_num):
                if_tile_shelved[i]=False
            usable_tile_index = [index for index, value in enumerate(
                zip(if_tile_stitched, if_tile_shelved, if_tile_shelved_bad, if_tile_being_stitched)) if not any(value)]
            if len(usable_tile_index)==0:
                for i in range(tile_num):
                    if_tile_shelved_bad[i]=False
                usable_tile_index = [index for index, value in enumerate(
                    zip(if_tile_stitched, if_tile_shelved, if_tile_shelved_bad, if_tile_being_stitched)) if not any(value)]
                if len(usable_tile_index)==0:
                    time.sleep(1)
                    lock.release()
                    #print('All shelved tile has been released')
                    continue
        i=usable_tile_index[random.randint(0,len(usable_tile_index)-1)]
        #choose reference tile
        j=choose_refe_tile(i,tile_contact[i,:],if_tile_stitched)
        if j==-1:
            if_tile_shelved[i]=True
            lock.release()
            #print('%d.th tile has no appropriate contact tile'%(i))
            continue
        if_tile_being_stitched[i]=True
        tile_pos_stitch[3*i:3*i+3]=tile_pos_stitch[3*j:3*j+3]+tile_pos[i,:]-tile_pos[j,:]
        lock.release()
        tile_contact_list=judge_tile_cont_of_one(i,dim_len,np.array(tile_pos_stitch).reshape(-1,3))
        index_j=choose_refe_tile_for_stit(tile_contact_list,if_tile_stitched)
        ovl_num=len(index_j)
        #print(f'{this_name} is stitching %{i}.th tile with {index_j}.th tiles')
        border_list=get_border_list(i,index_j,dim_elem_num,dim_len,voxel_len,tile_pos_stitch)
        xyz_shift,loss_min=calc_xyz_shift(i,index_j,border_list,path_format,img_path,img_name,channel_ordinal,
                                          dim_elem_num,img_type,img_data_type,img_mode,voxel_range,step,
                                          alpha=alpha,if_median_blur=if_median_blur,blur_kernel_size=blur_kernel_size)
        lock.acquire()
        if_tile_being_stitched[i]=False
        if loss_min>bad_tile_threshold+threshold_step*if_tile_bad[i]:
            if_tile_bad[i]=if_tile_bad[i]+1
            if if_tile_bad[i]>=max_calc_times:
                if_tile_stitched[i]=True
            else:
                if_tile_shelved_bad[i]=True
            print('%d.th tile is a bad tile, times is %d'%(i,if_tile_bad[i]))
        else:
            if_tile_stitched[i]=True
            tile_pos_stitch[3*i:3*i+3]=tile_pos_stitch[3*i:3*i+3]-np.array(xyz_shift,dtype='float64')*voxel_len
            stitch_num+=1
        lock.release()
        print('%s has stitched %d tiles, current stitch is %d.th tile with %s tiles,\nxyz_shift is (%d, %d, %d), loss is %.8f'
              %(this_name,stitch_num,i,str(index_j),*xyz_shift,loss_min))
    print('!!!!!!%s stops and has stitched %d tiles.!!!!!!'%(this_name,stitch_num))

def start_multi_stitchers(layer_num,info_file_type,info_file_path,info_IO_path,img_file_type,img_path,
                          img_name_format,img_name,channel_num,channel_ordinal,img_data_type,overlap_ratio_hori,
                          step,alpha,if_median_blur,blur_kernel_size,bad_tile_threshold,threshold_step,max_calc_times,
                          if_rename_file):
    r'''
    Parameters
    ----------
    layer_num - int
    info_file_type - str
    info_IO_path - str
    img_file_type - str
    img_path - str
    img_name_format - str
    img_name - str
    overlap_ratio_hori - list, float, (3)
    step - int
    if_median_blur - bool
    blur_kernel_size - int
    bad_tile_threshold - float
    threshold_step - float
    max_calc_times - int
    '''
    #input basic information
    if info_file_type=='xml':
        dim_elem_num,dim_len,voxel_len,tile_num,tile_pos=get_img_xml_info(info_file_path)
        if if_rename_file:
            rename_file(img_name_format, img_path, img_name, dim_elem_num[2], channel_num, img_file_type)
    elif info_file_type=='txt':
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos, tile_id_dict= get_img_txt_info(info_file_path)
        if if_rename_file:
            rename_file_tile_id(img_path,tile_id_dict)
            rename_file(img_name_format, img_path, img_name, dim_elem_num[2], channel_num, img_file_type)
    else:
        print('Input Error')
        return
    save_img_info(layer_num, info_IO_path, dim_elem_num, dim_len, voxel_len, tile_num, tile_pos)
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    else:
        print('Input Error')
        return
    voxel_range=np.int32(dim_elem_num.astype('float64')*np.array(overlap_ratio_hori,dtype='float64'))
    tile_contact=get_save_tile_cont_info(layer_num, info_IO_path, dim_len, tile_pos)
    lock=RLock()
    start_index = choose_start_img(img_name_format,tile_num,img_path, img_name,channel_ordinal,dim_elem_num,
                                   img_file_type,img_data_type,img_mode=img_mode)
    if_tile_stitched = Array(ctypes.c_bool, [False for i in range(tile_num)])
    if_tile_stitched[start_index] = True
    if_tile_being_stitched = Array(ctypes.c_bool, [False for i in range(tile_num)])
    if_tile_shelved = Array(ctypes.c_bool, [False for i in range(tile_num)])
    if_tile_shelved_bad=Array(ctypes.c_bool, [False for i in range(tile_num)])
    if_tile_bad = Array('i', [0 for i in range(tile_num)])
    tile_pos_stitch = Array('d', [0 for i in range(tile_num * 3)])
    process_num=min(round(0.5*multiprocessing.cpu_count()),20)
    #process_num=1
    print('Current processing quantities: %d' % (process_num))
    process_list = []
    for i in range(process_num):
        one_pro = Process(target=run_sift_stitcher,
                          args=(lock, img_name_format,img_path,img_name,channel_ordinal,img_file_type,img_data_type,
                                img_mode,dim_elem_num,dim_len,voxel_len,tile_num,tile_contact,tile_pos,tile_pos_stitch,
                                if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_shelved_bad,if_tile_being_stitched,
                                voxel_range,step,alpha,if_median_blur,blur_kernel_size,bad_tile_threshold,
                                threshold_step,max_calc_times))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    #####################################################################################
    print('start saving_data')
    tile_pos_stitch = np.array(tile_pos_stitch).reshape(tile_num, 3)
    axis_range, voxel_num = calc_axis_range(tile_pos_stitch, dim_elem_num, voxel_len)
    first_last_index = find_first_last_index(tile_pos_stitch, dim_elem_num, axis_range, voxel_len, voxel_num)
    save_img_info_hori_stit(layer_num, info_IO_path, tile_pos_stitch, axis_range, first_last_index)
    print('end saving data')
    return tile_pos_stitch
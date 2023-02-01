import cv2
import numpy as np
import os
from AxisRange import calc_max_axis_range_vert,calc_max_axis_range_vert_merged
from InfoIO import get_img_npy_basic_info,get_img_info_vert_stit,get_img_info_hori_stit,get_img_info_vert_stit_merged


def import_img(path_format, img_path, img_name, ordinal, channel_ordinal, dim_elem_num,
               img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    r'''
    Import 3D image and store in a 3D numpy array.

    Parameters
    ----------
    path_format - % format string
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_t%.4d_z%.4d_ch%.2d.%s'%(img_path,img_name,ordinal,z_dimension_ordinal,channel_ordinal,img_type)
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
        if choose cv2.IMREAD_GRAYSCALE, the RGB image will be put in to one gray color, and data type of array will be uint8.
        if choose cv2.IMREAD_UNCHANGED, the RGB image remain unchanged, and the data type of array remain unchanged as uint8 or uint16.

    Returns
    ----------
    voxel_array - 3D array, uint8 or uint16, (dim_elem_num[0],dim_elem_num[1],dim_elem_num[2])
        one 3D image array.

    Examples
    ----------
    >>>path_format=r'%s\%s_t%.4d_z%.4d_RAW_ch%.2d.tif'
    >>>img_path=r'D:\Data\Imaging\20220219'
    >>>img_name='Region'
    >>>ordinal=1
    >>>channel_ordinal=0
    >>>dim_elem_num=[512,512,200]
    >>>one_img=import_img(path_format,img_path,img_name,ordinal,channel_ordinal,dim_elem_num,
                            img_type='tif',img_data_type='uint8',img_mode=cv2.IMREAD_GRAYSCALE)
    '''
    voxel_array = np.zeros(tuple(dim_elem_num), dtype=img_data_type)
    for i in range(dim_elem_num[2]):
        one_img_name = path_format % (img_path, img_name, ordinal, i, channel_ordinal, img_type)
        voxel_array[:, :, i] = cv2.imread(one_img_name, img_mode)
    return voxel_array


def import_img_2D(path_format, img_path, img_name, z_th, channel_ordinal,
                  img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    r'''
    Import 2D image and store in 2D array.

    Parameters
    ----------
    path_format - % format string
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_z%.4d_ch%.2d.%s'%(img_path,img_name,z_dimension_ordinal,channel_ordinal,img_type)
    img_path - str
    img_name - str
    channel_ordinal - int
        the channel number of image.
    img_type - str
        'tif' and so on.
    img_data_type - str
        'uint8' or 'uint16'.
    img_mode - cv2 object
        usually cv2.IMREAD_GRAYSCALE for img_data_type='uint8', cv2.IMREAD_UNCHANGED for img_data_type='uint16'.
        if choose cv2.IMREAD_GRAYSCALE, the RGB image will be put in to one gray color, and data type of array will be uint8.
        if choose cv2.IMREAD_UNCHANGED, the RGB image remain unchanged, and the data type of array remain unchanged as uint8 or uint16.

    Returns
    ----------
    voxel_array - array, uint8 or uint16, (m,n)
        one 3D image array.

    Examples
    ----------
    >>>path_format=r'%s\%s_z%.4d_ch%.2d.tif'%
    >>>img_path=r'D:\Data\Imaging\20220219'
    >>>img_name='Region'
    >>>z_th=100
    >>>channel_ordinal=0
    >>>one_img=import_img_2D(path_format,img_path,img_name,z_th,channel_ordinal,
                            img_type='tif',img_data_type='uint8',img_mode=cv2.IMREAD_GRAYSCALE)
    '''
    one_img_name = path_format % (img_path, img_name, z_th, channel_ordinal, img_type)
    return cv2.imread(one_img_name, img_mode)

def import_img_2D_one_tile(path_format, img_path, img_name, ordinal, z_th, channel_ordinal,
                  img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    '''
    Import 2D image of one tile, and store in 2D array.

    Parameters
    ----------
    path_format - % format string
        the image path string format, %(img path, img name, tile ordinal, z dimension ordinal, channel ordinal, image file format).
        example:
            r'%s\%s_t%.4d_z%.4d_ch%.2d.%s'%(img_path,img_name,ordinal,z_dimension_ordinal,channel_ordinal,img_type)
    img_path - str
    img_name - str
    ordinal - int
        the tile ordinal, when the layer contains multiple tiles.
    z_th - int
        the z dimension index
    channel_ordinal - int
        the channel number of image.
    img_type - str
        'tif' and so on.
    img_data_type - str
        'uint8' or 'uint16'.
    img_mode - cv2 object
        usually cv2.IMREAD_GRAYSCALE for img_data_type='uint8', cv2.IMREAD_UNCHANGED for img_data_type='uint16'.
        if choose cv2.IMREAD_GRAYSCALE, the RGB image will be put in to one gray color, and data type of array will be uint8.
        if choose cv2.IMREAD_UNCHANGED, the RGB image remain unchanged, and the data type of array remain unchanged as uint8 or uint16.

    Returns
    ----------
    voxel_array - array, uint8 or uint16, (m,n)
        one 3D image array.
    '''
    one_img_name = path_format % (img_path, img_name, ordinal, z_th, channel_ordinal,img_type)
    return cv2.imread(one_img_name, img_mode)

def import_whole_img(z_index, path_format, img_path, img_name, channel_ordinal,
                     axis_range, dim_elem_num, voxel_num, voxel_len, tile_pos,
                     img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    r'''
    Parameters
    ----------
    z_index - int
        the z dimension index for horizontally or vertically stitched 3D image, value vary from 0 to voxel_num[2].
    '''
    tile_num = tile_pos.shape[0]
    whole_img = np.zeros((voxel_num[1::-1]), dtype='uint8')
    this_z = axis_range[2, 0] + voxel_len[2] * z_index
    for j in range(tile_num):
        z_th = np.int64(np.round((this_z - tile_pos[j, 2]) / voxel_len[2]))
        x_th = np.int64(np.round((tile_pos[j, 0] - axis_range[0, 0]) / voxel_len[0]))
        y_th = np.int64(np.round((tile_pos[j, 1] - axis_range[1, 0]) / voxel_len[0]))
        img_2D = import_img_2D_one_tile(path_format,img_path, img_name, j, z_th,channel_ordinal,img_type=img_type,
                                        img_data_type=img_data_type,img_mode=img_mode)
        whole_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
    return whole_img

def export_img_vert_stit(layer_num,info_IO_path,file_path,file_name_format,img_save_path,img_name_format,img_name,
                         channel_num,img_type,img_data_type,img_num):
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    max_xy_axis_range,max_xy_voxel_num=calc_max_axis_range_vert(layer_num,info_IO_path)
    for i in range(layer_num):
        dim_elem_num, dim_len, voxel_len = get_img_npy_basic_info(i, info_IO_path)
        tile_pos, axis_range, first_last_index=get_img_info_vert_stit(i,info_IO_path)
        tile_num=tile_pos.shape[0]
        img_path=os.path.join(file_path,file_name_format%(i))
        for j in range(first_last_index[0],first_last_index[1]):
            if channel_num == 1:
                this_img = np.zeros((max_xy_voxel_num[1::-1]), dtype=img_data_type)
                this_z=axis_range[2,0]+voxel_len[2]*j
                for k in range(tile_num):
                    if tile_pos[k,2]+dim_len[2]<this_z or tile_pos[k,2]>this_z:
                        continue
                    z_th=np.int64(np.round((this_z-tile_pos[k,2])/voxel_len[2]))
                    if z_th>=dim_elem_num[2] or z_th<0:
                        continue
                    x_th = np.int64(np.round((tile_pos[k, 0] - max_xy_axis_range[0, 0]) / voxel_len[0]))
                    y_th = np.int64(np.round((tile_pos[k, 1] - max_xy_axis_range[1, 0]) / voxel_len[1]))
                    if x_th < 0 or x_th + dim_elem_num[0] > max_xy_voxel_num[0] or y_th < 0 or y_th + dim_elem_num[1] > \
                            max_xy_voxel_num[1]:
                        continue
                    img_2D=import_img_2D_one_tile(img_name_format, img_path, img_name, k, z_th, 0,
                                               img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
                    this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
                cv2.imwrite(r'%s\a%.4d.%s'%(img_save_path,img_num,img_type),this_img)
                img_num+=1
            elif channel_num>1:
                this_img = np.zeros(np.hstack((max_xy_voxel_num[1::-1],[3])), dtype=img_data_type)
                this_z = axis_range[2, 0] + voxel_len[2] * j
                for k in range(tile_num):
                    if tile_pos[k, 2] + dim_len[2] < this_z or tile_pos[k, 2] > this_z:
                        continue
                    z_th = np.int64(np.round((this_z - tile_pos[k, 2]) / voxel_len[2]))
                    if z_th >= dim_elem_num[2] or z_th < 0:
                        continue
                    x_th = np.int64(np.round((tile_pos[k, 0] - max_xy_axis_range[0, 0]) / voxel_len[0]))
                    y_th = np.int64(np.round((tile_pos[k, 1] - max_xy_axis_range[1, 0]) / voxel_len[1]))
                    if x_th < 0 or x_th + dim_elem_num[0] > max_xy_voxel_num[0] or y_th < 0 or y_th + dim_elem_num[1] > \
                            max_xy_voxel_num[1]:
                        continue
                    for c in range(channel_num):
                        img_2D = import_img_2D_one_tile(img_name_format, img_path, img_name, k, z_th, c,
                                                    img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
                        this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0],c] = img_2D
                cv2.imwrite(r'%s\a%.4d.%s' % (img_save_path, img_num, img_type), this_img)
                img_num += 1
    return img_num

def export_img_vert_stit_merged(layer_num,info_IO_path,file_path,file_name_format,img_save_path,img_name_format,img_name,
                                channel_num,img_type,img_data_type,img_num):
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    xy_axis_range,xy_voxel_num=calc_max_axis_range_vert_merged(layer_num,info_IO_path)
    for i in range(layer_num):
        img_path=os.path.join(file_path,file_name_format%(i))
        dim_elem_num,axis_range,first_last_index=get_img_info_vert_stit_merged(i,info_IO_path)
        for j in range(first_last_index[0],first_last_index[1]):
            if channel_num==1:
                this_img=np.zeros(xy_voxel_num[1::-1],dtype=img_data_type)
                x_th,y_th=axis_range[0,0]-xy_axis_range[0,0],axis_range[1,0]-xy_axis_range[1,0]
                img_2D=import_img_2D(img_name_format,img_path,img_name,j,0,img_type=img_type,
                                     img_data_type=img_data_type,img_mode=img_mode)
                this_img[y_th:y_th+dim_elem_num[1],x_th:x_th+dim_elem_num[0]]=img_2D
                cv2.imwrite(r'%s\a%.4d.%s' % (img_save_path, img_num, img_type), this_img)
                img_num += 1
            if channel_num>1:
                this_img = np.zeros(np.hstack((xy_voxel_num[1::-1], [3])), dtype=img_data_type)
                x_th, y_th = axis_range[0, 0] - xy_axis_range[0, 0], axis_range[1, 0] - xy_axis_range[1, 0]
                for c in range(channel_num):
                    img2D=import_img_2D(img_name_format,img_path,img_name,j,c,img_type=img_type,
                                        img_data_type=img_data_type,img_mode=img_mode)
                    this_img[y_th:y_th+dim_elem_num[1],x_th:x_th+dim_elem_num[0],c]=img2D
                cv2.imwrite(r'%s\a%.4d.%s' % (img_save_path, img_num, img_type), this_img)
                img_num+=1


def export_img_hori_stit(layer_num,info_IO_path,img_path,img_save_path,img_name_format,img_name,channel_num,img_type,
                         img_data_type,img_num,export_type='whole'):
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    dim_elem_num, dim_len, voxel_len = get_img_npy_basic_info(layer_num, info_IO_path)
    tile_pos, axis_range, first_last_index = get_img_info_hori_stit(layer_num, info_IO_path)
    voxel_num = np.int64(np.round((axis_range[:, 1] - axis_range[:, 0]) / voxel_len))
    print(voxel_num)
    if export_type=='whole':
        first_last_index=np.array((0,voxel_num[2]-1),dtype='int64')
    tile_num = tile_pos.shape[0]
    for j in range(first_last_index[0], first_last_index[1]):
        if channel_num == 1:
            this_img = np.zeros((voxel_num[1::-1]), dtype=img_data_type)
            this_z = axis_range[2, 0] + voxel_len[2] * j
            for k in range(tile_num):
                if tile_pos[k, 2] + dim_len[2] < this_z or tile_pos[k, 2] > this_z:
                    continue
                z_th = np.int64(np.round((this_z - tile_pos[k, 2]) / voxel_len[2]))
                if z_th >= dim_elem_num[2] or z_th < 0:
                    continue
                x_th = np.int64(np.round((tile_pos[k, 0] - axis_range[0, 0]) / voxel_len[0]))
                y_th = np.int64(np.round((tile_pos[k, 1] - axis_range[1, 0]) / voxel_len[1]))
                if x_th < 0 or x_th + dim_elem_num[0] > voxel_num[0] or y_th < 0 or y_th + dim_elem_num[1] > \
                        voxel_num[1]:
                    continue
                img_2D = import_img_2D_one_tile(img_name_format, img_path, img_name, k, z_th, 0,
                                                img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
                this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
            cv2.imwrite(r'%s\a%.4d.%s' % (img_save_path, img_num, img_type), this_img)
            img_num += 1
        elif channel_num > 1:
            this_img = np.zeros(np.hstack((voxel_num[1::-1], [3])), dtype=img_data_type)
            this_z = axis_range[2, 0] + voxel_len[2] * j
            for k in range(tile_num):
                if tile_pos[k, 2] + dim_len[2] < this_z or tile_pos[k, 2] > this_z:
                    continue
                z_th = np.int64(np.round((this_z - tile_pos[k, 2]) / voxel_len[2]))
                if z_th >= dim_elem_num[2] or z_th < 0:
                    continue
                x_th = np.int64(np.round((tile_pos[k, 0] - axis_range[0, 0]) / voxel_len[0]))
                y_th = np.int64(np.round((tile_pos[k, 1] - axis_range[1, 0]) / voxel_len[1]))
                if x_th < 0 or x_th + dim_elem_num[0] > voxel_num[0] or y_th < 0 or y_th + dim_elem_num[1] > \
                        voxel_num[1]:
                    continue
                for c in range(channel_num):
                    img_2D = import_img_2D_one_tile(img_name_format, img_path, img_name, k, z_th, c,
                                                    img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
                    this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0], c] = img_2D
            cv2.imwrite(r'%s\a%.4d.%s' % (img_save_path, img_num, img_type), this_img)
            img_num += 1
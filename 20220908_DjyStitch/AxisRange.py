from InfoIO import get_img_npy_basic_info
import numpy as np

def calc_axis_range(tile_pos,dim_elem_num,voxel_len):
    r'''
    Calculate axis ranges and voxel numbers in XYZ dimensions.

    Parameters
    ----------
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile.
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.

    Returns
    ----------
    axis_range - array, float64, (3,2)
        the axis ranges in XYZ dimensions
        specific form:
            [[x_min,x_max],
            [y_min,y_max],
            [z_min,z_max]]
    voxel_num - vector, uint32, (3)
        the voxel number in XYZ dimensions
    '''
    axis_range=np.zeros((3,2))
    axis_range[0,0],axis_range[0,1]=np.min(tile_pos[:,0]),np.max(tile_pos[:,0])+voxel_len[0]*dim_elem_num[0]
    axis_range[1,0],axis_range[1,1]=np.min(tile_pos[:,1]),np.max(tile_pos[:,1])+voxel_len[1]*dim_elem_num[1]
    axis_range[2,0],axis_range[2,1]=np.min(tile_pos[:,2]),np.max(tile_pos[:,2])+voxel_len[2]*dim_elem_num[2]
    voxel_num=np.uint32(np.round((axis_range[:,1]-axis_range[:,0])/voxel_len))
    return axis_range,voxel_num

def find_first_last_index(tile_pos,dim_elem_num,axis_range,voxel_len,voxel_num):
    '''
    Calcualte the first and last index in Z dimension to output a whole image containing all tiles.

    Parameters
    ----------
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile.
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    axis_range - array, float64, (3,2)
        the axis ranges in XYZ dimensions
        specific form:
            [[x_min,x_max],
            [y_min,y_max],
            [z_min,z_max]]
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    voxel_num - vector, uint32, (3)
        the voxel number in XYZ dimensions

    Returns
    ----------
    first_last_index - vector, uint32, (2)
        the first and last index in Z dimension to output a whole image containing all tiles.
        to output the first whole stitched image, the Z-Pos is:
            axis_range[2,0]+voxel_len[2]*first_last_index[0]
        to output the last whole stitched image, the Z-Pos is:
            axis_range[2,0]+voxel_len[2]*first_last_index[1]
    '''
    first_last_index=np.array([voxel_num[2],0],dtype='uint32')
    tile_num=tile_pos.shape[0]
    for i in range(voxel_num[2]):
        num_one_layer=0
        this_z=axis_range[2,0]+voxel_len[2]*i
        for j in range(tile_num):
            if tile_pos[j,2]+voxel_len[2]*dim_elem_num[2]<this_z:
                continue
            if tile_pos[j,2]>this_z:
                continue
            z_th=np.int32(np.round((this_z-tile_pos[j,2])/voxel_len[2]))
            if z_th>=dim_elem_num[2]:
                continue
            num_one_layer+=1
        if num_one_layer==tile_num:
            if i<=first_last_index[0]:
                first_last_index[0]=i
            if i>=first_last_index[1]:
                first_last_index[1]=i
    return first_last_index

def calc_max_axis_range_vert(layer_num,save_path):
    dim_elem_num, dim_len, voxel_len = get_img_npy_basic_info(0, save_path)
    axis_range_array = np.zeros((layer_num, 6))
    for i in range(0, layer_num):
        axis_range_array[i, :] = np.load(save_path + r'\axis_range_zstitch_%.4d.npy' % (i)).reshape((1, -1))
    xy_axis_range = np.zeros((2, 2))
    xy_axis_range[0, 0], xy_axis_range[0, 1] = np.min(axis_range_array[:, 0]) - voxel_len[0] * 2, np.max(
        axis_range_array[:, 1]) + voxel_len[0] * 2
    xy_axis_range[1, 0], xy_axis_range[1, 1] = np.min(axis_range_array[:, 2]) - voxel_len[0] * 2, np.max(
        axis_range_array[:, 3]) + voxel_len[0] * 2
    xy_voxel_num = np.int64(np.round((xy_axis_range[:, 1] - xy_axis_range[:, 0]) / voxel_len[0:2]) + 1)
    return xy_axis_range, xy_voxel_num

def calc_max_axis_range_vert_merged(layer_num,save_path):
    axis_range_array=np.zeros((layer_num,4),dtype='int64')
    xy_axis_range=np.zeros((2,2),dtype='int64')
    for i in range(layer_num):
        axis_range_array[i,:]=np.load(save_path+r'\axis_range_zstitch_%.4d'%(i)).reshape((1,-1))
    xy_axis_range[0,0],xy_axis_range[0,1]=np.min(axis_range_array[:,0]),np.max(axis_range_array[:,1])
    xy_axis_range[1,0],xy_axis_range[1,1]=np.min(axis_range_array[:,2]),np.max(axis_range_array[:,3])
    xy_voxel_num=np.uint32(xy_axis_range[:,1]-xy_axis_range[:,0])
    return xy_axis_range,xy_voxel_num
import numpy as np


def get_2img_border(dim_elem_num, dim_len, voxel_len, tile_pos):
    r'''
    Get the border voxel index for two overlapping 3D images

    Parameters
    ----------
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    tile_pos - array, float64, (2,3)
        the XYZ position of two 3D images.

    Returns
    ----------
    voxel_border - array, uint32, (2,6)
        the border indexes on three dimensions of two images.
        i.th dimension values range is [0,dim_elem_num[i]-1].
        specific form:
            [[x1_min,x1_max,y1_min,y1_max,z1_min,z1_max],
            [x2_min,x2_max,y2_min,y2_max,z2_min,z2_max]]
    or---
    None -
        if the two tiles don't have overlapped area actually, or they lose contact during stitching, return None

    '''
    # x/y/z_min/max, the positions of overlapping image border
    x_min, x_max = np.max(tile_pos[:, 0]), np.min(tile_pos[:, 0]) + dim_len[0] - voxel_len[0]
    y_min, y_max = np.max(tile_pos[:, 1]), np.min(tile_pos[:, 1]) + dim_len[1] - voxel_len[1]
    z_min, z_max = np.max(tile_pos[:, 2]), np.min(tile_pos[:, 2]) + dim_len[2] - voxel_len[2]
    # x/y/zv_min/max, the voxel index of overlapping image border
    xv1_min, xv1_max = np.round((x_min - tile_pos[0, 0]) / voxel_len[0]), np.round(
        (x_max - tile_pos[0, 0]) / voxel_len[0])
    yv1_min, yv1_max = np.round((y_min - tile_pos[0, 1]) / voxel_len[1]), np.round(
        (y_max - tile_pos[0, 1]) / voxel_len[1])
    zv1_min, zv1_max = np.round((z_min - tile_pos[0, 2]) / voxel_len[2]), np.round(
        (z_max - tile_pos[0, 2]) / voxel_len[2])
    xv2_min, xv2_max = np.round((x_min - tile_pos[1, 0]) / voxel_len[0]), np.round(
        (x_max - tile_pos[1, 0]) / voxel_len[0])
    yv2_min, yv2_max = np.round((y_min - tile_pos[1, 1]) / voxel_len[1]), np.round(
        (y_max - tile_pos[1, 1]) / voxel_len[1])
    zv2_min, zv2_max = np.round((z_min - tile_pos[1, 2]) / voxel_len[2]), np.round(
        (z_max - tile_pos[1, 2]) / voxel_len[2])
    if (0 <= xv1_min < xv1_max < dim_elem_num[0] and 0 <= xv2_min < xv2_max < dim_elem_num[0] and
            0 <= yv1_min < yv1_max < dim_elem_num[1] and 0 <= yv2_min < yv2_max < dim_elem_num[1] and
            0 <= zv1_min < zv1_max < dim_elem_num[2] and 0 <= zv2_min < zv2_max < dim_elem_num[2]):
        voxel_border = np.array([[xv1_min, xv1_max, yv1_min, yv1_max, zv1_min, zv1_max],
                                 [xv2_min, xv2_max, yv2_min, yv2_max, zv2_min, zv2_max]], dtype='uint32')
        return voxel_border
    else:
        return np.array([])


def get_2img_border_after_shift(dim_elem_num, voxel_border, xyz_shift):
    r'''
    Calculate the border of two partly overlapping images after translation.

    Parameters
    ----------
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    voxel_border - array, uint32, (2,6)
        the border indexes on three dimensions of two images.
        i.th dimension values range is [0,dim_elem_num[i]-1].
        specific form:
            [[x1_min,x1_max,y1_min,y1_max,z1_min,z1_max],
            [x2_min,x2_max,y2_min,y2_max,z2_min,z2_max]]
    xyz_shift - vector or list, int, (3)
        the voxel shift values of stitched image in XYZ dimensions.

    Returns
    ----------
    border_after_shift - array, uint32, (2,6)
        the border indexes on three dimensions of two images.
        i.th dimension values range is [0,dim_elem_num[i]-1].
        specific form:
            [[x1_min,x1_max,y1_min,y1_max,z1_min,z1_max],
            [x2_min,x2_max,y2_min,y2_max,z2_min,z2_max]]
    or---
    None -
        if the two tiles don't have overlapped area after shift.
    '''
    border = voxel_border.astype('int64')  # crucial step
    border_after_shift = np.zeros((2, 6), dtype='uint32')
    for i in range(3):
        if xyz_shift[i] < 0:
            if np.abs(xyz_shift[i]) >= voxel_border[0, 2 * i] + voxel_border[0, 2 * i + 1]:
                return np.array([])
            elif np.abs(xyz_shift[i]) >= voxel_border[0, 2 * i]:
                border_after_shift[0, 2 * i] = 0
                border_after_shift[0, 2 * i + 1] = border[0, 2 * i + 1] + border[0, 2 * i] + xyz_shift[i]
            else:
                border_after_shift[0, 2 * i] = voxel_border[0, 2 * i] + xyz_shift[i]
                border_after_shift[0, 2 * i + 1] = border[0, 2 * i + 1]
        else:
            if np.abs(xyz_shift[i]) >= dim_elem_num[i] - 1 - voxel_border[0, 2 * i] + dim_elem_num[i] - 1 - \
                    voxel_border[0, 2 * i + 1]:
                return np.array([])
            elif np.abs(xyz_shift[i]) >= dim_elem_num[i] - 1 - voxel_border[0, 2 * i + 1]:
                border_after_shift[0, 2 * i] = border[0, 2 * i] + xyz_shift[i] - (
                            dim_elem_num[i] - 1 - voxel_border[0, 2 * i + 1])
                border_after_shift[0, 2 * i + 1] = dim_elem_num[i] - 1
            else:
                border_after_shift[0, 2 * i] = border[0, 2 * i]
                border_after_shift[0, 2 * i + 1] = voxel_border[0, 2 * i + 1] + xyz_shift[i]
        border_after_shift[1, 2 * i] = dim_elem_num[i] - 1 - border_after_shift[0, 2 * i + 1]
        border_after_shift[1, 2 * i + 1] = dim_elem_num[i] - 1 - border_after_shift[0, 2 * i]
        # print(xyz_shift)
    # print(border_after_shift)
    return border_after_shift


def get_border_list(i, index_j, dim_elem_num, dim_len, voxel_len, tile_pos_stitch):
    r'''
    Get borders of i.th tile and all the tiles in index_j.

    Parameters
    ----------
    i - int
        the ordinal number of being stitched tile.
    index_j - list, int
        the ordinal number of tiles contact with i.th tile.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    tile_pos_stitch - vector/array/multiprocessing.Array, float64, (tile_num*3)
        the XYZ position of all tiles.

    Returns
    ----------
    border_list - list for array((2,6),dtype='uint32')
        all the border indexes between i.th and j.th tile for j in index_j.
        border specific form:
            [[x1_min,x1_max,y1_min,y1_max,z1_min,z1_max],
            [x2_min,x2_max,y2_min,y2_max,z2_min,z2_max]]
    '''
    border_list = []
    for k in index_j:
        border_list.append(get_2img_border(dim_elem_num, dim_len, voxel_len,
                                           np.vstack((np.array(tile_pos_stitch[3 * i:3 * i + 3]),
                                                      np.array(tile_pos_stitch[3 * k:3 * k + 3])))))
    return border_list

def get_ovl_img(img1, img2, border):
    '''
    Get overlapped image of img1 and img2 according to the border of their overlapped area.

    Parameters
    ----------
    img1,img2 - array, uint8/uint16/float32/float64, (dim_elem_num[0],dim_elem_num[1],dim_elem_num[2])
    voxel_border - array, uint32, (2,6)
        the border indexes on three dimensions of two images.
        specific form:
            [[x1_min,x1_max,y1_min,y1_max,z1_min,z1_max],
            [x2_min,x2_max,y2_min,y2_max,z2_min,z2_max]]

    Returns
    ----------
    ovl1,ovl2 - 2D array, uint8/uint16/float32/float64 according to image input, size depends on border
        the array to store overlapped area of img1 and img2
    '''
    if border.shape[0]==0:
        return np.array([]), np.array([])
    for i in range(3):
        for j in range(2):
            if border[j, 2 * i] == border[j, 2 * i + 1]:
                return np.array([]), np.array([])
    ovl1 = np.vstack((img1[border[0, 2]:border[0, 3], border[0, 0], border[0, 4]:border[0, 5]],
                      img1[border[0, 2]:border[0, 3], border[0, 1], border[0, 4]:border[0, 5]],
                      img1[border[0, 2], border[0, 0]:border[0, 1], border[0, 4]:border[0, 5]],
                      img1[border[0, 3], border[0, 0]:border[0, 1], border[0, 4]:border[0, 5]]))
    ovl2 = np.vstack((img2[border[1, 2]:border[1, 3], border[1, 0], border[1, 4]:border[1, 5]],
                      img2[border[1, 2]:border[1, 3], border[1, 1], border[1, 4]:border[1, 5]],
                      img2[border[1, 2], border[1, 0]:border[1, 1], border[1, 4]:border[1, 5]],
                      img2[border[1, 3], border[1, 0]:border[1, 1], border[1, 4]:border[1, 5]]))
    return ovl1, ovl2
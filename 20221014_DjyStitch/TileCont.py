import numpy as np


def judge_tile_cont(dim_len, tile_pos, tile_cont_thre=0.6):
    r'''
    Judge if two tiles contact with each other and return a bool array for all tiles.

    Parameters
    ----------
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile
    tile_cont_thre - float
        the thredshold to judge if two tile contact with each other. if two tiles align in X direction,
        they can be judged as contacted tiles when the overlap ratios in y and z directions exceed tile_cont_thre

    Returns
    ----------
    tile_contact - array, bool, (tile_num,tile_num)
        the tile contact array for all tiles. tile_contact[i,j] means whether i.th tile and j.th tile have overlapped region.
    '''
    tile_num = tile_pos.shape[0]
    tile_contact = np.zeros((tile_num, tile_num), dtype='bool')
    for i in range(tile_num):
        for j in range(tile_num):
            if i == j:
                tile_contact[i, j] = False
                continue
            if np.sum(np.abs(tile_pos[i, :] - tile_pos[j, :]) < dim_len * np.array(
                    [1, 1 - tile_cont_thre, 1 - tile_cont_thre])) == 3:
                tile_contact[i, j] = True
                continue
            if np.sum(np.abs(tile_pos[i, :] - tile_pos[j, :]) < dim_len * np.array(
                    [1 - tile_cont_thre, 1, 1 - tile_cont_thre])) == 3:
                tile_contact[i, j] = True
                continue
    return tile_contact


def judge_tile_cont_of_one(i, dim_len, tile_pos, tile_cont_thre=0.6):
    r'''
    Judge the contact tiles of i.th tile

    Parameters
    ----------
    i - int
        the ordinal number of one tile.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile
    tile_cont_thre - float
        the thredshold to judge if two tile contact with each other. if two tiles align in X direction,
        they can be judged as contacted tiles when the overlap ratios in y and z directions exceed tile_cont_thre

    Returns
    ----------
    tile_contact_list - vector, bool, (tile_num)
        the tile contact bool vector for this tile. tile_contact[j] means whether i.th tile and j.th tile have overlapped region.
    '''
    tile_num = tile_pos.shape[0]
    tile_contact_list = np.zeros(tile_num, dtype='bool')
    for j in range(tile_num):
        if i == j:
            tile_contact_list[j] = False
            continue
        if np.sum(np.abs(tile_pos[i, :] - tile_pos[j, :]) < dim_len * np.array([1, 1 - tile_cont_thre, 1 - tile_cont_thre])) == 3:
            tile_contact_list[j] = True
            continue
        if np.sum(np.abs(tile_pos[i, :] - tile_pos[j, :]) < dim_len * np.array([1 - tile_cont_thre, 1, 1 - tile_cont_thre])) == 3:
            tile_contact_list[j] = True
            continue
    return tile_contact_list


def choose_refe_tile(i, tile_contact_list, if_tile_stitched):
    r'''
    Choose reference tile for this tile when i.th tile is being stitched.
    Reference tile must be tile that has been stitched.
    After choosing the reference tile, this tile will be put next to the reference tile and update its position for subsequent stitch.

    Parameters
    ----------
    i - int
        the ordinal number of this tile.
    tile_contat_list - vector, bool, (tile_num)
        the tile contact bool vector for this tile. tile_contact[j] means whether i.th tile and j.th tile have overlapped region.
    if_tile_stitched - vector/list/multiprocessing.Array, bool, (tile_num)
        record if tile has been stitched.

    Returns
    ----------
    j - int
        the ordinal number of reference tile.
    or---
    -1
        if this tile doesn't have any reference tile.
    '''
    index_j = []
    for j in range(len(if_tile_stitched)):
        if tile_contact_list[j] and if_tile_stitched[j]:
            index_j.append(j)
    if len(index_j) == 0:
        return -1
    index_j = np.array(index_j, dtype='int')
    dis_ij = np.abs(index_j - i)
    max_dis_index = np.argwhere(dis_ij == np.max(dis_ij))
    j = index_j[max_dis_index[0, 0]]
    return j

def choose_refe_tile_for_stit(tile_contact_list,if_tile_stitched):
    '''
    Choose all stitched tiles contact to the being stitched tile and return an ordinal list of them.

    Parameters
    ----------
    tile_contact_list - vector, bool, (tile_num)
        the tile contact bool vector for this tile. tile_contact[j] means whether i.th tile and j.th tile have overlapped region.
    if_tile_stitched - vector/list/multiprocessing.Array, bool, (tile_num)
        record if tile has been stitched.
    '''
    index_j=[]
    for j in range(len(if_tile_stitched)):
        if tile_contact_list[j] and if_tile_stitched[j]:
            index_j.append(j)
    if len(index_j)==0:
        return -1
    return index_j
import numpy as np


def loss_func(ovl1, ovl2, xyz_shift, alpha=0):
    r'''
    Calculate the loss between ovl1 and ovl2.

    Parameters
    ----------
    ovl1,ovl2 - 2D arary, float32/float64
        overlapped area array of two images.
    xyz_shift - vector or list, int, (3)
        the voxel shift values of stitched image in XYZ dimensions.
    alpha - float
        one hyperparameter to avoid images shift too far from actual correct position.

    Returns
    ----------
    loss - float
        the loss between ovl1 and ovl2.
    '''
    loss = np.sum((ovl1 - ovl2) ** 2) / np.sqrt(np.sum(ovl1 ** 2) * np.sum(ovl2 ** 2)) + alpha * np.sum(
        np.array(xyz_shift).astype('float32') ** 2)
    return loss

def loss_func_z_stitch(ovl1, ovl2):
    ovl1, ovl2 = ovl1.astype('float32'), ovl2.astype('float32')
    loss = np.sum((ovl1 - ovl2) ** 2) / np.sqrt(np.sum(ovl1 ** 2) * np.sum(ovl2 ** 2))
    return loss


def loss_func_for_list(ovl1_list, ovl2_list, xyz_shift, alpha=0):
    r'''
    Calculate and sum the loss between ovl1 and ovl2 for ovl1 in ovl1_list and ovl2 in ovl2_list.

    Parameters
    ----------
    ovl1_list,ovl2_list - list for 2D arary(float32/float64)
        overlapped area array list between stitched image with contact images
    xyz_shift - vector or list, int, (3)
        the voxel shift values of stitched image in XYZ dimensions.
    alpha - float
        one hyperparameter to avoid images shift too far from actual correct position.

    Returns
    ----------
    loss - float
        the loss sum between ovl1_list and ovl2_list.
    '''
    ovl_num = len(ovl1_list)
    a, b, c = 0, 0, 0
    for i in range(ovl_num):
        a = a + np.sum(ovl1_list[i] ** 2)
        b = b + np.sum(ovl2_list[i] ** 2)
    if a == 0 or b == 0:
        # print('no overlapping area')
        return np.inf
    for i in range(ovl_num):
        c = c + np.sum((ovl1_list[i] - ovl2_list[i]) ** 2)
    loss = c / np.sqrt(a * b) + alpha * np.sum(np.array(xyz_shift).astype('float32') ** 2)
    return loss
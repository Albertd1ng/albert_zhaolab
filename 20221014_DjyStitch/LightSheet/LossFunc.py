import numpy as np

def loss_func(ovl1, ovl2, xyz_shift=[0,0,0], alpha=0):
    r'''
    Calculate the loss between ovl1 and ovl2.

    :param ovl1: numpy.array(3D, dtype='float')
    :param ovl2: numpy.array(3D, dtype='float')
        overlapped area array of two images.
    :param xyz_shift: list or numpy.array(2, dtype='int')
        the voxel shift values of stitched image in XYZ dimensions.
    :param alpha: float
        one hyperparameter to avoid images shift too far from actual correct position.
    :return loss: float
        the loss between ovl1 and ovl2.
    '''
    ovl1, ovl2 = ovl1.astype('float32'), ovl2.astype('float32')
    a = np.sum((ovl1 - ovl2) ** 2)
    b = np.sqrt(np.sum(ovl1 ** 2) * np.sum(ovl2 ** 2))
    if b == 0:
        return np.inf
    loss = a / b + alpha * np.sum(np.array(xyz_shift).astype('float32') ** 2)
    return loss


def loss_func_for_list(ovl1_list, ovl2_list, xyz_shift, alpha=0):
    ovl_num = len(ovl1_list)
    if ovl_num == 0:
        return np.inf
    a, b, c = 0, 0, 0
    for i in range(ovl_num):
        ovl1, ovl2 = ovl1_list[i].astype('float32'), ovl2_list[i].astype('float32')
        a = a + np.sum(ovl1 ** 2)
        b = b + np.sum(ovl2 ** 2)
        c = c + np.sum((ovl1 - ovl2) ** 2)
    if a == 0 or b == 0:
        return np.inf
    loss = c / np.sqrt(a * b) + alpha * np.sum(np.array(xyz_shift).astype('float32') ** 2)
    return loss

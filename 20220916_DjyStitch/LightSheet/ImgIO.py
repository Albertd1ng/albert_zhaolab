import os
import cv2
import numpy as np


def import_img_2D_one(img_name_format, img_path, z_th, ch_th, img_type='tif'):
    '''
    Import one 2D image of one strip.

    :param img_name_format: str.
        example: '%s_%s'%(z_th,ch_th).
    :param img_path: str.
    :param z_th: int.
    :param ch_th: int.
        the chosen channel ordinal.
    :param img_type: str.
        'tif' et al.
    :return: numpy.array(2D, dtype='uint8' or 'uint16').
    '''
    this_img_name = os.path.join(img_path, ''.join((img_name_format % (z_th, ch_th), '.', img_type)))
    return cv2.imread(this_img_name, cv2.IMREAD_UNCHANGED)


def import_img_3D_section(img_name_format, img_path, z_range, ch_th, xy_v_num, img_type='tif', img_dtype='uint16'):
    '''
    Import 3D section image of one strip, the size in z dimension is limited by z_range.

    :param img_name_format: str.
        example: '%s_%s'%(z_th,ch_th).
    :param img_path: str.
    :param z_range: np.array(2, dtype='int64') or list(2, int).
        the start and end index in z dimension of this section.
    :param ch_th: int.
        the chosen channel ordinal.
    :param xy_v_num: np.array(2, dtype='int64').
        the voxel quantity in xy dimensions.
    :param img_type: str.
        'tif' et al.
    :param img_dtype: str
        'uint8' or 'uint16'.
    :return: numpy.array((xy_v_num[1], xy_v_num[0], z_range[1]-z_range[0]), dtype=img_dtype).
        3D numpy array to restore one section of a strip.
    '''
    strip_section = np.zeros((xy_v_num[1], xy_v_num[0], z_range[1]-z_range[0]), dtype=img_dtype)
    for z in range(z_range[0],z_range[1]):
        this_img_name = os.path.join(img_path, ''.join((img_name_format % (z, ch_th), '.', img_type)))
        strip_section[:,:,z-z_range[0]] = cv2.imread(this_img_name, cv2.IMREAD_UNCHANGED)
    return strip_section
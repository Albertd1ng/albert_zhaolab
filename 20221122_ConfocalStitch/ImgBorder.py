import numpy as np


def get_2img_border(dim_elem_num, dim_len, voxel_len, tile_pos):
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
                                 [xv2_min, xv2_max, yv2_min, yv2_max, zv2_min, zv2_max]], dtype='int64')
        return voxel_border
    else:
        return np.array([])


def get_border_pyr_down(border, down_multi):
    if border.shape[0] == 0:
        return np.array([])
    border_pyr_down = np.zeros((2,6), dtype='int64')
    border_pyr_down[:, 0:2] = np.int64(np.round(border[:, 0:2] * down_multi[0]))-1
    border_pyr_down[:, 2:4] = np.int64(np.round(border[:, 2:4] * down_multi[1]))-1
    border_pyr_down[:, 4:6] = np.int64(np.round(border[:, 4:6] * down_multi[2]))-1
    for i in range(2):
        for j in range(6):
            if border_pyr_down[i, j] < 0:
                border_pyr_down[:, j] = border_pyr_down[:, j] + 1
    for i in range(3):
        border_pyr_down[1, 2 * i + 1] = border_pyr_down[1, 2 * i] + border_pyr_down[0, 2 * i + 1] - border_pyr_down[
            0, 2 * i]
    return border_pyr_down

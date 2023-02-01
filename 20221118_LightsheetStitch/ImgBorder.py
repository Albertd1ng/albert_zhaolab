import numpy as np


def get_two_strip_border(strip_pos, xy_v_num, z_v_num):
    x_min, x_max = np.max(strip_pos[:, 0]), np.min(strip_pos[:, 0]) + xy_v_num[0] - 1
    y_min, y_max = np.max(strip_pos[:, 1]), np.min(strip_pos[:, 1]) + xy_v_num[1] - 1
    z_min, z_max = np.max(strip_pos[:, 2]), np.min(strip_pos[:, 2] + z_v_num - 1)
    xv1_min, xv1_max = x_min - strip_pos[0, 0], x_max - strip_pos[0, 0]
    yv1_min, yv1_max = y_min - strip_pos[0, 1], y_max - strip_pos[0, 1]
    zv1_min, zv1_max = z_min - strip_pos[0, 2], z_max - strip_pos[0, 2]
    xv2_min, xv2_max = x_min - strip_pos[1, 0], x_max - strip_pos[1, 0]
    yv2_min, yv2_max = y_min - strip_pos[1, 1], y_max - strip_pos[1, 1]
    zv2_min, zv2_max = z_min - strip_pos[1, 2], z_max - strip_pos[1, 2]
    if (0 <= xv1_min < xv1_max < xy_v_num[0] and 0 <= xv2_min < xv2_max < xy_v_num[0] and
            0 <= yv1_min < yv1_max < xy_v_num[1] and 0 <= yv2_min < yv2_max < xy_v_num[1] and
            0 <= zv1_min < zv1_max < z_v_num[0] and 0 <= zv2_min < zv2_max < z_v_num[1]):
        border = np.array([[xv1_min, xv1_max, yv1_min, yv1_max, zv1_min, zv1_max],
                           [xv2_min, xv2_max, yv2_min, yv2_max, zv2_min, zv2_max]], dtype='int64')
        return border
    else:
        return np.array([])


def get_two_strip_sec_border(strip_pos_sec, xy_v_num, z_depth):
    strip_pos = np.array(strip_pos_sec, dtype='int64').reshape(-1, 3)
    x_min, x_max = np.max(strip_pos[:, 0]), np.min(strip_pos[:, 0]) + xy_v_num[0] - 1
    y_min, y_max = np.max(strip_pos[:, 1]), np.min(strip_pos[:, 1]) + xy_v_num[1] - 1
    z_min, z_max = np.max(strip_pos[:, 2]), np.min(strip_pos[:, 2] + z_depth - 1)
    xv1_min, xv1_max = x_min - strip_pos[0, 0], x_max - strip_pos[0, 0]
    yv1_min, yv1_max = y_min - strip_pos[0, 1], y_max - strip_pos[0, 1]
    zv1_min, zv1_max = z_min - strip_pos[0, 2], z_max - strip_pos[0, 2]
    xv2_min, xv2_max = x_min - strip_pos[1, 0], x_max - strip_pos[1, 0]
    yv2_min, yv2_max = y_min - strip_pos[1, 1], y_max - strip_pos[1, 1]
    zv2_min, zv2_max = z_min - strip_pos[1, 2], z_max - strip_pos[1, 2]
    if (0 <= xv1_min < xv1_max < xy_v_num[0] and 0 <= xv2_min < xv2_max < xy_v_num[0] and
            0 <= yv1_min < yv1_max < xy_v_num[1] and 0 <= yv2_min < yv2_max < xy_v_num[1] and
            0 <= zv1_min < zv1_max < z_depth and 0 <= zv2_min < zv2_max < z_depth):
        border = np.array([[xv1_min, xv1_max, yv1_min, yv1_max, zv1_min, zv1_max],
                           [xv2_min, xv2_max, yv2_min, yv2_max, zv2_min, zv2_max]], dtype='int64')
        return border
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


def get_z_border_depth(border_list):
    z_depth_min = np.inf
    for border in border_list:
        if (z_depth := border[0, 5] - border[0, 4]) < z_depth_min:
            z_depth_min = z_depth
    return z_depth_min

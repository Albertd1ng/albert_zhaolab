import numpy as np
import os


def judge_tile_cont(dim_len, tile_pos, tile_cont_thre=0.8):
    tile_num = tile_pos.shape[0]
    tile_contact = np.zeros((tile_num, tile_num), dtype='bool')
    for i in range(tile_num):
        for j in range(tile_num):
            if i == j:
                tile_contact[i, j] = False
                continue
            if i > j:
                continue
            if np.sum(np.abs(tile_pos[i, :] - tile_pos[j, :]) < dim_len * np.array(
                    [1, 1 - tile_cont_thre, 1 - tile_cont_thre])) == 3:
                tile_contact[i, j] = True
                tile_contact[j, i] = True
                continue
            if np.sum(np.abs(tile_pos[i, :] - tile_pos[j, :]) < dim_len * np.array(
                    [1 - tile_cont_thre, 1, 1 - tile_cont_thre])) == 3:
                tile_contact[i, j] = True
                tile_contact[j, i] = True
                continue
    return tile_contact


def get_save_tile_cont_info(layer_num, save_path, dim_len, tile_pos, tile_cont_thre=0.8):
    save_path_file = os.listdir(save_path)
    if r'tile_contact_%.4d.npy' % (layer_num) in save_path_file:
        return np.load(os.path.join(save_path, 'tile_contact_%.4d.npy' % (layer_num)))
    else:
        tile_contact = judge_tile_cont(dim_len, tile_pos, tile_cont_thre=tile_cont_thre)
        np.save(os.path.join(save_path, 'tile_contact_%.4d.npy' % (layer_num)), tile_contact)
        return tile_contact

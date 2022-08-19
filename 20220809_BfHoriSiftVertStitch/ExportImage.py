#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
import cv2
import random
from SiftVertStitch import *

def get_max_xy_axis_range(save_path,voxel_len,max_index=0,border_num=800):
    try:
        xy_axis_range=np.load(save_path+r'\max_xy_axis_range.npy')
    except:
        axis_range=np.load(save_path+r'\axis_range_zstitch_%.4d.npy'%(max_index))#3,2
        xy_axis_range=np.zeros((2,2))
        xy_axis_range[0,0],xy_axis_range[0,1]=axis_range[0,0]-border_num*voxel_len[0],axis_range[0,1]+border_num*voxel_len[0]
        xy_axis_range[1,0],xy_axis_range[1,1]=axis_range[1,0]-border_num*voxel_len[1],axis_range[1,1]+border_num*voxel_len[1]
        np.save(save_path+r'\max_xy_axis_range.npy',xy_axis_range)
    return xy_axis_range

def start_export_one_layer(i,img_path_file,img_name,save_path,img_save_path,img_num):
    dim_elem_num,dim_len,voxel_len=get_img_info(i,save_path)
    axis_range=np.load(save_path+r'\axis_range_zstitch_%.4d.npy'%(i))
    xy_axis_range=get_max_xy_axis_range(save_path,voxel_len)
    xy_voxel_num=np.int64(np.round((xy_axis_range[:,1]-xy_axis_range[:,0])/voxel_len[0:2])+1)
    first_last_index=np.load(save_path+r'\first_last_index_zstitch_%.4d.npy'%(i))
    tile_pos=np.load(save_path+r'\tile_pos_zstitch_%.4d.npy'%(i))
    tile_num=tile_pos.shape[0]
    for j in range(first_last_index[0],first_last_index[1]):
        this_img=np.zeros((xy_voxel_num[1::-1]),dtype='uint8')
        this_z=axis_range[2,0]+voxel_len[2]*j
        for k in range(tile_num):
            if tile_pos[k,2]+dim_len[2]<this_z:
                continue
            if tile_pos[k,2]>this_z:
                continue
            z_th=np.int64(np.round((this_z-tile_pos[k,2])/voxel_len[2]))
            if z_th>=dim_elem_num[2]:
                continue
            x_th=np.int64(np.round((tile_pos[k,0]-xy_axis_range[0,0])/voxel_len[0]))
            y_th=np.int64(np.round((tile_pos[k,1]-xy_axis_range[1,0])/voxel_len[1]))
            img_2D=import_2D_img(img_path_file,img_name,k,z_th)
            if x_th<0 or x_th+dim_elem_num[0]>xy_voxel_num[0] or y_th<0 or y_th+dim_elem_num[1]>xy_voxel_num[1]:
                continue
            this_img[y_th:y_th+dim_elem_num[1],x_th:x_th+dim_elem_num[0]]=img_2D
        cv2.imwrite(r'%s\a%.4d.tif'%(img_save_path,img_num.value),this_img)
        img_num.value+=1
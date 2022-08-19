#!/usr/bin/env python
# coding: utf-8

# In[3]:


import BfHoriStitch
import SiftVertStitch
import ExportImage
import multiprocessing
from multiprocessing import Value
import cv2


#python C:\Users\dingj\ZhaoLab\20220809_BfHoriSiftVertStitch\StartAll.py
if __name__=='__main__':
    ##########################################################################################
    img_name=r'TileScan 1'
    img_path=r'F:'
    file_name='Whole_%.4d'
    save_path=r'C:\Users\dingj\ZhaoLab\20220809_BfHoriSiftVertStitch'
    img_save_path=r'F:\Whole'
    layer_num=5
    ##########################################################################################
    img_num=Value('i',0)
    for i in range(3,layer_num+3):
        if 0<=i-2<layer_num-1:
            vert_pro=multiprocessing.Process(target=SiftVertStitch.start_vertical_stitch,args=(i-2,img_path,save_path,file_name,img_name))
            vert_pro.start()
        else:
            vert_pro=None
         
        if 0<=i-3<layer_num:
            img_path_file=img_path+'\\'+file_name%(i-3)
            export_pro=multiprocessing.Process(target=ExportImage.start_export_one_layer,args=(i-3,img_path_file,img_name,save_path,img_save_path,img_num))
            export_pro.start()
        else:
            export_pro=None
        
        if 0<=i<layer_num:
            xml_path=img_path+'\\'+file_name%(i)+r'\MetaData'+'\\'+img_name+'.xml'
            img_path_file=img_path+'\\'+file_name%(i)
            BfHoriStitch.start_multi_stitchers(i,xml_path,img_path_file,img_name,save_path,bad_tile_threshold=1.5,voxel_range=[21,21,9],step=3)
        if vert_pro!=None:
            vert_pro.join()
        if export_pro!=None:
            export_pro.join()


#!/usr/bin/env python
# coding: utf-8

# In[2]:


from lxml import etree
import numpy as np
import time
import cv2
import multiprocessing
from multiprocessing import Value,Array,Process
import ctypes
import random
from numba import jit,njit
import sys

#functions
def get_img_xml_info(xml_path):
    '''
    Read xml file and extract the information of dimensions and each tile
    return - (1)dim_elem_num - linspace(uint), the quantity of voxels for each dimension,
    (2)dim_len - linspace(float), the length for one 3D image,
    (3)voxel_len - lingspace(float), the length for a voxel,
    (4)tile_num - int, the quantity of tiles,
    (5)tile_field - array(uint), the identifier of each file
    (6)tile_pos - array(float), the XYZ position information of each tile
    '''
    parser=etree.XMLParser()
    my_et=etree.parse(xml_path,parser=parser)
    dim_attrib=my_et.xpath('//Dimensions/DimensionDescription')
    dim_elem_num=np.zeros(3,dtype='uint')
    dim_len=np.zeros(3)
    for i in range(3):
        dim_elem_num[i],dim_len[i]=dim_attrib[i].attrib['NumberOfElements'],dim_attrib[i].attrib['Length']
    voxel_len=dim_len/dim_elem_num
    tile_attrib=my_et.xpath('//Attachment/Tile')
    tile_num=len(tile_attrib)
    tile_field=np.zeros((tile_num,2),dtype='uint')
    tile_pos=np.zeros((tile_num,3))
    for i in range(tile_num):
        tile_field[i,:]=[tile_attrib[i].attrib['FieldX'],tile_attrib[i].attrib['FieldY']]
        tile_pos[i,:]=[tile_attrib[i].attrib['PosX'],tile_attrib[i].attrib['PosY'],0]
    return dim_elem_num,dim_len,voxel_len,tile_num,tile_field,tile_pos

def judge_tile_contact(dim_len,tile_pos):
    '''
    judge if two tiles contact with each other
    return - 3Darray, the XY contact array for each two images
    dim_len - linspace(float), the length for one 3D image
    tile_pos - array(float), the XYZ position information of each tile
    '''
    tile_num=tile_pos.shape[0]
    tile_contact=np.zeros((tile_num,tile_num),dtype='bool')
    for i in range(tile_num):
        for j in range(tile_num):
            if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*np.array([1,0.3,0.3]))==3:
                tile_contact[i,j]=True
            if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*np.array([0.3,1,0.3]))==3:
                tile_contact[i,j]=True
            if i==j:
                tile_contact[i,j]=False
    return tile_contact

def import_img(img_path,ordinal,dim_elem_num):
    '''
    this function reads voxel information and return a 3D np_array.
    return - array, store the 3D image
    img_path - str, the file position,
    ordinal - int, the ordinal number for image,
    dim_elem_num - list, the quantities of voxels for each dimension.
    '''
    voxel_array=np.zeros(tuple(dim_elem_num),dtype='uint8')#the array for storing image, dtyte should be changed according to image type
    #next statements get the img information according to image names, need to be changed according to different naming methods
    for i in range(dim_elem_num[2]):
        img_name=r'%s\Region 1_s%.4d_z%.3d_RAW_ch00.tif'%(img_path,ordinal,i)
        voxel_array[:,:,i]=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    return voxel_array

def get_2img_border(dim_elem_num,dim_len,voxel_len,tile_pos):
    '''
    get the border voxel index for two overlapping images
    return - array, the x/y/z_min/max voxel ordinal for each image,
    dim_elem_num - list, the quantities of voxels for each dimension,
    dim_len - list, the image length,
    tile_pos - array, xyz positions of each img.
    '''
    #x/y/z_min/max, the positions of overlapping image border
    x_min,x_max=np.max(tile_pos[:,0]),np.min(tile_pos[:,0])+dim_len[0]
    y_min,y_max=np.max(tile_pos[:,1]),np.min(tile_pos[:,1])+dim_len[1]
    z_min,z_max=np.max(tile_pos[:,2]),np.min(tile_pos[:,2])+dim_len[2]
    #x/y/zv_min/max, the voxel index of overlapping image border
    xv1_min,xv1_max=np.round((x_min-tile_pos[0,0])/voxel_len[0]),np.round((x_max-tile_pos[0,0])/voxel_len[0])
    yv1_min,yv1_max=np.round((y_min-tile_pos[0,1])/voxel_len[1]),np.round((y_max-tile_pos[0,1])/voxel_len[1])
    zv1_min,zv1_max=np.round((z_min-tile_pos[0,2])/voxel_len[2]),np.round((z_max-tile_pos[0,2])/voxel_len[2])
    xv2_min,xv2_max=np.round((x_min-tile_pos[1,0])/voxel_len[0]),np.round((x_max-tile_pos[1,0])/voxel_len[0])
    yv2_min,yv2_max=np.round((y_min-tile_pos[1,1])/voxel_len[1]),np.round((y_max-tile_pos[1,1])/voxel_len[1])
    zv2_min,zv2_max=np.round((z_min-tile_pos[1,2])/voxel_len[2]),np.round((z_max-tile_pos[1,2])/voxel_len[2])
    voxel_border=np.array([[xv1_min,xv1_max,yv1_min,yv1_max,zv1_min,zv1_max],
              [xv2_min,xv2_max,yv2_min,yv2_max,zv2_min,zv2_max]],dtype='uint')
    return voxel_border

def get_2img_border_after_shift(dim_elem_num,voxel_border,xyz_shift):
    '''
    this function calculates the border of two partly overlapping images after translation
    return - array, the voxel index of overlapping area for each image
    dim_elem_num - the voxel quantities for each dimension
    voxel_border - array, the voxel index before translation
    xyz_shift - list, the translation for each dimension
    '''
    border=voxel_border.astype('int')#crucial step
    border_after_shift=np.zeros((2,6),dtype='uint')
    for i in range(3):
        if xyz_shift[i]<0:
            if abs(xyz_shift[i])>voxel_border[0,2*i]:
                border_after_shift[0,2*i]=0
                border_after_shift[0,2*i+1]=np.max([0,border[0,2*i+1]+xyz_shift[i]])
            else:
                border_after_shift[0,2*i]=voxel_border[0,2*i]+xyz_shift[i]
                border_after_shift[0,2*i+1]=dim_elem_num[i]
        else:
            if abs(xyz_shift[i])>dim_elem_num[i]-voxel_border[0,2*i+1]:
                border_after_shift[0,2*i]=np.min([dim_elem_num[i],border[0,2*i]+xyz_shift[i]-dim_elem_num[i]+voxel_border[0,2*i+1]])
                border_after_shift[0,2*i+1]=dim_elem_num[i]
            else:
                border_after_shift[0,2*i]=0
                border_after_shift[0,2*i+1]=voxel_border[0,2*i+1]+xyz_shift[i]
        border_after_shift[1,2*i]=dim_elem_num[i]-border_after_shift[0,2*i+1]
        border_after_shift[1,2*i+1]=dim_elem_num[i]-border_after_shift[0,2*i]           
    #print(xyz_shift)
    #print(border_after_shift)
    return border_after_shift

def choose_reference_tile(tile_contact_list,if_tile_stitched):
    '''
    choose best reference tile for i.th tile
    return - tuple, int (2).
    tile_contact_array - array, bool (2,n).
    if tile_stitched - list (n).
    '''
    index_j=[]
    for j in range(len(if_tile_stitched)):
        if tile_contact_list[j] and if_tile_stitched[j]:
            index_j.append(j)
    if len(index_j)==0:
        return -1
    return index_j[random.randint(0,len(index_j)-1)]

def adjust_contrast(img1,img2,border):
    '''
    return - the img1 img2 which have been adjusted contrast
    img1, img2 - array, float64.
    border - array, (2,3), the overlapping area border between img1 and img2
    '''
    img1,img2=cv2.medianBlur(img1,5),cv2.medianBlur(img2,5)
    #img1,img2=cv2.medianBlur(img1,5),cv2.medianBlur(img2,5)
    img1,img2=img1.astype('float64'),img2.astype('float64')
    ovl1=img1[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]]
    ovl2=img2[border[1,2]:border[1,3],border[1,0]:border[1,1],border[0,4]:border[0,5]]
    m1,m2=np.mean(ovl1),np.mean(ovl2)
    #print(m1,m2)
    if m1==0 or m2==0:
        return np.array([]),np.array([])
    elif m1<0.1 or m2<0.1:
        img1,img2=1/m1*img1,1/m2*img2
    elif m1<1 or m2<1:
        img1,img2=5/m1*img1,5/m2*img2
    elif m1<5 or m2<5:
        img1,img2=10/m1*img1,10/m2*img2
    elif m1<10 or m2<10:
        img1,img2=20/m1*img1,20/m2*img2
    elif m1<20 or m2<20:
        img1,img2=30/m1*img1,30/m2*img2
    elif m1<30 or m2<30:
        img1,img2=40/m1*img1,40/m2*img2
    elif np.abs(m1-m2)>5:
        m_max=np.max(np.array([m1,m2]))
        img1,img2=m_max/m1*img1,m_max/m2*img2
    img1,img2=np.clip(img1,0,255),np.clip(img2,0,255)
    return img1,img2

def get_overlapping_img(img1,img2,border):
    '''
    return - the overlapping 2D area between img1 and img2
    img1, img2 - array
    border - array, (2,3)
    '''
    for i in range(3):
        for j in range(2):
            if border[j,2*i]==border[j,2*i+1]:
                return np.array([]),np.array([])
    ovl1=np.vstack((img1[border[0,2]:border[0,3],border[0,0],border[0,4]:border[0,5]],
                    img1[border[0,2]:border[0,3],border[0,1]-1,border[0,4]:border[0,5]],
                    img1[border[0,2],border[0,0]:border[0,1],border[0,4]:border[0,5]],
                    img1[border[0,3]-1,border[0,0]:border[0,1],border[0,4]:border[0,5]]))
    ovl2=np.vstack((img2[border[1,2]:border[1,3],border[1,0],border[1,4]:border[1,5]],
                    img2[border[1,2]:border[1,3],border[1,1]-1,border[1,4]:border[1,5]],
                    img2[border[1,2],border[1,0]:border[1,1],border[1,4]:border[1,5]],
                    img2[border[1,3]-1,border[1,0]:border[1,1],border[1,4]:border[1,5]]))
    return ovl1,ovl2
    
@njit(nogil=True)
def loss_fun(ovl1,ovl2,xyz_shift,alpha):
    '''
    this fucntion calculate the loss function of two overlapping images
    return - float, loss fun of overlapping area
    xyz_shift - list, record the xyz shift of original position,
    alpha - a hyperparameter
    '''
    if 0 in ovl1.shape or 0 in ovl2.shape:
        print('no overlapping area')
        return np.inf
    if np.sum(ovl1)==0 or np.sum(ovl2)==0:
        return np.inf
    a=np.sum((ovl1-ovl2)**2)/np.sqrt(np.sum(ovl1**2)*(np.sum(ovl2**2)))+alpha*np.sum(xyz_shift**2)
    return a

def calculate_xyz_shift(i,j,img_path,dim_elem_num,voxel_border,voxel_range,step):
    '''
    #this function gets the best overlapping position for two partly overlapping images
    #return - list, the voxel translation quantities of img1 to be stitched to img2
    #voxel_border - array, voxel range for overlapping area,
    #voxel_range - list, the range need to be calculated,
    #step - int 
    '''
    img1,img2=import_img(img_path,i,dim_elem_num),import_img(img_path,j,dim_elem_num)
    img1,img2=adjust_contrast(img1,img2,voxel_border)#float64
    if 0 in img1.shape:
        return [0,0,0],np.inf
    xv_shift,yv_shift,zv_shift=0,0,0
    loss_min=np.inf
    alpha=5*10e-5
    for x in range(-voxel_range[0],voxel_range[0]+1,step):
        for y in range(-voxel_range[1],voxel_range[1]+1,step):
            for z in range(-voxel_range[2],voxel_range[2]+1,step):
                border=get_2img_border_after_shift(dim_elem_num,voxel_border,[x,y,z])
                ovl1,ovl2=get_overlapping_img(img1,img2,border)
                this_loss=loss_fun(ovl1,ovl2,np.array([x,y,z],dtype='float64'),alpha)
                #print((x,y,z),this_loss)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    for x in range(xv_shift-2*step+1,xv_shift+2*step):
        for y in range(yv_shift-2*step+1,yv_shift+2*step):
            for z in range(zv_shift-step+1,zv_shift+step):
                border=get_2img_border_after_shift(dim_elem_num,voxel_border,[x,y,z])
                ovl1,ovl2=get_overlapping_img(img1,img2,border)
                this_loss=loss_fun(ovl1,ovl2,np.array([x,y,z],dtype='float64'),alpha)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    #print(xv_shift,yv_shift,zv_shift,loss_min)
    return [xv_shift,yv_shift,zv_shift],loss_min

def choose_start_img(img_path,tile_num,dim_elem_num,previous=0):
    m,a=0,0
    while(m<10 or abs(a-previous)<0.35*round(tile_num)):
        a=random.randint(int(0.1*tile_num),int(0.9*tile_num))
        m=np.mean(import_img(img_path,a,dim_elem_num))
    print('start_tile is %d.th tile, average is %.8f'%(a,m))
    return a

def run_sift_stitcher(lock,img_path,dim_elem_num,dim_len,voxel_len,tile_num,tile_field,tile_pos,tile_contact,
                      if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_being_stitched,
                      tile_pos_index,tile_pos_stitch,voxel_range,step):
    stitch_num=0
    this_name=multiprocessing.current_process().name
    while(False in if_tile_stitched):
        lock.acquire()
        usable_tile_index=[index for index,value in enumerate(zip(if_tile_stitched,if_tile_shelved,if_tile_being_stitched)) if not any(value)]
        if len(usable_tile_index)==0:
            for i in range(tile_num):
                if_tile_shelved[i]=False
            time.sleep(5)
            lock.release()
            print('All shelved tile has been released')
            continue
        i=usable_tile_index[random.randint(0,len(usable_tile_index)-1)]
        j=choose_reference_tile(tile_contact[i,:],if_tile_stitched)
        if j==-1:
            if_tile_shelved[i]=True
            lock.release()
            #print('%d.th tile has no appropriate contact tile'%(i))
            continue
        if_tile_being_stitched[i]=True
        lock.release()
        voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,tile_pos[[i,j],:])
        xyz_shift,loss_min=calculate_xyz_shift(i,j,img_path,dim_elem_num,voxel_border,voxel_range,step)
        lock.acquire()
        if_tile_being_stitched[i]=False
        if loss_min>2:
            if_tile_bad[i]=if_tile_bad[i]+1
            if if_tile_bad[i]>=0.3:
                if_tile_stitched[i]=True
                tile_pos_index[i]=j
                tile_pos_stitch[3*i:3*i+3]=[0,0,0]
            else:
                if_tile_shelved[i]=True
            print('%d.th tile is a bad tile, times is %d'%(i,if_tile_bad[i]))
        else:
            if_tile_stitched[i]=True
            tile_pos_index[i]=j
            tile_pos_stitch[3*i:3*i+3]=xyz_shift
        lock.release()
        stitch_num+=1
        print('%s has stitched %d tiles, current stitch is %d.th tile with %d.th tile,\nxyz_shift is (%d, %d, %d), loss is %.8f'
              %(this_name,stitch_num,i,j,*xyz_shift,loss_min))
    print('!!!!!!%s stops and has stitched %d tiles.!!!!!!'%(this_name,stitch_num))
    
def start_multi_stitchers(xml_path,img_path,save_path,voxel_range=[21,21,9],step=3):
    dim_elem_num,dim_len,voxel_len,tile_num,tile_field,tile_pos=get_img_xml_info(xml_path)
    tile_contact=judge_tile_contact(dim_len,tile_pos)
    lock=multiprocessing.RLock()
    ######################################################################################
    if_tile_stitched=Array(ctypes.c_bool,[False for i in range(tile_num)])
    start_index=choose_start_img(img_path,tile_num,dim_elem_num,0)
    if_tile_stitched[start_index]=True
    if_tile_being_stitched=Array(ctypes.c_bool,[False for i in range(tile_num)])
    if_tile_shelved=Array(ctypes.c_bool,[False for i in range(tile_num)])
    if_tile_bad=Array('i',[0 for i in range(tile_num)])
    tile_pos_index=Array('i',[-1 for i in range(tile_num)])
    tile_pos_stitch=Array('i',[0 for i in range(tile_num*3)])
    process_num=round(0.3*multiprocessing.cpu_count())
    print('Current processing quantities: %d'%(process_num))
    process_list=[]
    for i in range(process_num):
        one_pro=multiprocessing.Process(target=run_sift_stitcher,
                                        args=(lock,img_path,dim_elem_num,dim_len,voxel_len,tile_num,tile_field,tile_pos,tile_contact,
                                             if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_being_stitched,
                                              tile_pos_index,tile_pos_stitch,voxel_range,step))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    tile_pos_stitch=np.array(tile_pos_stitch).reshape(tile_num,3)
    tile_pos_index=np.array(tile_pos_index,dtype='int')
    print('start saving_data')
    np.save(save_path+r'\tile_pos_index1.npy',tile_pos_index)
    np.save(save_path+r'\tile_pos_stitch1.npy',tile_pos_stitch)
    tile_pos1=update_pos(tile_pos,tile_pos_stitch,tile_pos_index,voxel_len)
    np.save(save_path+r'\tile_pos1.npy',tile_pos1)
    print('end saving data')
    ####################################################################################
    tile_contact=judge_tile_contact(dim_len,tile_pos1)
    if_tile_stitched=Array(ctypes.c_bool,[False for i in range(tile_num)])
    start_index=choose_start_img(img_path,tile_num,dim_elem_num,start_index)
    if_tile_stitched[start_index]=True
    if_tile_being_stitched=Array(ctypes.c_bool,[False for i in range(tile_num)])
    if_tile_shelved=Array(ctypes.c_bool,[False for i in range(tile_num)])
    if_tile_bad=Array('i',[0 for i in range(tile_num)])
    tile_pos_index=Array('i',[-1 for i in range(tile_num)])
    tile_pos_stitch=Array('i',[0 for i in range(tile_num*3)])
    print('Current processing quantities: %d'%(process_num))
    process_list=[]
    for i in range(process_num):
        one_pro=multiprocessing.Process(target=run_sift_stitcher,
                                        args=(lock,img_path,dim_elem_num,dim_len,voxel_len,tile_num,tile_field,tile_pos1,tile_contact,
                                             if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_being_stitched,
                                              tile_pos_index,tile_pos_stitch,voxel_range,step))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    tile_pos_stitch=np.array(tile_pos_stitch).reshape(tile_num,3)
    tile_pos_index=np.array(tile_pos_index,dtype='int')
    print('start saving_data')
    np.save(save_path+r'\tile_pos_index2.npy',tile_pos_index)
    np.save(save_path+r'\tile_pos_stitch2.npy',tile_pos_stitch)
    tile_pos2=update_pos(tile_pos1,tile_pos_stitch,tile_pos_index,voxel_len)
    np.save(save_path+r'\tile_pos2.npy',tile_pos2)
    print('end saving data')

def update_pos(tile_pos,tile_pos_stitch,tile_pos_index,voxel_len):
    '''
    this function updates positions of all tiles after being stitched
    return - array, size is the same as tile_pos
    tile_pos - array, the origin positions
    tile_pos_stitch - array, the position shift
    tile_pos_index - linspace(dtype=int), the index of reference tile
    '''
    def change(j,k):
        tile_pos1[k,:]=tile_pos1[k,:]-tile_pos_stitch1[j,:]
        index=np.argwhere(tile_pos_index==k)
        for [z] in index:
            change(j,z)
    print('start updating postions')
    tile_pos1=tile_pos.copy()
    tile_pos_stitch1=tile_pos_stitch.astype('float64')
    tile_pos_stitch1[:,0]=tile_pos_stitch1[:,0]*voxel_len[0]
    tile_pos_stitch1[:,1]=tile_pos_stitch1[:,1]*voxel_len[1]
    tile_pos_stitch1[:,2]=tile_pos_stitch1[:,2]*voxel_len[2]
    #i第一层，直接连接-1,j第一层每个
    #k第二层，l第二层每个
    i_=np.argwhere(tile_pos_index==-1)
    f_=np.argwhere(tile_pos_index!=-1)
    while(f_.shape[0]!=0):
        for [j_] in i_:
            k_=np.argwhere(tile_pos_index==j_)
            for l_ in k_:
                change(l_,l_)
                tile_pos_index[l_]=-1
        i_=np.argwhere(tile_pos_index==-1)
        f_=np.argwhere(tile_pos_index!=-1)
        #print(f_.shape,end=' ')
    print('finish updating postions')
    return tile_pos1

def import_2D_img(img_path,ordinal,dim_elem_num,z_th):
    img_name=r'%s\TileScan 1_s%.4d_z%.3d_RAW_ch00.tif'%(img_path,ordinal,z_th)
    return cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)


# In[ ]:


#python C:\Users\dingj\ZhaoLab\20220731_BF2DRandomStitch\BF2DRandomStitch_HighNoise.py
if __name__=='__main__':
    xml_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G\MetaData\Region 1.xml'
    img_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
    save_path=r'C:\Users\dingj\ZhaoLab\20220731_BF2DRandomStitch'
    sys.setrecursionlimit(10000)
    tile_pos=start_multi_stitchers(xml_path,img_path,save_path)
#!/usr/bin/env python
# coding: utf-8

# In[18]:


from lxml import etree
import numpy as np
import time
import cv2
import multiprocessing
from multiprocessing import Value,Array,Process
import ctypes
import random
from numba import jit,njit

#Functions
def import_img(img_path,ordinal,dim_elem_num):
    '''
    this function reads voxel information and return a 3D np_array.
    return - array, store the 3D image
    img_path - str, the file position,
    ordinal - int, the ordinal number for image,
    dim_elem_num - list, the quantities of voxels for each dimension.
    '''
    #s_time = time.time()
    voxel_array=np.zeros(tuple(dim_elem_num),dtype='uint8')#the array for storing image, dtyte should be changed according to image type
    #next statements get the img information according to image names, need to be changed according to different naming methods
    for i in range(dim_elem_num[2]):
        img_name=r'%s\Region 1_s%.4d_z%.3d_RAW_ch00.tif'%(img_path,ordinal,i)
        voxel_array[:,:,i]=cv2.imread(img_name)[:,:,0]
    #e_time = time.time()
    #print('import_img costs time - %f'%(e_time-s_time))
    return voxel_array

def get_2img_border(dim_elem_num,dim_len,voxel_len,tile_pos):
    '''
    get the border voxel index for two overlapping images
    return - array, the x/y/z_min/max voxel ordinal for each image,
    dim_elem_num - list, the quantities of voxels for each dimension,
    dim_len - list, the image length,
    tile_pos - array, xyz positions of each img.
    '''
    #s_time = time.time()
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
    #print(voxel_border)
    #assert xv1_min-xv1_max==xv2_min-xv2_max and yv1_min-yv1_max==yv2_min-yv2_max and zv1_min-zv1_max==zv2_min-zv2_max
    #e_time = time.time()
    #print('get_2img_border costs time - %f'%(e_time-s_time))
    return voxel_border

@njit(nogil=True)
def loss_fun(ovl1,ovl2,xyz_shift,alpha):
    '''
    this fucntion calculate the loss function of two overlapping images
    return - float, loss fun of overlapping area
    xyz_shift - list, record the xyz shift of original position,
    alpha - a hyperparameter
    '''
    #s_time = time.time()
    if 0 in ovl1.shape or 0 in ovl2.shape:
        print('no overlapping area')
        return np.inf
    ovl1,ovl2=ovl1.astype('int64'),ovl2.astype('int64')
    a=np.float64(np.sum((ovl1-ovl2)**2))/(np.sqrt(np.float64(np.sum(ovl1**2)))*np.sqrt(np.float64(np.sum(ovl2**2))))+alpha*np.sum(xyz_shift**2)
    #print(ovl1.shape,a)
    #e_time = time.time()
    #print('loss_fun time - %f'%(e_time-s_time))
    return a

def get_2img_border_after_shift(dim_elem_num,voxel_border,xyz_shift):
    '''
    this function calculates the border of two partly overlapping images after translation
    return - array, the voxel index of overlapping area for each image
    dim_elem_num - the voxel quantities for each dimension
    voxel_border - array, the voxel index before translation
    xyz_shift - list, the translation for each dimension
    '''
    #s_time = time.time()
    border_after_shift=np.zeros((2,6),dtype='uint')
    #calcualte the border after translation for three dimensions
    for i in range(3):
        if voxel_border[0,2*i]<voxel_border[1,2*i]:
            border_after_shift[0,2*i],border_after_shift[0,2*i+1]=0,np.max([0,voxel_border[0,2*i+1]+xyz_shift[i]])
            border_after_shift[1,2*i],border_after_shift[1,2*i+1]=np.min([dim_elem_num[i],voxel_border[1,2*i]-xyz_shift[i]]),dim_elem_num[i]
        elif voxel_border[0,2*i]>voxel_border[1,2*i]:
            border_after_shift[0,2*i],border_after_shift[0,2*i+1]=np.min([dim_elem_num[i],voxel_border[0,2*i]+xyz_shift[i]]),dim_elem_num[i]
            border_after_shift[1,2*i],border_after_shift[1,2*i+1]=0,np.max([0,voxel_border[1,2*i+1]-xyz_shift[i]])
        else:
            #in this case, x_min=0, x_max=dim_elem_num[i]
            if xyz_shift[i]<0:
                border_after_shift[0,2*i],border_after_shift[0,2*i+1]=0,voxel_border[0,2*i+1]+xyz_shift[i]
                border_after_shift[1,2*i],border_after_shift[1,2*i+1]=-xyz_shift[i],voxel_border[1,2*i+1]
            else:
                border_after_shift[0,2*i],border_after_shift[0,2*i+1]=xyz_shift[i],voxel_border[0,2*i+1]
                border_after_shift[1,2*i],border_after_shift[1,2*i+1]=0,voxel_border[1,2*i+1]-xyz_shift[i]
    #assert border_after_shift[0,1]-border_after_shift[0,0]==border_after_shift[1,1]-border_after_shift[1,0]
    #assert border_after_shift[0,3]-border_after_shift[0,2]==border_after_shift[1,3]-border_after_shift[1,2]
    #assert border_after_shift[0,5]-border_after_shift[0,4]==border_after_shift[1,5]-border_after_shift[1,4]
    #e_time = time.time()
    #print('get_2img_border_after_shift - %f'%(e_time-s_time))
    return border_after_shift

def calculate_img_xyz_shift(img1,img2,dim_elem_num,voxel_border,voxel_range,step):
    '''
    #this function gets the best overlapping position for two partly overlapping images
    #return - list, the voxel translation quantities of img1 to be stitched to img2
    #voxel_border - array, voxel range for overlapping area,
    #voxel_range - list, the range need to be calculated,
    #step - int 
    '''
    xv_shift,yv_shift,zv_shift=0,0,0
    loss_min=np.inf
    alpha=10e-6
    #img1,img2=img1.astype('float64'),img2.astype('float64')
    for x in range(-voxel_range[0],voxel_range[0]+1,step):
        for y in range(-voxel_range[1],voxel_range[1]+1,step):
            for z in range(-voxel_range[2],voxel_range[2]+1,step):
                border=get_2img_border_after_shift(dim_elem_num,voxel_border,[x,y,z])
                #print(x,y,z)
                this_loss=loss_fun(img1[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]],
                                   img2[border[1,2]:border[1,3],border[1,0]:border[1,1],border[1,4]:border[1,5]],
                                   np.array([x,y,z],dtype='float64'),alpha)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    #print(xv_shift,yv_shift,zv_shift,step)
    for x in range(xv_shift-2*step+1,xv_shift+2*step):
        for y in range(yv_shift-2*step+1,yv_shift+2*step):
            for z in range(zv_shift-step+1,zv_shift+step):
                border=get_2img_border_after_shift(dim_elem_num,voxel_border,[x,y,z])
                #print(x,y,z)
                this_loss=loss_fun(img1[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]],
                                   img2[border[1,2]:border[1,3],border[1,0]:border[1,1],border[1,4]:border[1,5]],
                                   np.array([x,y,z],dtype='float64'),alpha)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    #print(xv_shift,yv_shift,zv_shift,loss_min)
    return [xv_shift,yv_shift,zv_shift],loss_min

def if_closed_loop(i,j,tile_pos_index):
    '''
    this function judges if the i.th tile and j.th tile can form a closed loop accroding to stitch index
    '''
    while(j!=-1):
        if j==i:
            return True
        j=tile_pos_index[j]
    return False

def if_tile_blocked(i,j,tile_contact_list,tile_pos_index):
    j_contact_index=np.argwhere(tile_contact_list)
    j_contact_num=len(j_contact_index)-1
    count=0
    for k_ in j_contact_index:
        k=k_[0]
        if k==i:
            count+=1
        if tile_pos_index[k]==j:
            count+=1
    if count>=j_contact_num:
        return True
    return False

def update_pos(tile_pos,tile_pos_stitch,tile_pos_index):
    '''
    this function updates positions of all tiles after being stitched
    return - array, size is the same as tile_pos
    tile_pos - array, the origin positions
    tile_pos_stitch - array, the position shift
    tile_pos_index - linspace(dtype=int), the index of reference tile
    '''
    print('start updating postions')
    tile_len=tile_pos.shape[0]
    tile_pos_final=tile_pos.copy()
    no_update_pos=np.argwhere(tile_pos_index!=-1)
    num=0
    while(no_update_pos.shape[0]!=0):
        for i_ in no_update_pos:
            i=i_[0]
            j=tile_pos_index[i]
            tile_pos_final[i,:]=tile_pos_final[i,:]+tile_pos_stitch[i,:]
            tile_pos_index[i]=tile_pos_index[j]
        no_update_pos=np.argwhere(tile_pos_index!=-1)
        num+=1
        if num>tile_len*2:
            print('postion update failed for closed loop existing')
            return
    print('finish updating postions')
    return tile_pos_final
        
def get_img_xml_info(xml_path):
    '''
    Read xml file and extract the information of dimensions and each tile
    return - (1)dim_elem_num - linspace(uint), the quantity of voxels for each dimension,
    (2)dim_len - linspace(float), the length for one 3D image,
    (3)voxel_len - lingspace(float), the length for a voxel,
    (4)tile_len - int, the quantity of tiles,
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
        dim_len[i]=dim_attrib[i].attrib['Length']
    voxel_len=dim_len/dim_elem_num
    tile_attrib=my_et.xpath('//Attachment/Tile')
    tile_len=len(tile_attrib)
    tile_field=np.zeros((tile_len,2),dtype='uint')
    tile_pos=np.zeros((tile_len,3))
    for i in range(tile_len):
        tile_field[i,:]=[tile_attrib[i].attrib['FieldX'],tile_attrib[i].attrib['FieldY']]
        tile_pos[i,:]=[tile_attrib[i].attrib['PosX'],tile_attrib[i].attrib['PosY'],tile_attrib[i].attrib['PosZ']]
    return dim_elem_num,dim_len,voxel_len,tile_len,tile_field,tile_pos

def judge_tile_contact(dim_len,tile_pos):
    '''
    judge if two tiles contact with each other
    return - array, the contact array for each two images
    dim_len - linspace(float), the length for one 3D image
    tile_pos - array(float), the XYZ position information of each tile
    '''
    tile_len=tile_pos.shape[0]
    tile_contact=np.zeros((tile_len,tile_len),dtype='bool')
    for i in range(tile_len):
        for j in range(tile_len):
            if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*1.05)==3:
                tile_contact[i,j]=True
    return tile_contact

def run_multiprocess(lock,if_tile_stitched,if_tile_being_process,img_path,dim_elem_num,dim_len,voxel_len,tile_pos,
                     voxel_range,step,tile_pos_index,tile_pos_stitch,tile_contact):
    '''
    this function runs under multiprocessing.Process()
    '''
    #print(multiprocessing.current_process().name)
    while(False in if_tile_stitched):
        lock.acquire()
        usable_index=[index for index,i in enumerate(if_tile_stitched or if_tile_being_process) if i==False]
        i=usable_index[random.randint(0,len(usable_index)-1)]
        if_tile_being_process[i]=True
        img1=import_img(img_path,i,dim_elem_num)
        j,ovl_intense=find_stitched_tile(i,img1,tile_contact,dim_elem_num,dim_len,voxel_len,tile_pos,tile_pos_index)
        if j==-1:
            if_tile_being_process[i]=False
            print('%d.th tile has no appropriate contact tile'%(i))
            lock.release()
            continue
        tile_pos_index[i]=j
        lock.release()
        img2=import_img(img_path,j,dim_elem_num)
        border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
        ovl1=img1[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]]
        ovl2=img2[border[1,2]:border[1,3],border[1,0]:border[1,1],border[0,4]:border[0,5]]
        m1,m2=np.mean(ovl1),np.mean(ovl2)
        if m1<20 or m2<20:
            img1,img2=increase_contrast(img1,img2,border)
        elif np.abs(m2-m1)>5:
            img1,img2=adjust_contrast(img1,img2,border)
        xyz_shift,loss_min=calculate_img_xyz_shift(img1,img2,dim_elem_num,border,voxel_range,step)
        print('%s stitch %d.th tile with %d.th tile, shift is (%d, %d, %d) loss is %.8f'
              %(multiprocessing.current_process().name,i,j,*xyz_shift,loss_min))
        lock.acquire()
        if_tile_stitched[i]=True
        if_tile_being_process[i]=False
        tile_pos_stitch[3*i:3*i+3]=xyz_shift*voxel_len
        lock.release()
        time.sleep(5)

def find_stitched_tile(i,img,tile_contact,dim_elem_num,dim_len,voxel_len,tile_pos,tile_pos_index):
    '''
    this function finds the best tile to be stitched by i.th tile
    '''
    contact_index=np.argwhere(tile_contact[i,:])
    ovl_index=-1
    ovl_voxel_num=0
    ovl_intense=0
    #sort order: (1)voxel_num, (2)ovl_intese
    for j_ in contact_index:
        j=j_[0]
        if j==i:
            continue
        if if_closed_loop(i,j,tile_pos_index):
            continue
        if if_tile_blocked(i,j,tile_contact[j,:],tile_pos_index):
            continue
        border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
        one_intense=np.sum(img[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]])
        if one_intense>ovl_intense:
            ovl_index=j
            ovl_intense=one_intense
            ovl_voxel_num=(border[0,1]-border[0,0])*(border[0,3]-border[0,2])*(border[0,5]-border[0,4])
    return ovl_index,ovl_intense

def increase_contrast(img1,img2,border):
    '''
    increase contrast of image with low voxel value to average 20
    '''
    img1,img2=img1.astype('float64'),img2.astype('float64')
    ovl1=img1[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]]
    ovl2=img2[border[1,2]:border[1,3],border[1,0]:border[1,1],border[0,4]:border[0,5]]
    img1,img2=np.clip(20/np.mean(ovl1)*img1,0,255),np.clip(20/np.mean(ovl2)*img2,0,255)
    img1,img2=img1.astype('uint8'),img2.astype('uint8')
    return img1,img2

def adjust_contrast(img1,img2,border):
    '''
    adjust contrast of two images to be equal
    '''
    img1,img2=img1.astype('float64'),img2.astype('float64')
    ovl1=img1[border[0,2]:border[0,3],border[0,0]:border[0,1],border[0,4]:border[0,5]]
    ovl2=img2[border[1,2]:border[1,3],border[1,0]:border[1,1],border[0,4]:border[0,5]]
    m1,m2=np.mean(ovl1),np.mean(ovl2)
    if m1<m2:
        img1=np.clip(m2/m1*img1,0,255)
    else:
        img2=np.clip(m1/m2*img2,0,255)
    img1,img2=img1.astype('uint8'),img2.astype('uint8')
    return img1,img2

def start_BF_stitch(xml_path,img_path,voxel_range,step):
    '''
    main() function
    '''
    dim_elem_num,dim_len,voxel_len,tile_len,tile_field,tile_pos=get_img_xml_info(xml_path)
    tile_contact=judge_tile_contact(dim_len,tile_pos)
    if_tile_stitched=Array(ctypes.c_bool,[False for i in range(tile_len)])
    if_tile_stitched[0]=True
    if_tile_being_process=Array(ctypes.c_bool,[False for i in range(tile_len)])
    tile_pos_index=Array('i',[-1 for i in range(tile_len)])
    tile_pos_stitch=Array('d',[0 for i in range(tile_len*3)])
    lock=multiprocessing.RLock()
    process_num=round(0.4*multiprocessing.cpu_count())
    print('Current processing quantities: %d'%(process_num))
    process_list=[]
    #multiprocessing.freeze_support() 
    for i in range(process_num):
        one_pro=multiprocessing.Process(target=run_multiprocess,
                                        args=(lock,if_tile_stitched,if_tile_being_process,
                                              img_path,dim_elem_num,dim_len,voxel_len,tile_pos,voxel_range,step,
                                              tile_pos_index,tile_pos_stitch,tile_contact))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    tile_pos_stitch=np.array(tile_pos_stitch).reshape(tile_len,3)
    tile_pos_index=np.array(tile_pos_index,dtype='int')
    tile_pos_final=update_pos(tile_pos,tile_pos_stitch,tile_pos_index)
    return tile_pos_final

def update_xml(xml_path,tile_pos_final):
    return


# In[ ]:


#python C:\Users\dingj\ZhaoLab\20220724_BFStitch\ImageBruteForceStitch.py
if __name__=='__main__':
    xml_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G\MetaData\Region 1.xml'
    img_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
    voxel_range=[30,30,15]
    step=3
    tile_pos_final=start_BF_stitch(xml_path,img_path,voxel_range,step)
    print('start saving data')
    np.save('tile_pos_final.npy',tile_pos_final)
    print('end saving')


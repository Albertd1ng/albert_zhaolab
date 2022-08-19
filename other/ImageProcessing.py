#!/usr/bin/env python
# coding: utf-8

# In[14]:


from lxml import etree
import numpy as np
import time
import cv2
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import Manager,Value,Array,Process
import ctypes

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
    ovl1,ovl2=ovl1.astype('float'),ovl2.astype('float')
    g=lambda a: np.sum(np.array(a)**2)
    a=np.sum((ovl1-ovl2)**2)/(np.sum(ovl1)*np.sum(ovl2))+alpha*g(xyz_shift)
    #print(ovl1.shape,a)
    #e_time = time.time()
    #print('loss_fun time - %f'%(e_time-s_time))
    return a
"""
def loss_fun(ovl1,ovl2,xyz_shift,alpha):
    import torch
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
    ovl1,ovl2=torch.from_numpy(ovl1.astype('float')),torch.from_numpy(ovl2.astype('float'))
    ovl1,ovl2=ovl1.to('cuda'),ovl2.to('cuda')
    xyz_shift_c=torch.tensor(xyz_shift,device='cuda')
    a=torch.sum((ovl1-ovl2)**2)/(torch.sum(ovl1)*torch.sum(ovl2))+alpha*torch.sum(xyz_shift_c**2)
    a.to('cpu')
    #e_time = time.time()
    #print('loss_fun time - %f'%(e_time-s_time))
    return a
"""

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
    alpha=10e-10
    for x in range(-voxel_range[0],voxel_range[0]+1,step):
        for y in range(-voxel_range[1],voxel_range[1]+1,step):
            for z in range(-voxel_range[2],voxel_range[2]+1,step):
                border=get_2img_border_after_shift(dim_elem_num,voxel_border,[x,y,z])
                this_loss=loss_fun(img1[border[0,0]:border[0,1],border[0,2]:border[0,3],border[0,4]:border[0,5]],
                                   img2[border[1,0]:border[1,1],border[1,2]:border[1,3],border[1,4]:border[1,5]],[x,y,z],alpha)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
                #print(x,y,z,this_loss)
    for x in range(xv_shift-step+1,xv_shift+step):
        for y in range(yv_shift-step+1,yv_shift+step):
            for z in range(zv_shift-step+1,zv_shift+step):
                border=get_2img_border_after_shift(dim_elem_num,voxel_border,[x,y,z])
                this_loss=loss_fun(img1[border[0,0]:border[0,1],border[0,2]:border[0,3],border[0,4]:border[0,5]],
                                   img2[border[1,0]:border[1,1],border[1,2]:border[1,3],border[1,4]:border[1,5]],[x,y,z],alpha)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    print(xv_shift,yv_shift,zv_shift,loss_min)
    return xv_shift,yv_shift,zv_shift
"""
def run_multiprocess(lock,if_tile_stitched,if_tile_being_process,img_path,dim_elem_num,dim_len,voxel_len,tile_pos,voxel_range,step):
    print(multiprocessing.current_process().name)
    lock.acquire()
    usable_index=[index for index,i in enumerate(if_tile_stitched or if_tile_being_process) if i==False]
    i=usable_index[0]
    if_tile_being_process[i]=True
    lock.release()
    contact_index=np.argwhere(tile_contact[i,:])
    max_ovl=None
    max_ovl_voxel=0
    for j_ in contact_index:
        j=j_[0]
        if j==i:
            continue
        if if_closed_loop(i,j):
            continue
        voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
        ovl_voxel=(voxel_border[0,1]-voxel_border[0,0])*(voxel_border[0,3]-voxel_border[0,2])*(voxel_border[0,5]-voxel_border[0,4])
        if ovl_voxel>max_ovl_voxel:
            max_ovl_voxel=ovl_voxel
            max_ovl=j
    if max_ovl==None:
        lock.acquire()
        if_tile_being_process[i]=False
        lock.release()
        return tuple(0,0,np.zeros(3))
    j=max_ovl
    print(i,j,max_ovl_voxel)
    img1=import_img(img_path,i,dim_elem_num)
    img2=import_img(img_path,j,dim_elem_num)
    voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
    xyz_shift=calculate_img_xyz_shift(img1,img2,dim_elem_num,voxel_border,voxel_range,step)
    lock.acquire()
    if_tile_stitched[i]=True
    if_tile_being_process[i]=False
    lock.release()
    return tuple(i,j,np.array(xyz_shift))

def outer(tile_pos_index,tile_pos_stitch):
    def done(res):
        a=res.result()
        print(a)
        i,j=a[0],a[1]
        tile_pos_index[i]=j
        tile_pos_stitch[i,:]=a[3]*voxel_len
    return done
"""
def run_multiprocess(lock,if_tile_stitched,if_tile_being_process,img_path,dim_elem_num,dim_len,voxel_len,tile_pos,voxel_range,step,tile_pos_index,tile_pos_stitch):
    print(multiprocessing.current_process().name)
    while(False in if_tile_stitched):
        lock.acquire()
        usable_array=[i or j for i,j in zip(if_tile_stitched,if_tile_being_process)]
        usable_index=[index for index,i in enumerate(usable_array) if i==False]
        i=usable_index[0]
        if_tile_being_process[i]=True
        lock.release()
        contact_index=np.argwhere(tile_contact[i,:])
        max_ovl=None
        max_ovl_voxel=0
        for j_ in contact_index:
            j=j_[0]
            if j==i:
                continue
            if if_closed_loop(i,j,tile_pos_index):
                continue
            voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
            ovl_voxel=(voxel_border[0,1]-voxel_border[0,0])*(voxel_border[0,3]-voxel_border[0,2])*(voxel_border[0,5]-voxel_border[0,4])
            if ovl_voxel>max_ovl_voxel:
                max_ovl_voxel=ovl_voxel
                max_ovl=j
        if max_ovl==None:
            lock.acquire()
            if_tile_being_process[i]=False
            lock.release()
            continue
        j=max_ovl
        print(i,j,max_ovl_voxel)
        img1=import_img(img_path,i,dim_elem_num)
        img2=import_img(img_path,j,dim_elem_num)
        voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
        xyz_shift=calculate_img_xyz_shift(img1,img2,dim_elem_num,voxel_border,voxel_range,step)
        lock.acquire()
        if_tile_stitched[i]=True
        if_tile_being_process[i]=False
        tile_pos_index[i]=j
        tile_pos_stitch[3*i:3*i+3]=xyz_shift*voxel_len
        lock.release()

def if_closed_loop(i,j,tile_pos_index):
    while(j!=-1):
        if j==i:
            return True
        j=tile_pos_index[j]
    return False
        
def update_pos(tile_pos,tile_pos_stitch,tile_pos_index):
    '''
    this function updates positions of all tiles after being stitched
    return - array, size is the same as tile_pos
    tile_pos - array, the origin positions
    tile_pos_stitch - array, the position shift
    tile_pos_index - linspace(dtype=int), the index of reference tile
    '''
    tile_len=tile_pos.shape[0]
    for i in range(tile_len):
        j=tile_pos_index[i]
        if j==-1 or j==0:
            continue
        #从拼接的算法来说这个图一定没有闭环，所以这个算法一定不会陷入死循环
        while(j!=0 or j!=-1):
            tile_pos_stitch[i,:]=tile_pos_stitch[i,:]+tile_pos_stitch[j,:]
            j=tile_pos_index[j]
    return tile_pos+tile_pos_stitch


# In[3]:


#Read xml file and extract the information of dimensions and each tile
my_xml_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G\MetaData\Region 1.xml'
parser=etree.XMLParser()
my_et=etree.parse(my_xml_path,parser=parser)

#Calculate the attributes of dimensions
dim_attrib=my_et.xpath('//Dimensions/DimensionDescription')
dim_elem_num=np.zeros(3,dtype='uint')#the quantity of voxels for each dimension
dim_len=np.zeros(3)#the length for one 3D image
for i in range(3):
    dim_elem_num[i],dim_len[i]=dim_attrib[i].attrib['NumberOfElements'],dim_attrib[i].attrib['Length']
    dim_len[i]=dim_attrib[i].attrib['Length']
voxel_len=dim_len/dim_elem_num

#Calculate the identifier and position for each tile
tile_attrib=my_et.xpath('//Attachment/Tile')
tile_len=len(tile_attrib)#the quantity of tiles
tile_field=np.zeros((tile_len,2),dtype='uint')#the identifier of each file
#(X,Y,Z) for tile_pos
tile_pos=np.zeros((tile_len,3))#the position of each file
for i in range(tile_len):
    tile_field[i,:]=[tile_attrib[i].attrib['FieldX'],tile_attrib[i].attrib['FieldY']]
    tile_pos[i,:]=[tile_attrib[i].attrib['PosX'],tile_attrib[i].attrib['PosY'],tile_attrib[i].attrib['PosZ']]


# In[4]:


#Determine contacted tiles
tile_contact=np.zeros((tile_len,tile_len),dtype='bool')
for i in range(tile_len):
    for j in range(tile_len):
        if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*1.05)==3:
            tile_contact[i,j]=True


# In[31]:


#Run
'''
if_tile_stitched=np.zeros(tile_len,dtype='bool')#record if the tile is stitched
tile_pos_stitch=np.zeros((tile_len,3))
tile_pos_index=-np.ones(tile_len,dtype='int')
if_tile_stitched[0]=True
img_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
voxel_range=[50,50,50]
step=5
while(False in if_tile_stitched):
    for i in range(tile_len):
        if if_tile_stitched[i]==True:
            continue
        img1=import_img(img_path,i,dim_elem_num)
        contact_list=np.argwhere(np.logical_and(if_tile_stitched,tile_contact[i,:]))
        #print(contact_list)
        if contact_list.size==0:
            continue
        max_ovl=None
        max_ovl_voxel=0
        for j_ in contact_list:
            j=j_[0]
            if j==i:
                continue
            voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
            print(voxel_border)
            ovl_voxel=(voxel_border[0,1]-voxel_border[0,0])*(voxel_border[0,3]-voxel_border[0,2])*(voxel_border[0,5]-voxel_border[0,4])
            if ovl_voxel>max_ovl_voxel:
                max_ovl_voxel=ovl_voxel
                max_ovl=j
        j=max_ovl
        print(i,j)
        img2=import_img(img_path,j,dim_elem_num)
        voxel_border=get_2img_border(dim_elem_num,dim_len,voxel_len,np.vstack((tile_pos[i,:],tile_pos[j,:])))
        xyz_shift=calculate_img_xyz_shift(img1,img2,dim_elem_num,voxel_border,voxel_range,step)
        tile_pos_stitch[i,:]=np.array(xyz_shift)*voxel_len
        tile_pos_index[i]=j
        if_tile_stitched[i]=True
'''
"""
if __name__=='__main__':
    img_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
    voxel_range=[50,50,50]
    step=5
    if_tile_stitched=Manager().Array('i',[False for i in range(tile_len)])
    if_tile_stitched[0]=True
    if_tile_being_process=Manager().Array('i',[False for i in range(tile_len)])
    tile_pos_index=-np.ones(tile_len,dtype='int')
    tile_pos_stitch=np.zeros((tile_len,3))
    lock=Manager().RLock()
    process_num=round(0.5*multiprocessing.cpu_count())
    pool=ProcessPoolExecutor(process_num)
    print('Current processing quantities: %d'%(process_num))
    while(False in if_tile_stitched):
        one_pro=pool.submit(run_multiprocess,
                     lock,if_tile_stitched,if_tile_being_process,img_path,dim_elem_num,dim_len,voxel_len,tile_pos,voxel_range,step)
        one_pro.add_done_callback(outer(tile_pos_index,tile_pos_stitch))
    pool.shutdown(True)
"""
if __name__=='__main__':
    img_path=r'D:\Albert\Data\ZhaoLab\Imaging\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
    voxel_range=[50,50,50]
    step=5
    if_tile_stitched=Array(ctypes.c_bool,[False for i in range(tile_len)])
    if_tile_stitched[0]=True
    if_tile_being_process=Array(ctypes.c_bool,[False for i in range(tile_len)])
    lock=multiprocessing.RLock()
    #if_tile_stitched=Manager().Array('i',[False for i in range(tile_len)])
    #if_tile_stitched[0]=True
    #if_tile_being_process=Manager().Array('i',[False for i in range(tile_len)])
    #lock=Manager().RLock()
    tile_pos_index=Array('i',[-1 for i in range(tile_len)])
    tile_pos_stitch=Array('d',[0 for i in range(tile_len*3)])
    process_num=round(0.4*multiprocessing.cpu_count())
    print('Current processing quantities: %d'%(process_num))
    process_list=[]
    multiprocessing.freeze_support() 
    for i in range(process_num):
        one_pro=multiprocessing.Process(target=run_multiprocess,args=(lock,if_tile_stitched,if_tile_being_process,img_path,dim_elem_num,dim_len,voxel_len,tile_pos,voxel_range,step,tile_pos_index,tile_pos_stitch,))
        #print(one_pro.Daemon)
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    for i in tile_pos_index:
        print(i,end=" ")
    tile_pos_stitch=np.array(tile_pos_stitch).reshape(tile_len,3)
    tile_pos_index=np.array(tile_pos_index,dtype='int')
    print('start saving data')
    np.save('tile_pos_stitch.npy',tile_pos_stitch)
    np.save('tile_pos_index',tile_pos_index)
    print('end saving')
#update all positions for all tiles
#tile_final_pos=update_pos(tile_pos,tile_pos_stitch,tile_pos_index)
#print(tile_final_pos)
#python ZhaoLab\ImageProcessing.py
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

#BFhori_Functions
def get_img_xml_info(layer_num,xml_path,save_path):
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
    tile_pos=np.zeros((tile_num,3),dtype='float64')
    for i in range(tile_num):
        tile_field[i,:]=[tile_attrib[i].attrib['FieldX'],tile_attrib[i].attrib['FieldY']]
        tile_pos[i,:]=[tile_attrib[i].attrib['PosX'],tile_attrib[i].attrib['PosY'],0]
    np.save(save_path+r'\dim_elem_num_%.4d.npy'%(layer_num),dim_elem_num)
    np.save(save_path+r'\dim_len_%.4d.npy'%(layer_num),dim_len)
    np.save(save_path+r'\voxel_len.npy',voxel_len)
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
            if i==j:
                tile_contact[i,j]=False
                continue
            if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*np.array([1,0.3,0.3]))==3:
                tile_contact[i,j]=True
                continue
            if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*np.array([0.3,1,0.3]))==3:
                tile_contact[i,j]=True
                continue
    return tile_contact

def judge_tile_contact_list(i,dim_len,tile_pos):
    '''
    judge the contact tiles of i.th tile
    return - linspace(bool)
    dim_len - linspace(float), the length for one 3D image
    tile_pos - array(float), the XYZ position information of each tile
    '''
    tile_num=tile_pos.shape[0]
    tile_contact_list=np.zeros(tile_num,dtype='bool')
    for j in range(tile_num):
        if i==j:
            tile_contact_list[j]=False
            continue
        if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*np.array([1,0.3,0.3]))==3:
            tile_contact_list[j]=True
            continue
        if np.sum(np.abs(tile_pos[i,:]-tile_pos[j,:])<dim_len*np.array([0.3,1,0.3]))==3:
            tile_contact_list[j]=True
            continue
    return tile_contact_list

def import_img(img_path,img_name,ordinal,dim_elem_num):
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
        one_img_name=r'%s\%s_s%.4d_z%.3d_RAW_ch00.tif'%(img_path,img_name,ordinal,i)
        voxel_array[:,:,i]=cv2.imread(one_img_name,cv2.IMREAD_GRAYSCALE)
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

def choose_reference_tile(i,tile_contact_list,if_tile_stitched):
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
    index_j=np.array(index_j,dtype='int')
    dis_ij=np.abs(index_j-i)
    max_dis_index=np.argwhere(dis_ij==np.max(dis_ij))
    j=index_j[max_dis_index[0,0]]
    return j

def choose_reference_tile_for_stitch(tile_contact_list,if_tile_stitched):
    '''
    return - list(int), the contact index of reference tiles for i.th tile
    tile_contact_list - list(bool), if the tile contact to i.th tile. if yes, choose as reference tile
    if_tile_stitched - list(bool), if the tile has been stitched. if yes, choose as reference tile
    '''
    index_j=[]
    for j in range(len(if_tile_stitched)):
        if tile_contact_list[j] and if_tile_stitched[j]:
            index_j.append(j)
    if len(index_j)==0:
        return -1
    return index_j

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

def get_border_list(i,index_j,dim_elem_num,dim_len,voxel_len,tile_pos_stitch):
    '''
    return - list(numpy.array) i.th and j.th border list for j in index_j 
    i - int, the i.th tile to be stitched
    index_j - list(int), the index of reference tiles for i.th tile
    dim_elem_num, dim_len, voxel_len
    tile_pos_stitch - array(float), the stitched tiles' postions
    '''
    border_list=[]
    for k in index_j:
        border_list.append(get_2img_border(dim_elem_num,dim_len,voxel_len,
                                           np.vstack((np.array(tile_pos_stitch[3*i:3*i+3]),np.array(tile_pos_stitch[3*k:3*k+3])))))
    return border_list

def loss_fun(ovl1_list,ovl2_list,xyz_shift,alpha):
    '''
    this fucntion calculate the loss function of two overlapping images
    return - float, loss fun of overlapping area
    xyz_shift - list, record the xyz shift of original position,
    alpha - a hyperparameter
    '''
    ovl_num=len(ovl1_list)
    a,b,c=0,0,0
    for i in range(ovl_num):
        a=a+np.sum(ovl1_list[i]**2)
        b=b+np.sum(ovl2_list[i]**2)
    if a==0 or b==0:
        #print('no overlapping area')
        return np.inf
    for i in range(ovl_num):
        c=c+np.sum((ovl1_list[i]-ovl2_list[i])**2)
    loss=c/np.sqrt(a*b)+alpha*np.sum(xyz_shift**2)
    return loss

def calculate_xyz_shift(i,index_j,border_list,img_path,img_name,dim_elem_num,voxel_range,step):
    '''
    #this function gets the best overlapping position for two partly overlapping images
    #return - list, the voxel translation quantities of img1 to be stitched to img2
    #voxel_border - array, voxel range for overlapping area,
    #voxel_range - list, the range need to be calculated,
    #step - int 
    '''
    img1=np.float32(cv2.medianBlur(import_img(img_path,img_name,i,dim_elem_num),5))
    m1=np.mean(img1)
    if m1<0.001:
        return [0,0,0],0
    img2_list=[]
    for j in index_j:
        img2_list.append(np.float32(cv2.medianBlur(import_img(img_path,img_name,j,dim_elem_num),5)))
    ovl_num=len(border_list)
    xv_shift,yv_shift,zv_shift=0,0,0
    loss_min=np.inf
    alpha=5*10e-5
    for x in range(-voxel_range[0],voxel_range[0]+1,step):
        for y in range(-voxel_range[1],voxel_range[1]+1,step):
            for z in range(-voxel_range[2],voxel_range[2]+1,step):
                border_list_shift=[]
                ovl1_list=[]
                ovl2_list=[]
                for i in range(ovl_num):
                    border_list_shift.append(get_2img_border_after_shift(dim_elem_num,border_list[i],[x,y,z]))
                for i in range(ovl_num):
                    ovl1,ovl2=get_overlapping_img(img1,img2_list[i],border_list_shift[i])
                    ovl1_list.append(ovl1)
                    ovl2_list.append(ovl2)
                this_loss=loss_fun(ovl1_list,ovl2_list,np.array([x,y,z],dtype='float32'),alpha)
                #print((x,y,z),this_loss)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    for x in range(xv_shift-2*step+1,xv_shift+2*step):
        for y in range(yv_shift-2*step+1,yv_shift+2*step):
            for z in range(zv_shift-step+1,zv_shift+step):
                border_list_shift=[]
                ovl1_list=[]
                ovl2_list=[]
                for i in range(ovl_num):
                    border_list_shift.append(get_2img_border_after_shift(dim_elem_num,border_list[i],[x,y,z]))
                for i in range(ovl_num):
                    ovl1,ovl2=get_overlapping_img(img1,img2_list[i],border_list_shift[i])
                    ovl1_list.append(ovl1)
                    ovl2_list.append(ovl2)
                this_loss=loss_fun(ovl1_list,ovl2_list,np.array([x,y,z],dtype='float32'),alpha)
                #print((x,y,z),this_loss)
                if this_loss<loss_min:
                    loss_min=this_loss
                    xv_shift,yv_shift,zv_shift=x,y,z
    #print(xv_shift,yv_shift,zv_shift,loss_min)
    return [xv_shift,yv_shift,zv_shift],loss_min

def choose_start_img(img_path,img_name,tile_num,dim_elem_num,previous=0):
    '''
    choose the first image and set it been stitched
    previous - int, the start image need a distance from previous start image
    '''
    m,a=0,0
    while(m<10 or abs(a-previous)<0.1*round(tile_num)):
        a=random.randint(int(0.2*tile_num),int(0.3*tile_num))
        m=np.mean(import_img(img_path,img_name,a,dim_elem_num))
    print('start_tile is %d.th tile, average is %.8f'%(a,m))
    return a

def calculate_axis_range_voxel_num(tile_pos,dim_elem_num,voxel_len):
    '''
    return - 
        axis_range - array(float,(3,2)), the xyz min and max position for a whole layer
        voxel_num - array(uint,3), the voxel quantities for xy dimensions
    '''
    axis_range=np.zeros((3,2))
    axis_range[0,0],axis_range[0,1]=np.min(tile_pos[:,0]),np.max(tile_pos[:,0])+voxel_len[0]*dim_elem_num[0]
    axis_range[1,0],axis_range[1,1]=np.min(tile_pos[:,1]),np.max(tile_pos[:,1])+voxel_len[1]*dim_elem_num[1]
    axis_range[2,0],axis_range[2,1]=np.min(tile_pos[:,2]),np.max(tile_pos[:,2])+voxel_len[2]*dim_elem_num[2]
    voxel_num=np.uint32(np.round((axis_range[:,1]-axis_range[:,0])/voxel_len))
    return axis_range,voxel_num

def find_first_last_index(tile_pos,tile_num,dim_len,dim_elem_num,axis_range,voxel_len,voxel_num):
    '''
    return - list(int,2)
        list[0] and list[1] are the first and last indice need to be exported in the z direction for one layer
    '''
    first_last_index=np.array([voxel_num[2],0],dtype='uint')
    for i in range(voxel_num[2]):
        num_one_layer=0
        this_z=axis_range[2,0]+voxel_len[2]*i
        for j in range(tile_num):
            if tile_pos[j,2]+dim_len[2]<this_z:
                continue
            if tile_pos[j,2]>this_z:
                continue
            z_th=np.int32(np.round((this_z-tile_pos[j,2])/voxel_len[2]))
            if z_th>=dim_elem_num[2]:
                continue
            num_one_layer+=1
        if num_one_layer==tile_num:
            if i<=first_last_index[0]:
                first_last_index[0]=i
            if i>=first_last_index[1]:
                first_last_index[1]=i
    return first_last_index

def run_sift_stitcher(lock,img_path,img_name,dim_elem_num,dim_len,voxel_len,tile_num,tile_contact,tile_pos,
                      tile_pos_stitch,if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_being_stitched,
                      bad_tile_threshold,voxel_range,step):
    stitch_num=0
    this_name=multiprocessing.current_process().name
    while(False in if_tile_stitched):
        lock.acquire()
        #选择被拼接的tile
        usable_tile_index=[index for index,value in enumerate(zip(if_tile_stitched,if_tile_shelved,if_tile_being_stitched)) if not any(value)]
        if len(usable_tile_index)==0:
            for i in range(tile_num):
                if_tile_shelved[i]=False
            time.sleep(1)
            lock.release()
            #print('All shelved tile has been released')
            continue
        i=usable_tile_index[random.randint(0,len(usable_tile_index)-1)]
        #选择基准tile
        j=choose_reference_tile(i,tile_contact[i,:],if_tile_stitched)
        if j==-1:
            if_tile_shelved[i]=True
            lock.release()
            #print('%d.th tile has no appropriate contact tile'%(i))
            continue
        if_tile_being_stitched[i]=True
        #更新位置
        dis_ij=tile_pos[i,:]-tile_pos[j,:]
        tile_pos_stitch[3*i:3*i+3]=tile_pos_stitch[3*j:3*j+3]+dis_ij
        lock.release()
        #根据更新的位置选择相邻tile
        tile_contact_list=judge_tile_contact_list(i,dim_len,np.array(tile_pos_stitch).reshape(-1,3))
        index_j=choose_reference_tile_for_stitch(tile_contact_list,if_tile_stitched)
        #index_j肯定非空
        ovl_num=len(index_j)
        #print(f'{this_name} is stitching %{i}.th tile with {index_j}.th tiles')
        border_list=get_border_list(i,index_j,dim_elem_num,dim_len,voxel_len,tile_pos_stitch)
        xyz_shift,loss_min=calculate_xyz_shift(i,index_j,border_list,img_path,img_name,dim_elem_num,voxel_range,step)
        lock.acquire()
        if_tile_being_stitched[i]=False
        if loss_min>bad_tile_threshold+0.25*if_tile_bad[i]:
            if_tile_bad[i]=if_tile_bad[i]+1
            if if_tile_bad[i]>=3:
                if_tile_stitched[i]=True
            else:
                if_tile_shelved[i]=True
            print('%d.th tile is a bad tile, times is %d'%(i,if_tile_bad[i]))
        else:
            if_tile_stitched[i]=True
            tile_pos_stitch[3*i:3*i+3]=tile_pos_stitch[3*i:3*i+3]-np.array(xyz_shift,dtype='float64')*voxel_len
            stitch_num+=1
        lock.release()
        print('%s has stitched %d tiles, current stitch is %d.th tile with %s tiles,\nxyz_shift is (%d, %d, %d), loss is %.8f'
              %(this_name,stitch_num,i,str(index_j),*xyz_shift,loss_min))
    print('!!!!!!%s stops and has stitched %d tiles.!!!!!!'%(this_name,stitch_num))

def start_multi_stitchers(layer_num,xml_path,img_path,img_name,save_path,bad_tile_threshold=0.5,voxel_range=[21,21,9],step=3):
    dim_elem_num,dim_len,voxel_len,tile_num,tile_field,tile_pos=get_img_xml_info(layer_num,xml_path,save_path)
    try:
        tile_contact=np.load(save_path+r'\tile_contact_%.4d.npy'%(layer_num))
    except:
        tile_contact=judge_tile_contact(dim_len,tile_pos)
        np.save(save_path+r'\tile_contact_%.4d.npy'%(layer_num),tile_contact)
    lock=multiprocessing.RLock()
    if_tile_stitched=Array(ctypes.c_bool,[False for i in range(tile_num)])
    start_index=choose_start_img(img_path,img_name,tile_num,dim_elem_num,0)
    if_tile_stitched[start_index]=True
    if_tile_being_stitched=Array(ctypes.c_bool,[False for i in range(tile_num)])
    if_tile_shelved=Array(ctypes.c_bool,[False for i in range(tile_num)])
    if_tile_bad=Array('i',[0 for i in range(tile_num)])
    tile_pos_stitch=Array('d',[0 for i in range(tile_num*3)])
    process_num=min(round(0.3*multiprocessing.cpu_count()),10)
    print('Current processing quantities: %d'%(process_num))
    process_list=[]
    for i in range(process_num):
        one_pro=multiprocessing.Process(target=run_sift_stitcher,
                                        args=(lock,img_path,img_name,dim_elem_num,dim_len,voxel_len,tile_num,tile_contact,tile_pos,
                                              tile_pos_stitch,if_tile_stitched,if_tile_shelved,if_tile_bad,if_tile_being_stitched,
                                              bad_tile_threshold,voxel_range,step))
        one_pro.start()
        process_list.append(one_pro)
    for i in process_list:
        i.join()
    #####################################################################################
    print('start saving_data')
    tile_pos_stitch=np.array(tile_pos_stitch).reshape(tile_num,3)
    np.save(save_path+r'\tile_pos_stitch_%.4d.npy'%(layer_num),tile_pos_stitch)
    axis_range,voxel_num=calculate_axis_range_voxel_num(tile_pos_stitch,dim_elem_num,voxel_len)
    first_last_index=find_first_last_index(tile_pos_stitch,tile_num,dim_len,dim_elem_num,axis_range,voxel_len,voxel_num)
    np.save(save_path+r'\axis_range_stitch_%.4d.npy'%(layer_num),axis_range)
    np.save(save_path+r'\first_last_index_%.4d.npy'%(layer_num),first_last_index)
    print('end saving data')
    return tile_pos_stitch


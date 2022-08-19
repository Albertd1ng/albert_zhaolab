#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from lxml import etree
import numpy as np
import time
import cv2
import random

#SIFTvert_Functions
def get_img_info(i,save_path):
    dim_elem_num=np.load(save_path+r'\dim_elem_num_%.4d.npy'%(i))
    dim_len=np.load(save_path+r'\dim_len_%.4d.npy'%(i))
    voxel_len=np.load(save_path+r'\voxel_len.npy')
    return dim_elem_num,dim_len,voxel_len

def import_2D_img(img_path,img_name,ordinal,z_th):
    one_img_name=r'%s\%s_s%.4d_z%.3d_RAW_ch00.tif'%(img_path,img_name,ordinal,z_th)
    return cv2.imread(one_img_name,cv2.IMREAD_GRAYSCALE)

def get_whole_img(index,img_path,img_name,axis_range,dim_elem_num,voxel_num,voxel_len,tile_pos):
    tile_num=tile_pos.shape[0]
    whole_img=np.zeros((voxel_num[1::-1]),dtype='uint8')
    this_z=axis_range[2,0]+voxel_len[2]*index
    for j in range(tile_num):
        z_th=np.int64(np.round((this_z-tile_pos[j,2])/voxel_len[2]))
        x_th=np.int64(np.round((tile_pos[j,0]-axis_range[0,0])/voxel_len[0]))
        y_th=np.int64(np.round((tile_pos[j,1]-axis_range[1,0])/voxel_len[0]))
        #print(img_path,img_name,j,dim_elem_num,z_th)
        img_2D=import_2D_img(img_path,img_name,j,z_th)
        #print(np.mean(img_2D))
        whole_img[y_th:y_th+dim_elem_num[1],x_th:x_th+dim_elem_num[0]]=img_2D
    return whole_img

def pyr_down_img(img,times=4):
    img_down=cv2.pyrDown(img)
    for i in range(times-1):
        img_down=cv2.pyrDown(img_down)
    return img_down

def adjust_contrast(img1,img2):
    img1,img2=img1.astype('float32'),img2.astype('float32')
    m1,m2=np.mean(img1),np.mean(img2)
    m=np.max((m1,m2))
    img1,img2=np.uint8(np.clip(m/m1*img1,0,255)),np.uint8(np.clip(m/m2*img2,0,255))
    return img1,img2

def loss_fun(ovl1,ovl2):
    ovl1,ovl2=ovl1.astype('float32'),ovl2.astype('float32')
    loss=np.sum((ovl1-ovl2)**2)/np.sqrt(np.sum(ovl1**2)*np.sum(ovl2**2))
    return loss
    
def calculate_xy_shift_by_RANSAC(img1,img2,pts1,pts2,loss_threshold=1,sample_time=1000):
    count=0
    matches_num=pts1.shape[0]
    RANSAC_num=np.int32(np.min((4,matches_num*0.1)))
    #print(RANSAC_num)
    loss_threshold=0.05
    loss_min=np.inf
    xy_shift_min=np.array([np.inf,np.inf])
    while(loss_min>loss_threshold and count<sample_time):
        count+=1
        index_list=random.sample(range(matches_num),RANSAC_num)
        #print(index_list)
        xy_shift_all=pts2[index_list,:]-pts1[index_list,:]
        max_shift,min_shift=np.max(xy_shift_all,axis=0),np.min(xy_shift_all,axis=0)
        if any((max_shift-min_shift)>100):
            continue
        xy_shift=np.int32(np.round(np.mean(xy_shift_all,axis=0)))#XY
        if all(xy_shift==xy_shift_min):
            continue
        ovl1,ovl2=img1[np.max((0,-xy_shift[1])):,np.max((0,-xy_shift[0])):],img2[np.max((0,xy_shift[1])):,np.max((0,xy_shift[0])):]
        x_range,y_range=np.min((ovl1.shape[1],ovl2.shape[1])),np.min((ovl1.shape[0],ovl2.shape[0]))
        ovl1,ovl2=ovl1[0:y_range,0:x_range],ovl2[0:y_range,0:x_range]
        #print(ovl1.shape,ovl2.shape)
        this_loss=loss_fun(ovl1,ovl2)
        #print(xy_shift,this_loss)
        if this_loss<loss_min:
            loss_min=this_loss
            xy_shift_min=xy_shift
    #print(xy_shift_min,loss_min)
    return xy_shift_min,loss_min

def calculate_xy_shift_by_BF(img1_down,img2_down,img1,img2,xy_shift):
    ################################################################
    xy_shift_min=np.zeros(0)
    loss_min=np.inf
    for x in range(-5,6):
        for y in range(-5,6):
            this_xy_shift=xy_shift+np.array([x,y],dtype='int32')
            ovl1=img1_down[np.max((0,-this_xy_shift[1])):,np.max((0,-this_xy_shift[0])):]
            ovl2=img2_down[np.max((0,this_xy_shift[1])):,np.max((0,this_xy_shift[0])):]
            x_range,y_range=np.min((ovl1.shape[1],ovl2.shape[1])),np.min((ovl1.shape[0],ovl2.shape[0]))
            ovl1,ovl2=ovl1[0:y_range,0:x_range],ovl2[0:y_range,0:x_range]
            this_loss=loss_fun(ovl1,ovl2)
            if this_loss<loss_min:
                loss_min=this_loss
                xy_shift_min=this_xy_shift
            #print(this_xy_shift,this_loss)
    #print('first ',xy_shift_min,loss_min)
    ################################################################
    img1_down2=pyr_down_img(img1,times=2)
    img2_down2=pyr_down_img(img2,times=2)
    img1_down,img2_down=adjust_contrast(img1_down,img2_down)
    xy_shift_whole=xy_shift_min*4
    xy_shift_min=np.zeros(0)
    loss_min=np.inf
    for x in range(-18,19,3):
        for y in range(-18,19,3):
            this_xy_shift=xy_shift_whole+np.array([x,y],dtype='int32')
            ovl1=img1_down2[np.max((0,-this_xy_shift[1])):,np.max((0,-this_xy_shift[0])):]
            ovl2=img2_down2[np.max((0,this_xy_shift[1])):,np.max((0,this_xy_shift[0])):]
            x_range,y_range=np.min((ovl1.shape[1],ovl2.shape[1],2000)),np.min((ovl1.shape[0],ovl2.shape[0],2000))
            ovl1,ovl2=ovl1[0:y_range,0:x_range],ovl2[0:y_range,0:x_range]
#             if x==0 and y==0:
#                 cv2.imshow('1',ovl1)
#                 cv2.imshow('2',ovl2)
#                 cv2.waitKey(5000)
#                 cv2.destroyAllWindows()
            this_loss=loss_fun(ovl1,ovl2)
            if this_loss<loss_min:
                loss_min=this_loss
                xy_shift_min=this_xy_shift
    xy_shift_whole=xy_shift_min
    for x in range(-2,3):
        for y in range(-2,3):
            this_xy_shift=xy_shift_whole+np.array([x,y],dtype='int32')
            ovl1=img1_down2[np.max((0,-this_xy_shift[1])):,np.max((0,-this_xy_shift[0])):]
            ovl2=img2_down2[np.max((0,this_xy_shift[1])):,np.max((0,this_xy_shift[0])):]
            x_range,y_range=np.min((ovl1.shape[1],ovl2.shape[1],2000)),np.min((ovl1.shape[0],ovl2.shape[0],2000))
            ovl1,ovl2=ovl1[0:y_range,0:x_range],ovl2[0:y_range,0:x_range]
            this_loss=loss_fun(ovl1,ovl2)
            if this_loss<loss_min:
                loss_min=this_loss
                xy_shift_min=this_xy_shift
    #print('second ',xy_shift_min,loss_min)
    ################################################################
    xy_shift_whole=xy_shift_min*4
    xy_shift_min=np.zeros(0)
    loss_min=np.inf
    x_start,y_start=int(0.2*img1.shape[1]),int(0.2*img1.shape[0])#从靠中间位置索引开始，取2000*2000的局部图
    for x in range(-18,19,3):
        for y in range(-18,19,3):
            this_xy_shift=xy_shift_whole+np.array([x,y],dtype='int32')
            ovl1=img1[np.max((0,-this_xy_shift[1]))+y_start:np.max((0,-this_xy_shift[1]))+y_start+3000,
                      np.max((0,-this_xy_shift[0]))+x_start:np.max((0,-this_xy_shift[0]))+x_start+3000]
            ovl2=img2[np.max((0,this_xy_shift[1]))+y_start:np.max((0,this_xy_shift[1]))+y_start+3000,
                      np.max((0,this_xy_shift[0]))+x_start:np.max((0,this_xy_shift[0]))+x_start+3000]
#             if x==0 and y==0:
#                 cv2.imshow('1',ovl1)
#                 cv2.imshow('2',ovl2)
#                 cv2.waitKey(5000)
#                 cv2.destroyAllWindows()
            this_loss=loss_fun(ovl1,ovl2)
            if this_loss<loss_min:
                loss_min=this_loss
                xy_shift_min=this_xy_shift
            #print(this_xy_shift,this_loss)
    #print(xy_shift_min,loss_min)
    xy_shift_whole=xy_shift_min
    for x in range(-2,3):
        for y in range(-2,3):
            this_xy_shift=xy_shift_whole+np.array([x,y],dtype='int32')
            ovl1=img1[np.max((0,-this_xy_shift[1]))+y_start:np.max((0,-this_xy_shift[1]))+y_start+3000,
                      np.max((0,-this_xy_shift[0]))+x_start:np.max((0,-this_xy_shift[0]))+x_start+3000]
            ovl2=img2[np.max((0,this_xy_shift[1]))+y_start:np.max((0,this_xy_shift[1]))+y_start+3000,
                      np.max((0,this_xy_shift[0]))+x_start:np.max((0,this_xy_shift[0]))+x_start+3000]
            this_loss=loss_fun(ovl1,ovl2)
            if this_loss<loss_min:
                loss_min=this_loss
                xy_shift_min=this_xy_shift
            #print(this_xy_shift,this_loss)
    #print('third ',xy_shift_min,loss_min)
    ################################################################
    return xy_shift_min,loss_min
            
def update_lower_layer_info(xy_shift,tile_pos,axis_range1,axis_range2,dim_len,voxel_len):
    tile_pos[:,0]=tile_pos[:,0]-axis_range2[0,0]+axis_range1[0,0]-xy_shift[0]*voxel_len[0]
    tile_pos[:,1]=tile_pos[:,1]-axis_range2[1,0]+axis_range1[1,0]-xy_shift[1]*voxel_len[1]
    axis_range2[0,0],axis_range2[0,1]=np.min(tile_pos[:,0]),np.max(tile_pos[:,0])+dim_len[0]
    axis_range2[1,0],axis_range2[1,1]=np.min(tile_pos[:,1]),np.max(tile_pos[:,1])+dim_len[1]
    voxel_num=np.uint64(np.round((axis_range2[:,1]-axis_range2[:,0])/voxel_len))
    return tile_pos,axis_range2,voxel_num
            
def start_vertical_stitch(i,img_path,save_path,file_name,img_name,overlap_ratio=0.3,pyr_times=4):
    sift=cv2.SIFT_create()
    bf=cv2.BFMatcher()
    if i==0:
        tile_pos1=np.load(save_path+r'\tile_pos_stitch_%.4d.npy'%(i))
        axis_range1=np.load(save_path+r'\axis_range_stitch_%.4d.npy'%(i))
        index1=np.load(save_path+r'\first_last_index_%.4d.npy'%(i))
        np.save(save_path+r'\axis_range_zstitch_%.4d.npy'%(i),axis_range1)
        np.save(save_path+r'\first_last_index_zstitch_%.4d.npy'%(i),index1)
        np.save(save_path+r'\tile_pos_zstitch_%.4d.npy'%(i),tile_pos1)
    img_path1,img_path2=img_path+'\\'+file_name%(i),img_path+'\\'+file_name%(i+1)
    dim_elem_num1,dim_len1,voxel_len=get_img_info(i,save_path)
    dim_elem_num2,dim_len2,voxel_len=get_img_info(i+1,save_path)
    axis_range1,axis_range2=np.load(save_path+r'\axis_range_zstitch_%.4d.npy'%(i)),np.load(save_path+r'\axis_range_stitch_%.4d.npy'%(i+1))
    voxel_num1,voxel_num2=np.uint64(np.round((axis_range1[:,1]-axis_range1[:,0])/voxel_len)),np.uint64(np.round((axis_range2[:,1]-axis_range2[:,0])/voxel_len))
    index1,index2=np.load(save_path+r'\first_last_index_%.4d.npy'%(i)),np.load(save_path+r'\first_last_index_%.4d.npy'%(i+1))
    index1,index2=index1[1],index2[0]
    tile_pos1,tile_pos2=np.load(save_path+r'\tile_pos_zstitch_%.4d.npy'%(i)),np.load(save_path+r'\tile_pos_stitch_%.4d.npy'%(i+1))
    tile_num1,tile_num2=tile_pos1.shape[0],tile_pos2.shape[0]
    img1=get_whole_img(index1,img_path1,img_name,axis_range1,dim_elem_num1,voxel_num1,voxel_len,tile_pos1)
    step=3
    loss_min=np.inf
    index_min=-1
    xy_shift_min=np.zeros(0)
    for j in range(index2,np.int32(np.round(voxel_num2[2]*overlap_ratio)+index2),step):
        img2=get_whole_img(j,img_path2,img_name,axis_range2,dim_elem_num2,voxel_num2,voxel_len,tile_pos2)
        img2_down=pyr_down_img(img2,pyr_times)
        img1_down=pyr_down_img(img1,pyr_times)
        img1_down,img2_down=adjust_contrast(img1_down,img2_down)
        kpts1,des1=sift.detectAndCompute(img1_down,None)
        kpts2,des2=sift.detectAndCompute(img2_down,None)
        kp1,kp2=np.float32([kp.pt for kp in kpts1]),np.float32([kp.pt for kp in kpts2])
        matches=bf.knnMatch(des1,des2,k=2)
        good_matches=[]
#        good=[]
        for m in matches:
            if len(m)==2 and m[0].distance<0.75*m[1].distance:
                good_matches.append((m[0].queryIdx,m[0].trainIdx))
#                good.append(m[0])
        #print(len(good))
#         img3=cv2.drawMatches(img1_down,kpts1,img2_down,kpts2,good,None,flags=2)
#         cv2.imshow('3',img3)
#         cv2.waitKey(5000)
#         cv2.destroyAllWindows()
        pts1,pts2=np.float32([kp1[i,:] for (i,_) in good_matches]),np.float32([kp2[j,:] for (_,j) in good_matches])
        xy_shift,this_loss=calculate_xy_shift_by_RANSAC(img1_down,img2_down,pts1,pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f'%(j,str(xy_shift),this_loss))
        xy_shift,this_loss=calculate_xy_shift_by_BF(img1_down,img2_down,cv2.medianBlur(img1,5),cv2.medianBlur(img2,5),xy_shift)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f'%(j,str(xy_shift),this_loss))
        if this_loss<loss_min:
            loss_min=this_loss
            xy_shift_min=xy_shift
            index_min=j
            
    for j in range(np.max((index_min-step,index2)),index_min+step+1):
        img2=get_whole_img(j,img_path2,img_name,axis_range2,dim_elem_num2,voxel_num2,voxel_len,tile_pos2)
        img2_down=pyr_down_img(img2,pyr_times)
        img1_down=pyr_down_img(img1,pyr_times)
        img1_down,img2_down=adjust_contrast(img1_down,img2_down)
        kpts1,des1=sift.detectAndCompute(img1_down,None)
        kpts2,des2=sift.detectAndCompute(img2_down,None)
        kp1,kp2=np.float32([kp.pt for kp in kpts1]),np.float32([kp.pt for kp in kpts2])
        matches=bf.knnMatch(des1,des2,k=2)
        good_matches=[]
        for m in matches:
            if len(m)==2 and m[0].distance<0.75*m[1].distance:
                good_matches.append((m[0].queryIdx,m[0].trainIdx))
        pts1,pts2=np.float32([kp1[i,:] for (i,_) in good_matches]),np.float32([kp2[j,:] for (_,j) in good_matches])
        xy_shift,this_loss=calculate_xy_shift_by_RANSAC(img1_down,img2_down,pts1,pts2)
        print('%d.th layer for pyrDown image, xy_shift is %s, loss is %.8f'%(j,str(xy_shift),this_loss))
        xy_shift,this_loss=calculate_xy_shift_by_BF(img1_down,img2_down,cv2.medianBlur(img1,5),cv2.medianBlur(img2,5),xy_shift)
        print('%d.th layer for whole image, xy_shift is %s, loss is %.8f'%(j,str(xy_shift),this_loss))
        if this_loss<loss_min:
            loss_min=this_loss
            xy_shift_min=xy_shift
            index_min=j
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Finally the matched one is %d.th layer, xy_shift is %s, loss is %.8f'%(index_min,str(xy_shift_min),loss_min))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    index2=np.load(save_path+r'\first_last_index_%.4d.npy'%(i+1))
    index2[0]=index_min
    tile_pos2,axis_range2,voxel_num2=update_lower_layer_info(xy_shift,tile_pos2,axis_range1,axis_range2,dim_len2,voxel_len)
    print('start saving data')
    np.save(save_path+r'\axis_range_zstitch_%.4d.npy'%(i+1),axis_range2)
    np.save(save_path+r'\first_last_index_zstitch_%.4d.npy'%(i+1),index2)
    np.save(save_path+r'\tile_pos_zstitch_%.4d.npy'%(i+1),tile_pos2)
    print('end saving data')


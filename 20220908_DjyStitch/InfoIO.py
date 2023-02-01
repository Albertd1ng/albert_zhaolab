from TileCont import *

from lxml import etree
import numpy as np
import re
import os


def get_img_xml_info(xml_path):
    r'''
    Read xml file and extract the information of dimensions and each tile.

    XML Form
    ----------
    <Dimensions>
    <DimensionDescription DimID="1" NumberOfElements="512" Length="2.214287e-04"/>
    <DimensionDescription DimID="2" NumberOfElements="512" Length="2.214287e-04"/>
    <DimensionDescription DimID="3" NumberOfElements="126" Length="2.499821e-04"/>
    <DimensionDescription DimID="10" NumberOfElements="1628" Length="1.627000e+03"/>
    </Dimensions>
    <Attachment Name="TileScanInfo" Application="LAS AF" FlipX="0" FlipY="0" SwapXY="0">
    <Tile FieldX="0" FieldY="0" PosX="0.0154545825" PosY="0.0209818193" PosZ="-0.0001240090"/>
    <Tile FieldX="1" FieldY="0" PosX="0.0156538684" PosY="0.0209818193" PosZ="-0.0001240090"/>
    <Tile FieldX="2" FieldY="0" PosX="0.0158531542" PosY="0.0209818193" PosZ="-0.0001240090"/>
    ......
    <Tile FieldX="24" FieldY="0" PosX="0.0146574392" PosY="0.0213803910" PosZ="-0.0001240090"/>
    </Attachment>

    Parameters
    ----------
    xml_path - str
        the path for xml file, usually exists in MetaData dir under image path

    Returns
    ----------
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    tile_num - int
        the quantity of tiles.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile

    Examples
    ----------
    >>>dim_elem_num,dim_len,voxel_len,tile_num,tile_pos=get_img_xml_info(r'D:\Data\Imaging\20220219\MetaData\Region.xml')
    '''
    dim_elem_num = np.zeros(3, dtype='uint')
    dim_len = np.zeros(3)
    parser = etree.XMLParser()
    my_et = etree.parse(xml_path, parser=parser)
    dim_attrib = my_et.xpath('//Dimensions/DimensionDescription')
    for i in range(3):
        dim_elem_num[i], dim_len[i] = dim_attrib[i].attrib['NumberOfElements'], dim_attrib[i].attrib['Length']
    voxel_len = dim_len / dim_elem_num
    tile_attrib = my_et.xpath('//Attachment/Tile')
    tile_num = len(tile_attrib)
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    for i in range(tile_num):
        tile_pos[i, :] = [tile_attrib[i].attrib['PosX'], tile_attrib[i].attrib['PosY'], 0]
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos


def get_img_xml_info_Z_stit(xml_path):
    r'''
    Read xml file and extract the information of dimensions and each tile.
    Usually used in Z dimension stitched.

    XML Form
    ----------
    <Dimensions>
    <DimensionDescription DimID="1" NumberOfElements="512" Length="2.214287e-04"/>
    <DimensionDescription DimID="2" NumberOfElements="512" Length="2.214287e-04"/>
    <DimensionDescription DimID="3" NumberOfElements="126" Length="2.499821e-04"/>
    </Dimensions>

    Parameters
    ----------
    xml_path - str
        the path for xml file, usually exists in MetaData dir under image path

    Returns
    ----------
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.

    Exmaples
    ----------
    >>>dim_elem_num,dim_len,voxel_len=get_img_xml_info_Z_stit(r'D:\Data\Imaging\20220219\MetaData\Region.xml')
    '''
    dim_elem_num = np.zeros(3, dtype='uint')
    dim_len = np.zeros(3)
    parser = etree.XMLParser()
    my_et = etree.parse(xml_path, parser=parser)
    dim_attrib = my_et.xpath('//Dimensions/DimensionDescription')
    for i in range(3):
        dim_elem_num[i], dim_len[i] = dim_attrib[i].attrib['NumberOfElements'], dim_attrib[i].attrib['Length']
    voxel_len = dim_len / dim_elem_num
    return dim_elem_num, dim_len, voxel_len


def get_img_txt_info(txt_path):
    r'''
    Read txt file and extract the information of dimensions and each tile.

    txt Form
    ----------
    AA01 1000.00 1000.00
    AA02 1500.00 1000.00
    AA03 2000.00 1000.00
    ...
    AQ20 5000.00 5000.00
    Z-Pos -1000.00
    VoxelSize 0.43 0.43 1.50
    VoxelNum 1024 1024 200

    Parameters
    ----------
    txt_path - str
        the path for txt file, usually exists in MetaData dir under image path

    Returns
    ----------
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    tile_num - int
        the quantity of tiles.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile
    tile_id_dict - dict, {str : int}, (tile_num)
        store the tile_id and related ordinal number, such as {'AA01':0, 'AA02':1,...}

    Examples
    ----------
    >>>dim_elem_num,dim_len,voxel_len,tile_num,tile_pos,tile_id_dict=get_img_txt_info(r'D:\Data\Imaging\20220219\MetaData\Region.xml')
    '''
    dim_elem_num = np.zeros(3, dtype='uint')
    dim_len = np.zeros(3)
    tile_num = 0
    voxel_len = np.zeros(3, dtype='float64')
    empty_line = 0
    tile_id_dict = {}
    with open(txt_path, 'r', errors='ignore') as f:
        while (True):
            one_line = re.sub('\x00', '', f.readline()).split()
            if len(one_line) != 0:
                empty_line = 0
                if re.match(r'[A-Z][A-Z][0-9][0-9]', one_line[0]) != None:
                    tile_num += 1
            else:
                empty_line += 1
                if empty_line >= 5:
                    empty_line = 0
                    break
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    this_tile_num = 0
    with open(txt_path, 'r', errors='ignore') as f:
        while (True):
            one_line = re.sub('\x00', '', f.readline()).split()
            if len(one_line) != 0:
                empty_line = 0
                if re.match(r'[A-Z][A-Z][0-9][0-9]', one_line[0]) != None:
                    tile_pos[this_tile_num, 0] = np.float64(re.match(r'(-|)\d+(\.|)\d*', one_line[1]).group())
                    tile_pos[this_tile_num, 1] = np.float64(re.match(r'(-|)\d+(\.|)\d*', one_line[2]).group())
                    tile_id_dict[re.match(r'[A-Z][A-Z][0-9][0-9]', one_line[0]).group()] = this_tile_num
                    this_tile_num += 1
                elif re.match(r'Z-Pos', one_line[0]) != None:
                    tile_pos[:, 2] = np.float64(one_line[1])
                elif re.match(r'VoxelSize', one_line[0]) != None:
                    voxel_len[0] = np.float64(one_line[1])
                    voxel_len[1] = np.float64(one_line[2])
                    voxel_len[2] = np.float64(one_line[3])
                elif re.match(r'VoxelNum', one_line[0]) != None:
                    dim_elem_num[0] = np.uint(one_line[1])
                    dim_elem_num[1] = np.uint(one_line[2])
                    dim_elem_num[2] = np.uint(one_line[3])
            else:
                empty_line += 1
                if empty_line >= 5:
                    empty_line = 0
                    break
    dim_len = voxel_len * dim_elem_num
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos, tile_id_dict

def get_img_npy_basic_info(layer_num, save_path):
    r'''
    Parameters
    ----------
    layer_num - int
    save_path - str

    Returns
    ----------
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    '''
    dim_elem_num = np.load(save_path + r'\dim_elem_num_%.4d.npy' % (layer_num))
    dim_len = np.load(save_path + r'\dim_len_%.4d.npy' % (layer_num))
    voxel_len = np.load(save_path + r'\voxel_len.npy')
    return dim_elem_num, dim_len, voxel_len

def get_img_info_hori_stit(layer_num,save_path):
    r'''
    Parameters
    ----------
    layer_num - int
    save_path - str
    tile_pos_stitch - array, float64, (tile_num,3)
    axis_range - array, float64, (3,2)
    first_last_index -vector, uint32, (2)
    '''
    tile_pos_stitch=np.load(save_path+r'\tile_pos_stitch_%.4d.npy'%(layer_num))
    axis_range=np.load(save_path + r'\axis_range_stitch_%.4d.npy' % (layer_num))
    first_last_index=np.load(save_path + r'\first_last_index_%.4d.npy' % (layer_num))
    return tile_pos_stitch,axis_range,first_last_index

def get_img_info_vert_stit(layer_num,save_path):
    tile_pos_stitch = np.load(save_path + r'\tile_pos_zstitch_%.4d.npy' % (layer_num))
    axis_range = np.load(save_path + r'\axis_range_zstitch_%.4d.npy' % (layer_num))
    first_last_index = np.load(save_path + r'\first_last_index_zstitch%.4d.npy' % (layer_num))
    return tile_pos_stitch, axis_range, first_last_index

def get_save_tile_cont_info(layer_num, save_path, dim_len, tile_pos, tile_cont_thre=0.6):
    r'''
    Get the npy file containing tile contact information of all tiles for one layer.
    If the npy file does not exist, judge tile contact and save the contact bool array.

    Parameters
    ----------
    layer_num - int
    save_path - str
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile
    tile_cont_thre - float
        the thredshold to judge if two tile contact with each other. if two tiles align in X direction,
        they can be judged as contacted tiles when the overlap ratios in y and z directions exceed tile_cont_thre

    Returns
    ----------
    tile_contact - array, bool, (tile_num,tile_num)
        the tile contact array for all tiles. tile_contact[i,j] means whether i.th tile and j.th tile have overlapped region.
    '''
    save_path_file = os.listdir(save_path)
    if r'tile_contact_%.4d.npy' % (layer_num) in save_path_file:
        return np.load(save_path + r'\tile_contact_%.4d.npy' % (layer_num))
    else:
        tile_contact = judge_tile_cont(dim_len, tile_pos, tile_cont_thre=tile_cont_thre)
        np.save(save_path + r'\tile_contact_%.4d.npy' % (layer_num), tile_contact)
        return tile_contact


def save_img_info(layer_num, save_path, dim_elem_num, dim_len, voxel_len, tile_num, tile_pos):
    r'''
    Save neccesary information of one layer in .npy file for subsequent computation.

    Parameters
    ----------
    layer_num - int
    save_path - str
    dim_elem_num - vector, unit32, (3)
        the quantity of voxels for each dimension.
    dim_len - vector, float64, (3)
        the length of each dimension for one 3D image.
    voxel_len - vector, float64, (3)
        the length of each dimension for a voxel.
    tile_num - int
        the quantity of tiles.
    tile_pos - array, float64, (tile_num,3)
        the XYZ position information of each tile

    Examples
    ----------
    >>>save_img_info(1,r'C:\Users\dingj\20220902_BfHoriSiftVertStitch',dim_elem_num,dim_len,voxel_len,tile_num,tile_pos)
    '''
    np.save(save_path + r'\dim_elem_num_%.4d.npy' % (layer_num), dim_elem_num)
    np.save(save_path + r'\dim_len_%.4d.npy' % (layer_num), dim_len)
    np.save(save_path + r'\voxel_len.npy', voxel_len)
    np.save(save_path + r'\tile_num_%.4d.npy' % (layer_num), tile_num)
    np.save(save_path + r'\tile_pos_%.4d.npy' % (layer_num), tile_pos)

def save_img_info_hori_stit(layer_num,save_path,tile_pos_stitch,axis_range,first_last_index):
    r'''
    Parameters
    ----------
    layer_num - int
    save_path - str
    tile_pos_stitch - array, float64, (tile_num,3)
    axis_range - array, float64, (3,2)
    first_last_index -vector, uint32, (2)
    '''
    np.save(save_path+r'\tile_pos_stitch_%.4d.npy'%(layer_num),tile_pos_stitch)
    np.save(save_path + r'\axis_range_stitch_%.4d.npy' % (layer_num), axis_range)
    np.save(save_path + r'\first_last_index_%.4d.npy' % (layer_num), first_last_index)

def save_img_info_vert_stit(layer_num,save_path,tile_pos,axis_range,first_last_index):
    np.save(save_path + r'\tile_pos_zstitch_%.4d.npy' % (layer_num), tile_pos)
    np.save(save_path + r'\axis_range_zstitch_%.4d.npy' % (layer_num), axis_range)
    np.save(save_path + r'\first_last_index_zstitch_%.4d.npy' % (layer_num), first_last_index)

def save_img_info_vert_stit_merged(layer_num,save_path,dim_elem_num,axis_range,first_last_index):
    np.save(save_path + r'\dim_elem_num_zstitch_%.4d' % (layer_num), dim_elem_num)
    np.save(save_path + r'\axis_range_zstitch_%.4d' % (layer_num), axis_range)
    np.save(save_path + r'\first_last_index_zstitch_%.4d' % (layer_num), first_last_index)

def get_img_info_vert_stit_merged(layer_num,save_path):
    dim_elem_num=np.save(save_path + r'\dim_elem_num_zstitch_%.4d' % (layer_num))
    axis_range=np.save(save_path + r'\axis_range_zstitch_%.4d' % (layer_num))
    first_last_index=np.save(save_path + r'\first_last_index_zstitch_%.4d' % (layer_num))
    return dim_elem_num,axis_range,first_last_index
from xml.etree import ElementTree as ET
from readlif.reader import LifFile
import numpy as np
import re
import os
import nd2


def get_img_nd2_info(img):
    dim_elem_num = np.zeros(3, dtype='int64')
    dim_elem_num[0] = img.sizes['X']
    dim_elem_num[1] = img.sizes['Y']
    dim_elem_num[2] = img.sizes['Z']
    voxel_len = np.zeros(3, dtype='float64')
    voxel_len[0] = img.voxel_size().x
    voxel_len[1] = img.voxel_size().y
    voxel_len[2] = img.voxel_size().z
    dim_len = dim_elem_num * voxel_len
    tile_pos_list = img.experiment[0].parameters.points
    tile_num = len(tile_pos_list)
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    for i in range(tile_num):
        tile_pos[i, :] = np.array(tile_pos_list[i].stagePositionUm, dtype='float64')
    dim_num = img.ndim
    img_data_type = str(img.dtype)
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos, dim_num, img_data_type


def get_img_nd2_info_vert(img):
    dim_elem_num = np.zeros(3, dtype='int64')
    dim_elem_num[0] = img.sizes['X']
    dim_elem_num[1] = img.sizes['Y']
    dim_elem_num[2] = img.sizes['Z']
    dim_num = img.ndim
    img_data_type = str(img.dtype)
    return dim_elem_num, dim_num, img_data_type


def get_img_lif_info(img):
    lif_root = img.xml_root
    dim_elem_num = []
    dim_len = []
    for i in lif_root.iter('TimeStampList'):
        i.text = ''
    for i in lif_root.iter('DimensionDescription'):
        dim_elem_num.append(int(i.attrib['NumberOfElements']))
        dim_len.append(np.float64(i.attrib['Length']))
    tile_num = dim_elem_num[3]
    dim_elem_num = np.array(dim_elem_num[:3], dtype='int64')
    dim_len = np.array(dim_len[:3], dtype='float64')
    voxel_len = dim_len / dim_elem_num
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    this_num = 0
    for i in lif_root.iter('Tile'):
        tile_pos[this_num, 0] = np.float64(i.attrib['PosX'])
        tile_pos[this_num, 1] = np.float64(i.attrib['PosY'])
        this_num = this_num + 1
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos


def get_img_xml_info(xml_path):
    r"""
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
    """
    tree = ET.parse(xml_path)
    this_root = tree.getroot()
    dim_elem_num = []
    dim_len = []
    for i in this_root.iter('TimeStampList'):
        i.text = ''
    for i in this_root.iter('DimensionDescription'):
        dim_elem_num.append(int(i.attrib['NumberOfElements']))
        dim_len.append(np.float64(i.attrib['Length']))
    tile_num = dim_elem_num[3]
    dim_elem_num = np.array(dim_elem_num[:3], dtype='int64')
    dim_len = np.array(dim_len[:3], dtype='float64')
    voxel_len = dim_len / dim_elem_num
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    this_num = 0
    for i in this_root.iter('Tile'):
        tile_pos[this_num, 0] = np.float64(i.attrib['PosX'])
        tile_pos[this_num, 1] = np.float64(i.attrib['PosY'])
        this_num = this_num + 1
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos

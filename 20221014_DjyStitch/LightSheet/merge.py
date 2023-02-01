#! python
# @Time    : 22/10/03 下午 10:34
# @Author  : azzhu 
# @FileName: merge.py
# @Software: PyCharm
import os
import time
from multiprocessing import Pool
from pathlib import Path
from loguru import logger

import cv2
import pickle
import numpy as np
from scipy import ndimage as ndi
# from calculate_tile_pos_20220610 import get_file_list

# params
datadir1 = Path('/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/220908/frames')
datadir2 = Path('/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/220923/frames')
outputdir = Path('/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi/out220929/src_1')  # 原始未降采样结果
fileend = '0'  # 针对多通道情况，先处理哪一个通道（文件名以什么结尾），不设置设为None
outputdir_ds = None  # 降采样结果
downsample = 1  # 降采样倍数
cor_p = 'position_file_all_ds332.txt'
# img_size = [2048, 1024]  # X,Y
img_size = [1024, 2048]  # X,Y

CPUs = 55  # 使用多少核心并行加速
# task_list = list(range(10000, 17031)) # 已跑上
# CPUs = 35  # 使用多少核心并行加速
task_list = list(range(5000))  # ai04已跑上
# task_list = list(range(5000, 10000))  # ai03已跑上

# alias
u8 = np.uint8
u16 = np.uint16
f16 = np.float16
f32 = np.float32

logger_file = 'merge_ch1_fat03_3.log'
if Path(logger_file).exists():
    os.remove(logger_file)
logger.add(logger_file)

'''
'''


def load_cor():
    start_line = 4
    lines = open(cor_p, 'r').readlines()
    lines = lines[start_line:]

    # vn = {}
    # for line in lines:
    #     key = line.split('_')[0]
    #     if key not in vn:
    #         vn[key] = 1
    #     else:
    #         vn[key] += 1
    #
    # lines = [_ for _ in lines if _.startswith('Z-24122.57')]

    res = {}
    for line in lines:
        line = line.strip().replace(' ', '')
        s = line.split(';;')
        key = s[0].split('.tif')[0]
        v = s[1].replace('(', '').replace(')', '').split(',')
        v = list(map(float, v))
        res[key] = np.round(np.array(v)).astype(np.int)
    res = {k: res[k] for k in sorted(res)}
    return res


def load_range():
    """
    22.09.10 新数据
    返回的坐标信息：[z,x,y]
    """
    name = get_file_list()
    res = {}  # 左上角坐标
    for line in name:
        zxy = []
        for v in line.split('_'):
            v = v[1:]
            if not v[0].isdigit(): v = v[1:]
            zxy.append(float(v))
        if len(zxy) != 3:
            print('error: ', line)
        res[line] = np.round(np.array(zxy)).astype(np.int)

    with open('fileinfo.pkl', 'rb') as f:
        fileinfo = pickle.load(f)  # 每个文件夹（一条3D图像）的size

    # 整合信息，输入每个维度的range
    ran = {}
    for k in res:
        c = res[k]
        fi = fileinfo[k][1:-1].split(',')
        fi = list(map(int, fi))
        s = np.array([fi[0], fi[2], fi[3]])
        r = c + s
        ran[k] = np.stack([c, r], -1)

    return ran


class Pbar():
    def __init__(self, total, pbar_len=50, pbar_value='|', pbar_blank='-'):
        self.total = total
        self.pbar_len = pbar_len
        self.pbar_value = pbar_value
        self.pbar_blank = pbar_blank
        self.now = 0
        self.last = self.now
        self.time = time.time()
        self.start_time = time.time()
        self.close_flag = False

    def update(self, nb=1, set=False, set_value=None):
        if set:
            self.now = set_value
        else:
            self.now += nb
        percent = int(round(self.now / self.total * 100))  # 百分比数值
        pbar_now = round(self.pbar_len * percent / 100)  # 进度条当前长度
        if pbar_now > self.pbar_len: pbar_now = self.pbar_len  # 允许now>total，但是不允许pbar_now>pbar_len
        blank_len = self.pbar_len - pbar_now
        time_used = time.time() - self.time  # 当前更新耗时
        eps = 1e-4  # 一个比较小的值，加在被除数上，防止除零
        speed = (self.now - self.last) / (time_used + eps)  # 速度

        total_time_used = time.time() - self.start_time  # 总耗时
        total_time_used_min, total_time_used_sec = divmod(total_time_used, 60)
        total_time_used = f'{int(total_time_used_min):0>2d}:{int(total_time_used_sec):0>2d}'

        # # 根据瞬时速度来算剩余时间
        # if speed != 0:
        #     remaining_it = self.total - self.now if self.total - self.now >= 0 else 0  # 剩余任务
        #     remaining_time = remaining_it / speed  # 剩余时间
        #     remaining_time_min, remaining_time_sec = divmod(remaining_time, 60)
        #     remaining_time = f'{int(remaining_time_min):0>2d}:{int(remaining_time_sec):0>2d}'
        # else:
        #     remaining_time = 'null'

        # 根据总体进度来算剩余时间(now必须从0开始增加)
        if self.now != 0:
            remaining_time = (time.time() - self.start_time) / self.now * (self.total - self.now)
            remaining_time_min, remaining_time_sec = divmod(remaining_time, 60)
            remaining_time = f'{int(remaining_time_min):0>2d}:{int(remaining_time_sec):0>2d}'
        else:
            remaining_time = 'null'

        pbar = f'{percent:>3d}%|{self.pbar_value * pbar_now}{self.pbar_blank * blank_len}| ' \
               f'{self.now}/{self.total} [{total_time_used}<{remaining_time}, {speed:.2f}it/s]'
        print(f'\r{pbar}', end='')
        self.time = time.time()
        self.last = self.now

    def close(self, reset_done=True):
        if self.close_flag: return
        if reset_done:
            self.update(set=True, set_value=self.total)
        print()
        self.close_flag = True

    def __str__(self):
        return f'[total:{self.total} now:{self.now}]'


class MergeSolution():
    """
    算法原理：
        传入若干固定尺寸图像以及对应坐标，一个一个往大图像里面放，每次放的时候跟前面已经放好的使用过渡算法过渡。
    """

    def __init__(self, imgs: list, big_img_size, to_imgdtype=u16):
        '''

        :param imgs: list，不止有img，还要有对应坐标，形如这样[[img1,[x1,y1]],[img2,[x2,y2]],....]
        :param big_img_size: 要拼成的大图像的尺寸，比如这样[1000,2000]，分别对应高和宽
        :param to_imgdtype: 默认图像类型，比如uint16
        '''
        self.imgs, self.coord = [], []
        for img, coord in imgs:
            self.imgs.append(img)
            self.coord.append(coord)  # [x_r, y_r]
        self.result = np.zeros([big_img_size[0] + 2, big_img_size[1] + 2], f32)  # 往四周扩一个像素方便处理
        self.mask = np.zeros([big_img_size[0] + 2, big_img_size[1] + 2], bool)
        self.coord = np.array(self.coord) + 1  # 因为扩了一个像素，这里坐标值都加一
        self.to_imgdtype = to_imgdtype

    def do(self):
        # 第一张不用merge，直接放上
        img0 = self.imgs[0].astype(f32)
        xr, yr = self.coord[0]
        self.result[yr[0]:yr[1], xr[0]:xr[1]] = img0
        self.mask[yr[0]:yr[1], xr[0]:xr[1]] = True

        for i in range(1, len(self.imgs)):
            img = self.imgs[i].astype(f32)
            xr, yr = self.coord[i]
            mask = self.mask[yr[0]:yr[1], xr[0]:xr[1]].astype(u8)
            w_bg = ndi.distance_transform_edt(mask)
            mask = self.mask[yr[0] - 1:yr[1] + 1, xr[0] - 1:xr[1] + 1].copy()  # 往外扩一个像素来取mask
            mask = mask == False  # 值反转
            mask[1:-1, 1:-1] = True
            mask = mask.astype(u8)
            w_fg = ndi.distance_transform_edt(mask)
            w_fg = w_fg[1:-1, 1:-1]
            # w_fg = w_fg[1:w_bg.shape[0]+1, 1:w_bg.shape[1]+1]
            wsum = w_bg + w_fg
            w_bg /= wsum
            w_fg /= wsum
            self.result[yr[0]:yr[1], xr[0]:xr[1]] = \
                w_bg * self.result[yr[0]:yr[1], xr[0]:xr[1]] + \
                w_fg * img
            self.mask[yr[0]:yr[1], xr[0]:xr[1]] = True  # 对应mask置True
        self.result = np.round(self.result[1:-1, 1:-1]).astype(self.to_imgdtype)
        return self.result

    def do_to_test(self):
        '''
        供测试，不使用过渡算法，且在边界和对角线地方画线
        :return:
        '''
        th = 5
        for i in range(len(self.imgs)):
            h, w = self.imgs[i].shape[:2]
            cv2.line(self.imgs[i], (0, 0), (w, 0), int(2 ** 16 - 1), thickness=th)
            cv2.line(self.imgs[i], (0, 0), (0, h), int(2 ** 16 - 1), thickness=th)
            cv2.line(self.imgs[i], (w, 0), (w, h), int(2 ** 16 - 1), thickness=th)
            cv2.line(self.imgs[i], (0, h), (w, h), int(2 ** 16 - 1), thickness=th)
            # cv2.line(self.imgs[i], (0, 0), (w, h), int(2 ** 16 - 1), thickness=th)
            # cv2.line(self.imgs[i], (w, 0), (0, h), int(2 ** 16 - 1), thickness=th)

            img = self.imgs[i].astype(f32)
            xr, yr = self.coord[i]
            self.result[yr[0]:yr[1], xr[0]:xr[1]] = img

        return self.result


def process_a_z(z):
    logger.info(f'begin_[{z}]:')
    with open('params/stacks_range.pkl', 'rb') as f:
        stacks_range = pickle.load(f)
    with open('params/stacks_file.pkl', 'rb') as f:
        stacks_file = pickle.load(f)
    with open('params/stack_id_to_stack_names.pkl', 'rb') as f:
        stack_id_to_stack_names = pickle.load(f)
    with open('params/largeimg_shape.pkl', 'rb') as f:
        largeimg_shape = pickle.load(f)
    with open('params/z_stacks.pkl', 'rb') as f:
        z_stacks = pickle.load(f)
    stacksnamelist = z_stacks[z]

    def get_imgs_and_coord():
        result = []
        for name in stacksnamelist:
            x_r, y_r = stacks_range[name][:2]
            file_id = z - stacks_range[name][-1][0]
            filepath = stacks_file[name][file_id]
            date_id, stack_id, tif_id = filepath.split()
            tif_p = f'/home/zhangli_lab/zhuqingjie/dataset/zhaohu/liyouqi' \
                    f'/2209{date_id}/frames/{stack_id_to_stack_names[int(stack_id)]}/{tif_id}.tif'
            img = cv2.imread(tif_p, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f'img is None: {tif_p}')
            result.append([img, [x_r, y_r]])
        return result

    imgs = get_imgs_and_coord()

    # t1 = time.time()
    ms = MergeSolution(imgs, big_img_size=largeimg_shape)
    result = ms.do()
    # result = ms.do_to_test()
    cv2.imwrite(str(outputdir / f'a_{z:05d}.tif'), result)
    if outputdir_ds is not None:
        result = cv2.resize(result, dsize=None, fx=1 / downsample, fy=1 / downsample)
        cv2.imwrite(str(outputdir_ds / f'a_{z:06d}.tif'), result)

    logger.info(f'finish_[{z}]!')


def gen_params_renturn_z_range():
    def gen_total_params():
        if Path('params/stacks_range.pkl').exists() and \
                Path('params/stacks_file.pkl').exists() and \
                Path('params/stack_id_to_stack_names.pkl').exists() and \
                Path('params/largeimg_shape.pkl').exists() and \
                Path('params/z_stacks.pkl').exists():
            with open('params/stacks_range.pkl', 'rb') as f:
                stacks_range = pickle.load(f)
            with open('params/stacks_file.pkl', 'rb') as f:
                stacks_file = pickle.load(f)
            with open('params/stack_id_to_stack_names.pkl', 'rb') as f:
                stack_id_to_stack_names = pickle.load(f)
            with open('params/largeimg_shape.pkl', 'rb') as f:
                largeimg_shape = pickle.load(f)
            with open('params/z_stacks.pkl', 'rb') as f:
                z_stacks = pickle.load(f)
            return stacks_range, stacks_file, stack_id_to_stack_names, largeimg_shape, z_stacks

        cor = load_cor()
        cor_v = np.array([v for k, v in cor.items()])
        maxxy = np.max(cor_v, axis=0)[:2] + np.array(img_size)  # 最大xy

        stack_names = get_file_list()
        stack_names_to_stack_id = {k: i for i, k in enumerate(stack_names)}
        stack_id_to_stack_names = {v: k for k, v in stack_names_to_stack_id.items()}

        stacks = {}
        for sn in stack_names:
            if Path(datadir1 / sn).exists():
                stacks[sn] = datadir1 / sn
            elif Path(datadir2 / sn).exists():
                stacks[sn] = datadir2 / sn
            else:
                raise NotImplementedError
        stacks = {f: stacks[f] for f in cor}  # 保持跟cor里面顺序对应
        print('stacks:', len(stacks))

        stacks_len = []  # 每个stack的z轴大小
        stacks_file = {}  # 每个stack对应的文件名
        # sortfn = lambda p: int(p.name.split('_')[1].split('.')[0])  # 文件名排序规则函数
        sortfn = lambda p: float(p.stem.split('_')[0]) + float(p.stem.split('_')[1]) / 10  # 文件名排序规则函数

        stacks_file_p = f'stacks_file_ch{fileend}.pkl'
        if Path(stacks_file_p).exists():
            with open(stacks_file_p, 'rb') as f:
                stacks_file = pickle.load(f)
                for _, v in stacks_file.items():
                    stacks_len.append(len(v))
        else:
            pb = Pbar(total=len(stacks))
            print('get file info...')
            for f in stacks:
                tifs = list(stacks[f].iterdir())
                if fileend is not None:
                    tifs = [_ for _ in tifs if _.stem.endswith(fileend)]
                tifs = sorted(tifs, key=sortfn)
                tifs = [f'{t.parts[-4][-2:]} {stack_names_to_stack_id[t.parts[-2]]} {t.parts[-1][:-4]}' for t in tifs]
                stacks_file[stack_names_to_stack_id[f]] = tifs
                stacks_len.append(len(tifs))
                pb.update()
            pb.close()
            with open(stacks_file_p, 'wb') as f:
                pickle.dump(stacks_file, f)
        # exit()

        maxz = np.max(np.array(stacks_len) + np.array(cor_v[:, -1]))  # 最大z
        print(f'Large image size:({maxxy[0]},{maxxy[1]},{maxz})')

        # 计算每个stack在整个图像中xyz的范围
        stacks_range = {}
        for k, v in cor.items():
            x_r = [v[0], v[0] + img_size[0]]  # 前闭后开
            y_r = [v[1], v[1] + img_size[1]]
            z_r = [v[2], v[2] + len(stacks_file[stack_names_to_stack_id[k]])]
            stacks_range[stack_names_to_stack_id[k]] = [x_r, y_r, z_r]
        # 计算每层z轴上分别有哪些stack
        z_stacks = [[] for _ in range(maxz)]
        for k, v in stacks_range.items():
            for z in range(*v[-1]):
                z_stacks[z].append(k)
        largeimg_shape = maxxy[::-1]

        with open('params/stacks_range.pkl', 'wb') as f:
            pickle.dump(stacks_range, f)
        with open('params/stacks_file.pkl', 'wb') as f:
            pickle.dump(stacks_file, f)
        with open('params/stack_id_to_stack_names.pkl', 'wb') as f:
            pickle.dump(stack_id_to_stack_names, f)
        with open('params/largeimg_shape.pkl', 'wb') as f:
            pickle.dump(largeimg_shape, f)
        with open('params/z_stacks.pkl', 'wb') as f:
            pickle.dump(z_stacks, f)

        return stacks_range, stacks_file, stack_id_to_stack_names, largeimg_shape, z_stacks

    stacks_range, stacks_file, stack_id_to_stack_names, largeimg_shape, z_stacks = gen_total_params()
    return len(z_stacks)


if __name__ == '__main__':
    # z_range = gen_params_renturn_z_range()
    # logger.info(f'z_range: [{z_range}]')
    # task_list = list(range(z_range))
    pool = Pool(CPUs)
    pool.map(process_a_z, task_list)

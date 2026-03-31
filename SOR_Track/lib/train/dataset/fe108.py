# lib/train/dataset/fe108.py
"""
FE108 Dataset — dual-frame 训练格式
对齐 VisEvent 的 get_frames 返回规范：
  (frame_list, anno_frames, object_meta, frame_event_img_list)

目录结构假设（fe108_gen.py 生成后）：
  {root}/{split}/{seq_name}/
  ├ img/                      RGB 帧，0001.jpg ~ NNNN.jpg（1-indexed）
  ├ event_stack_center/       Event 帧，0001.jpg ~ NNNN.jpg（1-indexed）
  └ groundtruth_rect.txt      [x, y, w, h]，逗号分隔
"""

import os
import csv
import numpy as np
import torch
import pandas
from collections import OrderedDict

from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import opencv_loader
from lib.train.admin import env_settings


class Fe108(BaseVideoDataset):
    """
    FE108 数据集，返回格式与 VisEvent 完全对齐。

    Args:
        root          : 数据集根目录，应指向 {FE108_ROOT}/{split}
                        （train 和 test 分开实例化）
        image_loader  : 图像读取函数，默认 opencv_loader
        split         : 'train' | 'val'（val 从 train 子集划分）
        seq_ids       : 手动指定序列索引列表
        data_fraction : 只使用前 N% 序列（调试用）
        align         : event_stack 对齐模式，对应 fe108_gen.py 的 --align
                        决定读取 event_stack_{align}/ 目录
    """

    def __init__(self,
                 root=None,
                 image_loader=opencv_loader,
                 split=None,
                 seq_ids=None,
                 data_fraction=None,
                 align: str = 'center'):

        root = env_settings().fe108_dir if root is None else root
        super().__init__('Fe108', root, image_loader)

        # event_stack 目录名由 align 参数决定
        self.align            = align
        self.event_stack_dir  = f'event_stack_{align}'

        self.sequence_list = self._get_sequence_list()

        #  split / seq_ids 路由 
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split and seq_ids.')
            if split == 'train':
                file_path = os.path.join(self.root, 'train.txt')
            elif split == 'val':
                file_path = os.path.join(self.root, 'val.txt')
            else:
                raise ValueError(f'[Fe108] Unknown split: {split!r}')
            seq_ids = (
                pandas.read_csv(file_path, header=None, dtype=np.int64)
                .squeeze("columns")
                .values.tolist()
            )
        elif seq_ids is None:
            seq_ids = list(range(len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            n = max(1, int(len(self.sequence_list) * data_fraction))
            self.sequence_list = self.sequence_list[:n]

    #  基础接口 
    def get_name(self):
        return 'fe108'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    # 序列列表 
    def _get_sequence_list(self) -> list:
        list_file = os.path.join(self.root, 'list.txt')
        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f'[Fe108] list.txt not found: {list_file}'
            )
        with open(list_file) as f:
            return [row[0] for row in csv.reader(f) if row]

    # 路径工具 
    def _get_seq_root(self, seq_id: int) -> str:
        """序列根目录"""
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_rgb_path(self, seq_id: int) -> str:
        return os.path.join(self._get_seq_root(seq_id), 'img')

    def _get_event_path(self, seq_id: int) -> str:
        return os.path.join(self._get_seq_root(seq_id), self.event_stack_dir)

    # GT 读取 
    def _read_bb_anno(self, seq_id: int) -> torch.Tensor:
        """
        读取 groundtruth_rect.txt
        格式：x, y, w, h（逗号分隔，与 VisEvent 相同格式）
        """
        gt_file = os.path.join(self._get_seq_root(seq_id),
                               'groundtruth_rect.txt')
        gt = pandas.read_csv(
            gt_file, delimiter=',', header=None,
            dtype=np.float32, na_filter=False, low_memory=False
        ).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id: int) -> dict:
        bbox    = self._read_bb_anno(seq_id)
        valid   = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    #  帧读取 

    def _get_frame_path(self, frame_dir: str, frame_id: int) -> str:
        """
        FE108 文件名规则：
          {frame_id+1:04d}.jpg   ← frame_id 是 0-indexed GT 行号
                                   文件名是 1-indexed（zfill(4)）
        支持 .jpg / .png 两种后缀（gen 脚本输出 jpg，少数可能为 png）
        """
        base = str(frame_id + 1).zfill(4)          # 0-indexed → 1-indexed
        for ext in ('.jpg', '.png', '.bmp'):
            p = os.path.join(frame_dir, base + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f'[Fe108] Frame not found in {frame_dir}: '
            f'tried {base}.jpg/.png/.bmp  (frame_id={frame_id})'
        )

    def _get_frame(self, frame_dir: str, frame_id: int) -> np.ndarray:
        return self.image_loader(self._get_frame_path(frame_dir, frame_id))

    # 主接口
    def get_frames(self,
                   seq_id: int,
                   frame_ids: list,
                   anno=None):
        """
        返回格式与 VisEvent.get_frames 完全一致：
          frame_list           : List[np.ndarray]  RGB 帧   (H,W,3)
          anno_frames          : Dict               逐帧标注
          object_meta          : OrderedDict
          frame_event_img_list : List[np.ndarray]  Event 帧 (H,W,3)

        注意：原 Fe108 返回 5 元组（含末尾 None），
              此处统一为 4 元组，与 VisEvent / sampler 期望对齐。
        """
        rgb_dir   = self._get_rgb_path(seq_id)
        event_dir = self._get_event_path(seq_id)

        # RGB 帧 
        frame_list = [
            self._get_frame(rgb_dir, f_id)
            for f_id in frame_ids
        ]

        # Event 帧 
        frame_event_img_list = [
            self._get_frame(event_dir, f_id)
            for f_id in frame_ids
        ]

        # 标注 
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {
            key: [value[f_id, ...].clone() for f_id in frame_ids]
            for key, value in anno.items()
        }

        # 元信息 
        object_meta = OrderedDict({
            'object_class_name': None,
            'motion_class'     : None,
            'major_class'      : None,
            'root_class'       : None,
            'motion_adverb'    : None,
        })

        return frame_list, anno_frames, object_meta, frame_event_img_list
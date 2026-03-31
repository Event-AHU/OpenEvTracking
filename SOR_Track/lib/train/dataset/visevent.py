# lib/train/dataset/visevent.py — 完整修正版

import os
import os.path
import numpy as np
import torch
import csv
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.data.image_loader import opencv_loader


class VisEvent(BaseVideoDataset):

    def __init__(self, root=None, image_loader=opencv_loader,
                 split=None, seq_ids=None, data_fraction=None):
        root = env_settings().visevent_dir if root is None else root
        super().__init__('VisEvent', root, image_loader)

        self.sequence_list = self._get_sequence_list()

        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split and seq_ids.')
            if split == 'train':
                file_path = os.path.join(self.root, 'train.txt')
            elif split == 'val':
                file_path = os.path.join(self.root, 'val.txt')
            else:
                raise ValueError(f'Unknown split: {split}')
            seq_ids = pandas.read_csv(
                file_path, header=None, dtype=np.int64
            ).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            n = max(1, int(len(self.sequence_list) * data_fraction))
            self.sequence_list = self.sequence_list[:n]

        # 缓存每个序列的帧起始偏移量，避免每次 get_frames 都重新扫描磁盘
        # key: seq_id(int) → offset(int)
        self._frame_offset_cache = {}

    # 基础接口 
    def get_name(self):
        return 'visevent'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    # 序列列表 
    def _get_sequence_list(self):
        list_file = os.path.join(self.root, 'list.txt')
        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"[VisEvent] list.txt not found at {list_file}"
            )
        with open(list_file) as f:
            dir_list = [row[0] for row in csv.reader(f) if row]
        return dir_list

    # 路径工具 
    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id], 'vis_imgs')

    def _get_event_img_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id], 'event_imgs')

    def _get_groundtruth_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    # 帧偏移量检测 
    def _detect_frame_offset(self, seq_path: str) -> int:
        """
        扫描 seq_path 目录 最小的帧编号作为 offset
        """
        import re
        min_id = None
        try:
            files = os.listdir(seq_path)
        except FileNotFoundError:
            return 0

        pattern = re.compile(r'^frame(\d+)\.(png|bmp)$', re.IGNORECASE)
        for fname in files:
            m = pattern.match(fname)
            if m:
                fid = int(m.group(1))
                if min_id is None or fid < min_id:
                    min_id = fid

        return min_id if min_id is not None else 0

    def _get_frame_offset(self, seq_id: int, seq_path: str) -> int:
        """offset 获取"""
        if seq_id not in self._frame_offset_cache:
            self._frame_offset_cache[seq_id] = self._detect_frame_offset(seq_path)
        return self._frame_offset_cache[seq_id]

    #  帧路径 

    def _get_frame_path(self, seq_path: str, frame_id: int, offset: int = 0) -> str:
        """
        Args:
            seq_path : 帧目录路径
            frame_id : groundtruth 行索引（从 0 开始）
            offset   : 文件名起始编号（由 _detect_frame_offset 获得）

        文件名编号 = frame_id + offset
        """
        real_id  = frame_id + offset
        png_path = os.path.join(seq_path, f'frame{real_id:04d}.png')
        bmp_path = os.path.join(seq_path, f'frame{real_id:04d}.bmp')

        if os.path.exists(png_path):
            return png_path
        if os.path.exists(bmp_path):
            return bmp_path

        raise FileNotFoundError(
            f"[VisEvent] Frame not found: {png_path} or {bmp_path}\n"
            f"  seq_path={seq_path}, frame_id={frame_id}, offset={offset}, "
            f"real_id={real_id}"
        )

    def _get_frame(self, seq_path: str, frame_id: int, offset: int = 0):
        path = self._get_frame_path(seq_path, frame_id, offset)
        return self.image_loader(path)

    #  标注读取 

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth.txt')
        gt = pandas.read_csv(
            bb_anno_file, delimiter=',', header=None,
            dtype=np.float32, na_filter=False, low_memory=False
        ).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox_path = self._get_groundtruth_path(seq_id)
        bbox      = self._read_bb_anno(bbox_path)
        valid     = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible   = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}


    def _suppress_event_background(self, evt_frame: np.ndarray) -> np.ndarray:
        """
        对白底红蓝点 Event Frame 做背景抑制
        """
        # min_channel: [H, W, 1]，每像素三通道最小值作为背景估计
        min_channel = evt_frame.min(axis=2, keepdims=True)         
        suppressed = (evt_frame.astype(np.int16) - min_channel.astype(np.int16))
        suppressed = suppressed.clip(0, 255).astype(np.uint8)
        return suppressed

    # 主接口 
    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        Returns:
            frame_list           : List[np.ndarray]  RGB 帧
            anno_frames          : Dict               逐帧标注
            object_meta          : OrderedDict
            frame_event_img_list : List[np.ndarray]  Event 可视化帧
        """
        # 路径 & 偏移量 
        seq_rgb_path       = self._get_sequence_path(seq_id)
        seq_event_img_path = self._get_event_img_sequence_path(seq_id)

        # 以 RGB 目录为准检测 offset
        offset = self._get_frame_offset(seq_id, seq_rgb_path)

        # RGB 帧 
        frame_list = [
            self._get_frame(seq_rgb_path, f_id, offset)
            for f_id in frame_ids
        ]

        # Event 帧 
        evt_cache_key = -(seq_id + 1)
        if evt_cache_key not in self._frame_offset_cache:
            self._frame_offset_cache[evt_cache_key] = \
                self._detect_frame_offset(seq_event_img_path)
        evt_offset = self._frame_offset_cache[evt_cache_key]
        frame_event_img_list = []
        for f_id in frame_ids:
            evt_frame = self._get_frame(seq_event_img_path, f_id, evt_offset)
            # 背景抑制
            # evt_frame = self._suppress_event_background(evt_frame)
            frame_event_img_list.append(evt_frame)

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
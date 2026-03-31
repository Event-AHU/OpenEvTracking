# lib/test/evaluation/fe108dataset.py
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import re

_IMG_EXTS = ('.png', '.bmp', '.jpg', '.jpeg')   # ← 新增 .jpg / .jpeg


def _num_key(fname):
    """从文件名中提取末尾数字用于排序，兼容 0001.jpg / frame_0001.png"""
    nums = re.findall(r'\d+', os.path.splitext(fname)[0])
    return int(nums[-1]) if nums else 0


def _list_imgs(dir_path):
    """返回 dir_path 下所有图像文件的完整路径列表（按数字升序）。"""
    if not os.path.isdir(dir_path):
        return []
    files = [f for f in os.listdir(dir_path)
             if f.lower().endswith(_IMG_EXTS)]
    return [os.path.join(dir_path, f)
            for f in sorted(files, key=_num_key)]


class FE108Dataset(BaseDataset):

    def __init__(self, split):
        super().__init__()
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.fe108_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.fe108_path, 'train')
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)

        #  GT 
        anno_path = os.path.join(seq_dir, 'groundtruth_rect.txt')
        ground_truth_rect = load_text(
            str(anno_path), delimiter=',', dtype=np.float64
        ).reshape(-1, 4)

        #  RGB 帧：img/ 优先，fallback stack/ aps/ 
        frames_list = []
        for rgb_dir_name in ('img', 'stack', 'aps'):
            frames_list = _list_imgs(os.path.join(seq_dir, rgb_dir_name))
            if frames_list:
                break
        if not frames_list:
            raise FileNotFoundError(
                f"[FE108Dataset] {sequence_name}: 找不到 RGB 帧\n"
                f"  已搜索: img/ stack/ aps/\n"
                f"  目录内容: {os.listdir(seq_dir)}"
            )

        #  Event 图：event_stack_center/ 优先，fallback dvs/ 
        event_img_list = []
        for evt_dir_name in ('event_stack_center', 'dvs', 'event_imgs'):
            event_img_list = _list_imgs(os.path.join(seq_dir, evt_dir_name))
            if event_img_list:
                break
        if not event_img_list:
            print(f"[FE108Dataset] WARNING: {sequence_name} 无 event 图目录，"
                  f"运行时将使用零图像替代")

        #  voxel (.mat)：可选 
        voxel_dir = os.path.join(seq_dir, 'voxel')
        if os.path.isdir(voxel_dir):
            frame_event_list = sorted(
                [os.path.join(voxel_dir, f)
                 for f in os.listdir(voxel_dir) if f.endswith('.mat')],
                key=lambda p: _num_key(os.path.basename(p))
            )
        else:
            frame_event_list = []

        #  帧数一致性对齐 
        n_rgb = len(frames_list)
        n_evt = len(event_img_list)
        n_gt  = ground_truth_rect.shape[0]

        if n_gt != n_rgb:
            min_len = min(n_gt, n_rgb)
            print(f"[FE108Dataset] {sequence_name}: "
                  f"gt={n_gt} vs rgb={n_rgb} → 截断到 {min_len}")
            frames_list       = frames_list[:min_len]
            ground_truth_rect = ground_truth_rect[:min_len]

        if n_evt > 0 and n_evt != len(frames_list):
            min_len = min(n_evt, len(frames_list))
            print(f"[FE108Dataset] {sequence_name}: "
                  f"event={n_evt} vs rgb={len(frames_list)} → 截断到 {min_len}")
            frames_list    = frames_list[:min_len]
            event_img_list = event_img_list[:min_len]

        assert len(frames_list) > 0, \
            f"[FE108Dataset] {sequence_name} frames_list 最终为空！"

        return Sequence(
            sequence_name,
            frames_list,
            'fe108',
            ground_truth_rect,
            frame_event_list=frame_event_list,
            event_img_list=event_img_list
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        list_file = os.path.join(self.base_path, 'list.txt')
        with open(list_file) as f:
            sequence_list = [l.strip() for l in f if l.strip()]

        if split in ('val', 'train'):
            spec = os.path.join(
                self.env_settings.dataspec_path, f'{split}.txt'
            )
            with open(spec) as f:
                seq_ids = f.read().splitlines()
            sequence_list = [sequence_list[int(x)] for x in seq_ids]

        return sequence_list
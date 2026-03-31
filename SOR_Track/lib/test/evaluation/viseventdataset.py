# lib/test/evaluation/viseventdataset.py

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VisEventDataset(BaseDataset):

    def __init__(self, split):
        super().__init__()
        self.split = split
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.visevent_path, 'test')
        elif split in ['val', 'train']:
            self.base_path = os.path.join(self.env_settings.visevent_path, 'train')
        else:
            raise ValueError(
                f"Unknown split '{split}'. Use 'test', 'val', or 'train'."
            )

        self.sequence_list = self._get_sequence_list(split)

    def get_sequence_list(self):
        return SequenceList(
            [self._construct_sequence(s) for s in self.sequence_list]
        )

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(
            self.base_path, sequence_name
        )
        ground_truth_rect = load_text(
            str(anno_path), delimiter=',', dtype=np.float64
        )

        # RGB frames 
        frames_path = '{}/{}/vis_imgs'.format(self.base_path, sequence_name)
        frame_list = sorted(
            [f for f in os.listdir(frames_path)
             if f.endswith(('.png', '.bmp'))],
            key=lambda f: int(f[-8:-4])
        )
        frames_list = [os.path.join(frames_path, f) for f in frame_list]

        # Event frames（图像格式）
        event_img_path = '{}/{}/event_imgs'.format(
            self.base_path, sequence_name
        )
        event_img_list_raw = sorted(
            [f for f in os.listdir(event_img_path)
             if f.endswith(('.png', '.bmp'))],
            key=lambda f: int(f[-8:-4])
        )
        event_img_list = [
            os.path.join(event_img_path, f) for f in event_img_list_raw
        ]

        # Voxel（.mat格式，base模式置空）
        voxel_path = '{}/{}/voxel'.format(self.base_path, sequence_name)
        if os.path.isdir(voxel_path):
            frame_event_list_raw = sorted(
                [f for f in os.listdir(voxel_path) if f.endswith('.mat')],
                key=lambda f: int(f[-8:-4])
            )
            frame_event_list = [
                os.path.join(voxel_path, f) for f in frame_event_list_raw
            ]
        else:
            frame_event_list = [None] * len(frames_list)

        # 帧数一致性检查 
        n_rgb = len(frames_list)
        n_evt = len(event_img_list)
        if n_rgb != n_evt:
            print(f"[VisEventDataset] WARNING: {sequence_name}: "
                  f"rgb={n_rgb} vs event={n_evt}, truncating to min")
            min_len          = min(n_rgb, n_evt)
            frames_list      = frames_list[:min_len]
            event_img_list   = event_img_list[:min_len]
            frame_event_list = frame_event_list[:min_len]

        return Sequence(
            sequence_name,
            frames_list,
            'visevent',
            ground_truth_rect.reshape(-1, 4),
            frame_event_list=frame_event_list,
            event_img_list=event_img_list
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        """
        读取序列列表
        """
        list_file = '{}/list.txt'.format(self.base_path)
        with open(list_file) as f:
            sequence_list = f.read().splitlines()

        # train/val：沿用原有 spec_file 逻辑，不过滤
        if split in ['val', 'train']:
            spec_file = '{}/{}.txt'.format(
                self.env_settings.dataspec_path, split
            )
            with open(spec_file) as f:
                seq_ids = f.read().splitlines()
            sequence_list = [sequence_list[int(x)] for x in seq_ids]
            return sequence_list

        valid_file_candidates = [
            os.path.join(self.base_path, 'valid_test_sequences.txt'),
        ]

        valid_file = None
        for candidate in valid_file_candidates:
            if os.path.isfile(candidate):
                valid_file = candidate
                break

        if valid_file is None:
            # fallback：使用全量序列，并提示
            print(
                f"[VisEventDataset] WARNING: valid_test_sequences.txt 未找到，"
                f"使用全量 {len(sequence_list)} 个测试序列。\n"
                f"  搜索路径：\n"
                + "\n".join(f"    {p}" for p in valid_file_candidates)
            )
            return sequence_list

        # 读取有效序列集合
        with open(valid_file) as f:
            valid_set = {
                line.strip() for line in f if line.strip()
            }

        filtered_list = [s for s in sequence_list if s in valid_set]

        not_in_list = valid_set - set(sequence_list)
        if not_in_list:
            print(
                f"[VisEventDataset] WARNING: "
                f"valid_test_sequences.txt 中有 {len(not_in_list)} 个序列"
                f"不在 list.txt 中（将被忽略）：\n"
                + ", ".join(sorted(not_in_list))
            )

        print(
            f"[VisEventDataset] 序列过滤完成：\n"
            f"  list.txt 全量：{len(sequence_list)} 个\n"
            f"  valid_file   ：{len(valid_set)} 个（{valid_file}）\n"
            f"  过滤后使用   ：{len(filtered_list)} 个"
        )

        return filtered_list
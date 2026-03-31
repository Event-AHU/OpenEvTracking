import torch
import numpy as np
from lib.utils.misc import NestedTensor


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)

class EventPreprocessor(object):
    def __init__(self,
                 mean: list = None,
                 std:  list = None,
                 mode: str = 'imagenet'):
        assert mode in ('imagenet', 'event'), \
            f"mode must be 'imagenet' or 'event', got '{mode}'"
        self.mode = mode
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STD  = [0.229, 0.224, 0.225]
        _EVENT_MEAN = [0.0346, 0.0000, 0.0549] 
        _EVENT_STD  = [0.1569, 0.0000006, 0.1832]
        if mode == 'imagenet':
            mean, std = _IMAGENET_MEAN, _IMAGENET_STD
        else:
            mean, std = _EVENT_MEAN, _EVENT_STD
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).cuda()
        self.std  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1).cuda()
        print(f"[EventPreprocessor] mode={mode}, "
              f"mean={mean}, std={std}")

    def _suppress_event_background(self, img_arr: np.ndarray) -> np.ndarray:
        """
        推理阶段背景抑制
        """
        min_ch = img_arr.min(axis=2, keepdims=True)
        suppressed = (img_arr.astype(np.int16) - min_ch.astype(np.int16))
        return suppressed.clip(0, 255).astype(np.uint8)
    
    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray) -> NestedTensor:
        """
        Args:
            img_arr   : (H, W, 3) uint8，原始 Event 帧
            amask_arr : (H, W)    bool，attention mask
        Returns:
            NestedTensor: tensors shape (1,3,H,W)，mask shape (1,H,W)
        """
        # 反色
        # img_processed = self._suppress_event_background(img_arr)
        img_processed = img_arr
        # img_inv = (255 - img_arr.astype(np.int16)).clip(0, 255).astype(np.uint8)
        # img_inv = img_arr.astype(np.int16).clip(0, 255).astype(np.uint8)
        # ToTensor + 归一化
        img_tensor = (torch.from_numpy(img_processed)
                      .cuda()
                      .float()
                      .permute(2, 0, 1)       # HWC → CHW
                      .unsqueeze(0))           # (1,3,H,W)
        img_tensor_norm = (img_tensor / 255.0 - self.mean) / self.std
        amask_tensor = (torch.from_numpy(amask_arr)
                        .to(torch.bool)
                        .cuda()
                        .unsqueeze(0))         # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)

class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)

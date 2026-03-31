"""
SOR 特征可视化器 - 推理阶段非侵入式 Hook 实现
"""

import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')          # 无头服务器环境，禁用 GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Section 1: Hook 管理器
class _FeatureCache:
    """线程安全的特征缓存（单 forward pass 生命周期）"""
    def __init__(self):
        self._store: OrderedDict = OrderedDict()

    def make_hook(self, key: str):
        def _hook(module, inp, output):
            feat = output[0] if isinstance(output, (tuple, list)) else output
            self._store[key] = feat.detach().float().cpu()
        return _hook

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def clear(self):
        self._store.clear()

    def keys(self):
        return list(self._store.keys())


class SORHookManager:
    """
    管理 SORFrontend 各子模块的 forward hook。

    注册目标：
      stem_rgb_out   → SPDStem(rgb) 输出
      stem_evt_out   → SPDStem(evt) 输出
      gabor_kernels  → GaborKernelGenerator 输出（核形状）
      R_rgb_normed   → gn_rgb 输出（方向响应，GroupNorm后）
      R_event_normed → gn_event 输出
      proj_out       → SORModule.proj 输出（残差相加前）
      f_spatial      → SORModule 整体输出
      f_reduced      → reduction 输出（若非 Identity）
    """

    def __init__(self, sor_frontend: nn.Module):
        self.frontend = sor_frontend
        self.cache    = _FeatureCache()
        self._handles = []

    def register(self):
        """注册所有 hook，返回 self 以支持链式调用"""
        assert len(self._handles) == 0, \
            "Hook already registered. Call remove() first."

        fe  = self.frontend
        sor = fe.sor

        _reg = self._reg   # 简写

        # SPDStem 输出
        _reg(fe.stem_rgb,       'stem_rgb_out')
        _reg(fe.stem_evt,       'stem_evt_out')

        # Gabor 核生成器
        _reg(sor.kernel_gen,    'gabor_kernels')

        # GroupNorm 后的多方向响应
        _reg(sor.gn_rgb,        'R_rgb_normed')
        _reg(sor.gn_event,      'R_event_normed')

        # 1×1 proj 输出（残差相加前）
        _reg(sor.proj,          'proj_out')

        # SORModule 整体输出
        _reg(sor,               'f_spatial')

        # Reduction（Identity 时跳过）
        if not isinstance(fe.reduction, nn.Identity):
            _reg(fe.reduction,  'f_reduced')

        return self

    def _reg(self, module: nn.Module, key: str):
        h = module.register_forward_hook(self.cache.make_hook(key))
        self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.cache.clear()

    def get_features(self) -> _FeatureCache:
        return self.cache

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *_):
        self.remove()


# 
# Section 2: 特征图 → 可视化图像的转换工具
# 

def _to_energy_map(feat: torch.Tensor,
                   target_hw: tuple,
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    [C, H, W] → 伪彩色热力图 (H_out, W_out, 3) uint8

    能量度量：对通道维度求 L2 范数
    归一化：min-max → [0, 255]
    上采样：双线性插值到 target_hw = (H_out, W_out)
    """
    energy = feat.norm(dim=0).numpy()              # [H, W]
    lo, hi = energy.min(), energy.max()
    if hi - lo > 1e-8:
        energy = (energy - lo) / (hi - lo)
    energy_u8 = (energy * 255).astype(np.uint8)

    H_out, W_out = target_hw
    energy_u8 = cv2.resize(energy_u8, (W_out, H_out),
                            interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(energy_u8, colormap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def _overlay(base_rgb: np.ndarray,
             heatmap: np.ndarray,
             alpha: float = 0.55) -> np.ndarray:
    """将热力图以 alpha 权重叠加到 RGB 图上，输出 uint8"""
    base  = base_rgb.astype(np.float32)
    heat  = heatmap.astype(np.float32)
    blend = (1 - alpha) * base + alpha * heat
    return np.clip(blend, 0, 255).astype(np.uint8)


def _denorm(tensor: torch.Tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    [1, 3, H, W] 归一化张量 → (H, W, 3) uint8 RGB numpy
    """
    t = tensor.squeeze(0).float().cpu()
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    t = (t * s + m).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _draw_bbox(img: np.ndarray,
               bbox,           # [x, y, w, h]
               color,
               thickness: int = 2) -> np.ndarray:
    """在 img 副本上画 bbox，返回新图"""
    out = img.copy()
    if bbox is None:
        return out
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


# 
# Section 3: 各子图的绘制函数
# 

def _plot_overview(rgb_vis: np.ndarray,
                   evt_vis: np.ndarray,
                   cache: _FeatureCache,
                   gt_bbox,
                   pred_bbox,
                   frame_id: int,
                   split: str) -> plt.Figure:
    """
    4格纵览图：
    [RGB输入] [Event输入] [SPDStem能量] [SOR输出能量]
    GT=绿框  Pred=红框
    """
    H, W = rgb_vis.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(
        f'SOR Feature Overview  |  Frame {frame_id:05d}  |  {split}',
        fontsize=13, fontweight='bold'
    )

    titles = ['RGB Input', 'Event Input',
              'SPDStem RGB Energy', 'SOR Output Energy']

    # 准备4张图
    imgs = [rgb_vis, evt_vis]

    stem_feat = cache.get('stem_rgb_out')
    if stem_feat is not None:
        heat = _to_energy_map(stem_feat[0], (H, W))
        imgs.append(_overlay(rgb_vis, heat))
    else:
        imgs.append(np.zeros_like(rgb_vis))

    sor_feat = cache.get('f_spatial')
    if sor_feat is not None:
        heat = _to_energy_map(sor_feat[0], (H, W))
        imgs.append(_overlay(rgb_vis, heat))
    else:
        imgs.append(np.zeros_like(rgb_vis))

    for ax, img, title in zip(axes, imgs, titles):
        # 在最后两格叠加 bbox
        show = img.copy()
        if title in ('SPDStem RGB Energy', 'SOR Output Energy'):
            show = _draw_bbox(show, gt_bbox,   (0, 200, 0))
            show = _draw_bbox(show, pred_bbox, (220, 50, 50))
        else:
            show = _draw_bbox(show, gt_bbox,   (0, 200, 0))
            show = _draw_bbox(show, pred_bbox, (220, 50, 50))

        ax.imshow(show)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # 图例
    legend = [
        mpatches.Patch(color=(0/255, 200/255, 0/255),   label='GT'),
        mpatches.Patch(color=(220/255, 50/255, 50/255), label='Pred'),
    ]
    fig.legend(handles=legend, loc='lower right',
               ncol=2, fontsize=9, framealpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def _plot_directional_response(cache: _FeatureCache,
                                rgb_vis: np.ndarray,
                                K: int,
                                gt_bbox,
                                frame_id: int,
                                split: str,
                                modal: str = 'rgb') -> plt.Figure:
    """
    K 方向 Gabor 响应分解图（RGB 或 Event 均可）
    key: 'R_rgb_normed' 或 'R_event_normed'
    """
    key  = f'R_{modal}_normed'
    feat = cache.get(key)
    H, W = rgb_vis.shape[:2]

    fig, axes = plt.subplots(1, K, figsize=(K * 4.5, 4.5))
    if K == 1:
        axes = [axes]

    fig.suptitle(
        f'Directional Gabor Response ({modal.upper()})  |  '
        f'Frame {frame_id:05d}  |  {split}',
        fontsize=11
    )

    for k in range(K):
        ax = axes[k]
        if feat is not None:
            C_total = feat.shape[1]
            C = C_total // K
            r_k = feat[0, k * C:(k + 1) * C]        # [C, H', W']
            heat = _to_energy_map(r_k, (H, W))
            ax.imshow(_overlay(rgb_vis, heat, alpha=0.6))
        else:
            ax.imshow(np.zeros_like(rgb_vis))

        ax.set_title(f'Dir {k}  (θ={k * 180 // K}°)', fontsize=9)
        if gt_bbox is not None:
            bx = gt_bbox.tolist() if hasattr(gt_bbox, 'tolist') else gt_bbox
            ax.add_patch(plt.Rectangle(
                (bx[0], bx[1]), bx[2], bx[3],
                linewidth=1.5, edgecolor='lime', facecolor='none'
            ))
        ax.axis('off')

    plt.tight_layout()
    return fig


def _plot_gate_mask(sor_module,
                    rgb_vis: np.ndarray,
                    K: int,
                    frame_id: int,
                    split: str) -> plt.Figure:
    """
    门控 mask 方向分解图：sigmoid(R_event)+1，∈(1,2)
    从 sor_module._last_mask 读取（需在 sor_module.py 中暴露）
    """
    H, W = rgb_vis.shape[:2]
    mask = getattr(sor_module, '_last_mask', None)

    fig, axes = plt.subplots(1, K, figsize=(K * 3.5, 3.5))
    if K == 1:
        axes = [axes]

    fig.suptitle(
        f'Gate Mask per Direction (∈[1,2])  |  '
        f'Frame {frame_id:05d}  |  {split}',
        fontsize=11
    )

    for k in range(K):
        ax = axes[k]
        if mask is not None:
            CK = mask.shape[1]
            C  = CK // K
            mk = mask[0, k * C:(k + 1) * C].mean(0).numpy()   # [H', W']
            mk_resized = cv2.resize(mk, (W, H), interpolation=cv2.INTER_LINEAR)
            im = ax.imshow(mk_resized, cmap='hot', vmin=1.0, vmax=2.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, '_last_mask not found\nAdd to sor_module.py',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='red')

        ax.set_title(f'Gate Dir {k}  (θ={k * 180 // K}°)', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    return fig


def _plot_gabor_kernels(cache: _FeatureCache,
                         frame_id: int) -> plt.Figure:
    """
    Gabor 卷积核形状可视化（仅首帧或每 N 帧调用一次即可）
    kernels: [B, K, 1, ks, ks]，取 batch=0
    """
    kernels = cache.get('gabor_kernels')

    if kernels is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.text(0.5, 0.5, 'gabor_kernels not cached',
                ha='center', va='center')
        return fig

    K  = kernels.shape[1]
    ks = kernels.shape[-1]
    k0 = kernels[0, :, 0].numpy()        # [K, ks, ks]

    fig, axes = plt.subplots(1, K, figsize=(K * 2.5, 2.8))
    if K == 1:
        axes = [axes]

    for i in range(K):
        k = k0[i]
        vmax = np.abs(k).max()
        axes[i].imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'θ={i * 180 // K}°', fontsize=9)
        axes[i].axis('off')

    fig.suptitle(f'Gabor Kernels  |  Frame {frame_id:05d}  '
                 f'(σ/λ/γ learnable)', fontsize=10)
    plt.tight_layout()
    return fig


# 
# Section 4: 主可视化器接口
# 

class SORVisualizer:
    """
    推理阶段 SOR 特征可视化器。

    典型使用：
        vis = SORVisualizer(model, cfg, save_dir)
        # 在 tracker.track() 的 torch.no_grad() 块后调用：
        if vis.should_visualize(frame_id):
            vis.run(frame_id, rgb_tensor, evt_tensor,
                    search_tensor, evt_search_tensor,
                    gt_bbox, pred_bbox,
                    split='search')
    """

    def __init__(self,
                 model:     nn.Module,
                 cfg,
                 save_dir:  str,
                 vis_every: int = None):
        """
        Args:
            model     : CEUTrack 实例（arch_mode='sor' 或 'stor'）
            cfg       : easydict config（读取 TEST.VIS_SOR / VIS_EVERY）
            save_dir  : 输出根目录（通常复用 debug_vis/...）
            vis_every : 每多少帧可视化一次，None 则从 cfg 读取
        """
        self.model    = model
        self.cfg      = cfg
        self.save_dir = Path(save_dir) / 'sor_vis'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 可视化开关
        self.enabled   = getattr(cfg.TEST, 'VIS_SOR', False)
        self.vis_every = (vis_every
                          or getattr(cfg.TEST, 'VIS_EVERY', 10))

        # 确认 arch_mode 支持 SOR 可视化
        arch = getattr(model, 'arch_mode', 'base')
        if self.enabled and arch not in ('sor', 'stor', 'sor_nostem'):
            print(f"[SORVisualizer] WARNING: arch_mode='{arch}' "
                  f"has no sor_frontend, disabling visualization.")
            self.enabled = False

        if self.enabled:
            self._frontend = model.sor_frontend
            self._sor_mod  = model.sor_frontend.sor
            self._K        = model.sor_frontend.sor_K
            print(f"[SORVisualizer] Enabled | "
                  f"every={self.vis_every} frames | "
                  f"save_dir={self.save_dir}")

    def should_visualize(self, frame_id: int) -> bool:
        return self.enabled and (frame_id % self.vis_every == 0)

    def run(self,
            frame_id:    int,
            rgb_tensor:  torch.Tensor,   # [1,3,H,W] 归一化，search区域
            evt_tensor:  torch.Tensor,   # [1,3,H,W] 归一化，search区域
            gt_bbox      = None,         # [x,y,w,h] 原始图像坐标
            pred_bbox    = None,         # [x,y,w,h]
            split:       str = 'search',
            img_mean:    tuple = (0.485, 0.456, 0.406),
            img_std:     tuple = (0.229, 0.224, 0.225),
            ):
        """
        执行单帧的完整可视化流程：
          1. 注册 Hook
          2. 在 SORFrontend 上再跑一次独立 forward（不影响主推理结果）
          3. 抓取特征，生成所有图表
          4. 移除 Hook，保存文件

        Args:
            rgb_tensor : 已经过 preprocessor 处理的搜索区 RGB  [1,3,H,W]
            evt_tensor : 已经过 event_preprocessor 处理的搜索区 Event [1,3,H,W]
            gt_bbox    : [x,y,w,h]，裁剪后坐标系（search patch内）
            pred_bbox  : [x,y,w,h]，裁剪后坐标系
        """
        # 反归一化得到 RGB 显示图
        rgb_vis = _denorm(rgb_tensor, img_mean, img_std)  # (H,W,3) uint8
        evt_vis = _denorm(evt_tensor,
                          getattr(self.cfg.DATA, 'EVENT_MEAN',
                                  (0.0586, 0.0865, 0.0461)),
                          getattr(self.cfg.DATA, 'EVENT_STD',
                                  (0.2108, 0.2623, 0.1727)))

        #  Hook 注册 + 独立前向 
        hook_mgr = SORHookManager(self._frontend)
        with hook_mgr:                        # __enter__ 注册，__exit__ 移除
            with torch.no_grad():
                phi = torch.zeros(1,
                                  device=rgb_tensor.device,
                                  dtype=rgb_tensor.dtype)
                _ = self._frontend(rgb_tensor, evt_tensor, phi)

            cache = hook_mgr.get_features()

            #  生成并保存各子图 

            # 图1：4格纵览
            fig1 = _plot_overview(
                rgb_vis, evt_vis, cache,
                gt_bbox, pred_bbox, frame_id, split
            )
            self._save(fig1, f'{frame_id:05d}_{split}_overview.png')

            # 图2：K方向 RGB 响应
            fig2 = _plot_directional_response(
                cache, rgb_vis, self._K,
                gt_bbox, frame_id, split, modal='rgb'
            )
            self._save(fig2, f'{frame_id:05d}_{split}_dir_rgb.png')

            # 图3：K方向 Event 响应
            fig3 = _plot_directional_response(
                cache, evt_vis, self._K,
                gt_bbox, frame_id, split, modal='event'
            )
            self._save(fig3, f'{frame_id:05d}_{split}_dir_event.png')

            # 图4：门控 mask
            fig4 = _plot_gate_mask(
                self._sor_mod, rgb_vis, self._K, frame_id, split
            )
            self._save(fig4, f'{frame_id:05d}_{split}_gate_mask.png')

            # 图5：Gabor 核（仅第1帧 + 每100帧更新一次，参数变化缓慢）
            if frame_id % 100 == 0 or frame_id <= 1:
                fig5 = _plot_gabor_kernels(cache, frame_id)
                self._save(fig5, f'{frame_id:05d}_gabor_kernels.png')

        # hook_mgr.__exit__ 已自动调用 remove()

    def _save(self, fig: plt.Figure, filename: str):
        out = self.save_dir / filename
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
from __future__ import annotations

import argparse
import math
import os
from abc import ABC
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from .eval import *
from .utils import *
from src.utils import safe_load_state_dict
from .ot_head_merge import merge_and_save_task_heads


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


KEEP_FROM_TARGET_MODALITY = frozenset({
    "ttokens",
    "pos_embed_ttoken",
    "pos_embed",
    "template_background_token",
    "template_foreground_token",
    "search_token",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "fc_norm.weight",
    "fc_norm.bias",
    "cls_token",
})

class TaskVectorABC(ABC):
    def __init__(self, pretrained_checkpoint: Optional[str] = None, 
                 finetuned_checkpoint: Optional[str] = None, 
                 vector: Optional[Dict[str, torch.Tensor]] = None, 
                 without_overlap: bool = False, 
                 pretrained_state_dict=None, 
                 ):

        self.without_overlap = without_overlap

        if vector is not None:
            self.vector = vector
            return

        if finetuned_checkpoint is None:
            raise ValueError("finetuned_checkpoint cannot be None when vector is not provided.")

        if pretrained_state_dict is None:
            if pretrained_checkpoint is None:
                raise ValueError("pretrained_checkpoint cannot be None when pretrained_state_dict is not provided.")
            pretrained_state_dict = load_and_adjust_pretrained(pretrained_checkpoint)

        finetuned_state_dict = safe_load_state_dict(finetuned_checkpoint)
        self.vector = {}

        with torch.no_grad():
            for key, pretrained_value in pretrained_state_dict.items():
                if key not in finetuned_state_dict:
                    print(f"Warning: key '{key}' is missing from finetuned checkpoint: {finetuned_checkpoint}")
                    continue

                if not (torch.is_floating_point(pretrained_value) or torch.is_complex(pretrained_value)):
                    continue

                finetuned_value = finetuned_state_dict[key]

                if pretrained_value.shape != finetuned_value.shape:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': pretrained={tuple(pretrained_value.shape)}, "
                        f"finetuned={tuple(finetuned_value.shape)}."
                    )

                self.vector[key] = finetuned_value.detach() - pretrained_value.detach()

    def __add__(self, other: TaskVectorABC):
        if not isinstance(other, TaskVectorABC):
            return NotImplemented

        new_vector = {}

        with torch.no_grad():
            for key, current_value in self.vector.items():
                if key not in other.vector:
                    print(f"Warning: key '{key}' is not present in both task vectors.")
                    continue

                new_vector[key] = current_value + other.vector[key]

                if self.without_overlap or other.without_overlap:
                    overlap_mask = (current_value != 0) & (other.vector[key] != 0)
                    if "attn" in key or "mlp" in key:
                        replacement = torch.full_like(new_vector[key], 1e-3)
                        new_vector[key] = torch.where(overlap_mask, replacement, new_vector[key])

        return TaskVectorABC(
            vector=new_vector,
            without_overlap=self.without_overlap or other.without_overlap,
        )

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, (int, float, torch.Tensor)):
            return NotImplemented

        with torch.no_grad():
            new_vector = {key: other * value for key, value in self.vector.items()}

        return TaskVectorABC(
            vector=new_vector,
            without_overlap=self.without_overlap,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        with torch.no_grad():
            new_vector = {key: -value for key, value in self.vector.items()}

        return TaskVectorABC(
            vector=new_vector,
            without_overlap=self.without_overlap,
        )

    def apply_to(self, pretrained_state_dict, scaling_coef: float = 1.0):
        new_state_dict = OrderedDict()

        with torch.no_grad():
            for key, pretrained_value in pretrained_state_dict.items():
                if key in self.vector:
                    new_state_dict[key] = pretrained_value.detach() + scaling_coef * self.vector[key]
                else:
                    new_state_dict[key] = pretrained_value.detach().clone()

        return new_state_dict

    def sum_two(self, other: TaskVectorABC):
        if not isinstance(other, TaskVectorABC):
            raise TypeError(f"other must be TaskVectorABC, but received {type(other)}")

        new_vector = {}

        with torch.no_grad():
            for key, current_value in self.vector.items():
                if key not in other.vector:
                    print(f"Warning: key '{key}' is not present in both task vectors.")
                    continue

                other_value = other.vector[key]
                overlap_mask = (current_value != 0) & (other_value != 0)
                summed_value = current_value + other_value
                new_vector[key] = torch.where(overlap_mask, current_value, summed_value)

        return TaskVectorABC(
            vector=new_vector,
            without_overlap=self.without_overlap or other.without_overlap,
        )

    def mul_according_to_mask(self, mask):
        new_vector = {}

        with torch.no_grad():
            for key, value in self.vector.items():
                if key not in mask:
                    print(f"Warning: key '{key}' is missing from mask, keeping the original task-vector value.")
                    new_vector[key] = value.detach().clone()
                    continue

                current_mask = mask[key].to(device=value.device, dtype=value.dtype)
                scaled_mask = torch.where(current_mask != 0, torch.full_like(current_mask, 3), current_mask)
                new_vector[key] = value * scaled_mask

        return TaskVectorABC(
            vector=new_vector,
            without_overlap=self.without_overlap,
        )

    def add_scale_according_to_mask(self, mask, scale=None):
        new_vector = {}

        if scale is None:
            scale = [0.25 for _ in range(len(self.vector))]

        if len(scale) != len(self.vector):
            raise ValueError(f"scale length must equal the number of task-vector parameters: {len(self.vector)}, received {len(scale)}")

        with torch.no_grad():
            for (key, value), current_scale in zip(self.vector.items(), scale):
                if key not in mask:
                    print(f"Warning: key '{key}' is missing from mask, keeping the original task-vector value.")
                    new_vector[key] = value.detach().clone()
                    continue

                current_mask = mask[key].to(device=value.device, dtype=value.dtype)
                scaled_mask = torch.where(current_mask != 0, torch.full_like(current_mask, current_scale), current_mask)
                new_vector[key] = value + value * scaled_mask

        return TaskVectorABC(
            vector=new_vector,
            without_overlap=self.without_overlap,
        )


class TaskVector(TaskVectorABC):
    def __init__(self, pretrained_checkpoint: Optional[str] = None, 
                 finetuned_checkpoint: Optional[str] = None, 
                 vector: Optional[Dict[str, torch.Tensor]] = None, 
                 without_overlap: bool = False, 
                 pretrained_state_dict=None):
        super().__init__(
            pretrained_checkpoint=pretrained_checkpoint,
            finetuned_checkpoint=finetuned_checkpoint,
            vector=vector,
            without_overlap=without_overlap,
            pretrained_state_dict=pretrained_state_dict,
        )


class TaskVectorMiddleKeep(TaskVectorABC):
    def __init__(self, pretrained_checkpoint=None, 
                 finetuned_checkpoint=None, vector=None, 
                 top_k_keep: float = 1.0, 
                 top_k_remove: float = 0.0, 
                 remove_first: bool = True, 
                 without_overlap: bool = False, 
                 pretrained_state_dict=None, 
                ):
        super().__init__(
            pretrained_checkpoint=pretrained_checkpoint,
            finetuned_checkpoint=finetuned_checkpoint,
            vector=vector,
            without_overlap=without_overlap,
            pretrained_state_dict=pretrained_state_dict,
        )

        if not 0.0 <= top_k_keep <= 1.0:
            raise ValueError(f"top_k_keep must be between 0 and 1, received {top_k_keep}")

        if not 0.0 <= top_k_remove <= 1.0:
            raise ValueError(f"top_k_remove must be between 0 and 1, received {top_k_remove}")

        self.top_k_keep = top_k_keep
        self.top_k_remove = top_k_remove

        with torch.no_grad():
            for key, value in self.vector.items():
                if remove_first:
                    self.vector[key] = self.mask_keep_top(self.mask_remove_top(value))
                else:
                    self.vector[key] = self.mask_remove_top(self.mask_keep_top(value))

    def mask_keep_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor

        top_k_int = int(tensor.shape[-1] * self.top_k_keep)

        if top_k_int <= 0:
            return torch.zeros_like(tensor)

        top_k_int = min(top_k_int, tensor.shape[-1])
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int, dim=-1)
        mask = torch.zeros_like(tensor)
        mask.scatter_(tensor.ndim - 1, masked_indices, 1)

        return mask * tensor

    def mask_remove_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor

        top_k_int = int(tensor.shape[-1] * self.top_k_remove)

        if top_k_int <= 0:
            return tensor

        top_k_int = min(top_k_int, tensor.shape[-1])
        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int, dim=-1)
        mask = torch.ones_like(tensor)
        mask.scatter_(tensor.ndim - 1, masked_indices, 0)

        return mask * tensor

class TaskVectorFFT(TaskVector):
    def __init__(
        self,
        finetuned_checkpoint=None,
        pretrained_state_dict=None,
        vector=None,
        args=None,
    ):
        super().__init__(
            finetuned_checkpoint=finetuned_checkpoint,
            pretrained_state_dict=pretrained_state_dict,
            vector=vector,
        )
        self.filter_ratio=args.special.filter_ratio
        self.find_low_high_fft()
    
    def find_low_high_fft(self):
        for key in self.vector.keys():
            if 'weight' in key and ('attn' in key or 'mlp' in key):
                if 'ffn_ln' in key:
                    continue
                if self.vector[key].ndim >= 2:
                    low_freq_spatial,high_freq_spatial,band_pass_spatial=self.cal_low_and_high_fft_for_tensor(self.vector[key])
                    self.vector[key]=band_pass_spatial+high_freq_spatial

    def cal_low_and_high_fft_for_tensor(self,weight):
        is_1x1_conv = (weight.ndim == 4 and weight.shape[-2:] == (1, 1))
        
        if is_1x1_conv:
            weight = weight[..., 0, 0]
        elif weight.ndim == 2:
            weight = weight
        else:
            raise ValueError(f"Unsupported FFT weight shape: {tuple(weight.shape)}")
        
        freq_domain = torch.fft.fft2(weight)
        H, W = freq_domain.shape
        low_radius = min(H, W) // 10  
        high_radius = min(H, W) * self.filter_ratio
        center = (H // 2, W // 2)
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist_from_center = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        high_freq_mask = (dist_from_center <= low_radius).float()
        low_freq_mask = (dist_from_center > high_radius).float()

        band_pass_mask = ((dist_from_center >= low_radius) & (dist_from_center <= high_radius)).float()
        low_freq_component = freq_domain * low_freq_mask
        high_freq_component = freq_domain * high_freq_mask
        band_pass_component = freq_domain * band_pass_mask

        low_freq_spatial = torch.fft.ifft2(low_freq_component).real
        high_freq_spatial = torch.fft.ifft2(high_freq_component).real
        band_pass_spatial = torch.fft.ifft2(band_pass_component).real
        
        if is_1x1_conv:
            low_freq_spatial = low_freq_spatial.unsqueeze(-1).unsqueeze(-1)
            high_freq_spatial = high_freq_spatial.unsqueeze(-1).unsqueeze(-1)
            band_pass_spatial = band_pass_spatial.unsqueeze(-1).unsqueeze(-1)
            
        return low_freq_spatial,high_freq_spatial,band_pass_spatial


def load_and_adjust_pretrained(pretrained_path: str):
    state_dict = torch.load(pretrained_path, map_location="cpu")

    state_dict = OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())
    
    for key in KEEP_FROM_TARGET_MODALITY:
        state_dict.pop(key, None)

    return state_dict


def load_plain_state_dict(checkpoint_path: str):
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint must contain a dictionary, but received {type(state_dict)}: {checkpoint_path}")

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        return state_dict["state_dict"]

    if "net" in state_dict and isinstance(state_dict["net"], dict):
        return state_dict["net"]

    return state_dict


def add_prefix_to_state_dict(source_state_dict, prefix: str):
    result = OrderedDict()

    for key, value in source_state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key

        if clean_key.startswith(prefix):
            result[clean_key] = value
        else:
            result[prefix + clean_key] = value

    return result


def pack_sparse_expert_override(
    specialized_state_dict,
    unified_state_dict,
    prefix="encoder.body.",
):
    packed_parameters = OrderedDict()
    
    with torch.no_grad():
        for key, specialized_tensor in specialized_state_dict.items():
            if key not in unified_state_dict:
                raise KeyError(
                    f"Key '{key}' is missing from unified state dict."
                )

            unified_tensor = unified_state_dict[key]

            if specialized_tensor.shape != unified_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': "
                    f"specialized={tuple(specialized_tensor.shape)}, "
                    f"unified={tuple(unified_tensor.shape)}"
                )

            if not (
                torch.is_floating_point(specialized_tensor)
                or torch.is_complex(specialized_tensor)
            ):
                continue

            specialized_cpu = (
                specialized_tensor.detach()
                .cpu()
                .contiguous()
            )
            unified_cpu = (
                unified_tensor.detach()
                .cpu()
                .contiguous()
            )

            specialized_flat = specialized_cpu.reshape(-1)
            unified_flat = unified_cpu.reshape(-1)

            changed_indices = torch.nonzero(
                specialized_flat != unified_flat,
                as_tuple=False,
            ).flatten()

            if changed_indices.numel() == 0:
                continue

            output_key = (
                key
                if key.startswith(prefix)
                else prefix + key
            )

            packed_parameters[output_key] = {
                "shape": tuple(specialized_cpu.shape),
                "indices": changed_indices.to(torch.int32),
                "values": specialized_flat[
                    changed_indices
                ].clone(),
            }
            
    return packed_parameters


parser = argparse.ArgumentParser(description="Merge SUTrack modality checkpoints")
parser.add_argument("--config_root_path", type=str, default=os.path.join(PROJECT_ROOT, "config"))
parser.add_argument("--model", type=str, default="ViT-B-16")
parser.add_argument("--task", type=str, default="8")
parser.add_argument("--method", type=str, default="FREE")
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--head_target_weight", type=float, default=0.7, help="Weight assigned to the final head.pth during head merging.") # 0.5
parser.add_argument("--head_ot_reg", type=float, default=0.05, help="Entropy regularization used by Sinkhorn OT.") # 0.05
parser.add_argument("--head_ot_iters", type=int, default=300, help="Maximum Sinkhorn iterations.") # 300
parser.add_argument("--head_identity_prior", type=float, default=0.02, help="Small diagonal prior for checkpoints from the same training trajectory.") # 0.02
parser.add_argument("--head_stats_weight", type=float, default=0.1, help="Contribution of bias and BatchNorm statistics to OT matching.") # 0.1
args = parser.parse_args()
args = read_config(args)


def merging(args):
    pretrained_checkpoint = os.path.join(PROJECT_ROOT, "pretrained", "backbone.pth")
    pretrained_state_dict = load_and_adjust_pretrained(pretrained_checkpoint)

    datasets = ["sutrack_b224_rgbt", "sutrack_b224_rgbe", "sutrack_b224_rgbd"]  # base

    shared_vectors = [
        TaskVectorFFT(
            finetuned_checkpoint=os.path.join(args.model_home, dataset, "backbone.pth"),
            pretrained_state_dict=pretrained_state_dict,
            args=args,
        )
        for dataset in datasets
    ]

    if args.special.with_align:
        shared_vectors = align_each_layer(shared_vectors)

    shared_task_vector_sum = sum(shared_vectors)

    if args.special.shared_only:
        backbone_state_dict = shared_task_vector_sum.apply_to(pretrained_state_dict, scaling_coef=args.special.scaling_coef)
    else:
        task_experts = [
            TaskVectorMiddleKeep(
                finetuned_checkpoint=os.path.join(args.model_home, dataset, "backbone.pth"),
                pretrained_state_dict=pretrained_state_dict,
                top_k_keep=args.special.top_k_keep,
                top_k_remove=args.special.top_k_remove,
            )
            for dataset in datasets
        ]

        rescaling = cal_rescaling(shared_vectors, task_experts, args.special.top_k_keep)
        task_experts_dict = {}

        for index, dataset in enumerate(datasets):
            task_experts_dict[dataset] = task_experts[index] * rescaling[index]

        # location_vector = task_experts_dict[target_dataset].sum_two(shared_task_vector_sum * args.special.scaling_coef)
        # backbone_state_dict = location_vector.apply_to(pretrained_state_dict, scaling_coef=args.special.scaling_coef)

    shared_vector_for_fusion = (shared_task_vector_sum * args.special.scaling_coef)
    unified_backbone_state_dict = shared_vector_for_fusion.apply_to(pretrained_state_dict, scaling_coef=args.special.scaling_coef)
            
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "unified_backbone")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "unified_backbone.pth")

    merged_state_dict = OrderedDict()
    merged_state_dict.update(add_prefix_to_state_dict(unified_backbone_state_dict, "encoder.body."))

    torch.save(
        {
            "net": merged_state_dict,
            "shared_only": False,
            "scaling_coef": args.special.scaling_coef,
            "datasets": datasets,
        },
        save_path,
    )

    print(f"Saved merged backbone to: {save_path}")
    
    expert_save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "experts")
    os.makedirs(expert_save_dir, exist_ok=True)
    
    for dataset in datasets:
        current_expert = task_experts_dict[dataset]

        location_vector = current_expert.sum_two(shared_vector_for_fusion)
        specialized_backbone_state_dict = location_vector.apply_to(pretrained_state_dict, scaling_coef=args.special.scaling_coef)

        # specialized is reconstructed by overriding selected unified parameters.
        packed_expert = pack_sparse_expert_override(
            specialized_state_dict=specialized_backbone_state_dict,
            unified_state_dict=unified_backbone_state_dict,
            prefix="encoder.body.",
        )
  
        expert_save_path = os.path.join(
            expert_save_dir,
            f"{dataset}_expert.pth",
        )

        torch.save(
            {
                "dataset": dataset,
                "scaling_coef": args.special.scaling_coef,
                "top_k_keep": args.special.top_k_keep,
                "top_k_remove": args.special.top_k_remove,
                "parameters": packed_expert,
            },
            expert_save_path,
        )
        
        print(f"Saved sparse expert to: {expert_save_path}")
    
    head_save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "merged_heads")

    merged_head_state_dicts = merge_and_save_task_heads(
        model_home=args.model_home,
        datasets=datasets,
        save_dir=head_save_dir,
        target_weight=args.head_target_weight,
        ot_reg=args.head_ot_reg,
        ot_max_iter=args.head_ot_iters,
        identity_prior=args.head_identity_prior,
        stats_weight=args.head_stats_weight,
    )

if __name__ == "__main__":
    merging(args)

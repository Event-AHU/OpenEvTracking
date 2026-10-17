from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


HEAD_EARLY_CHECKPOINT = {
    "sutrack_b224_rgbe": "head_24ep.pth",
    "sutrack_b224_rgbt": "head_24ep.pth",
    "sutrack_b224_rgbd": "head_24ep.pth",
}

CENTER_HEAD_BRANCHES = ("ctr", "offset", "size")


def _load_plain_state_dict(checkpoint_path: str) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must contain a dictionary, got {type(checkpoint)}: "
            f"{checkpoint_path}"
        )

    for wrapper_key in ("state_dict", "net", "model"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, dict):
            return wrapped

    return checkpoint


def _add_prefix_to_state_dict(
    source_state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> OrderedDict:
    result = OrderedDict()

    for key, value in source_state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        result[
            clean_key if clean_key.startswith(prefix) else prefix + clean_key
        ] = value

    return result


def extract_tracking_head_state_dict(checkpoint_path: str) -> OrderedDict:
    """
    Load a tracking-head checkpoint and normalize keys to decoder-local form,
    for example: ``conv1_ctr.0.weight``.

    Supported inputs:
      * a full-model checkpoint containing ``decoder.*`` keys;
      * a head-only checkpoint containing ``conv1_ctr.*`` keys;
      * checkpoints wrapped by ``net``, ``model`` or ``state_dict``.
    """
    raw_state_dict = _load_plain_state_dict(checkpoint_path)
    cleaned = OrderedDict()

    for key, value in raw_state_dict.items():
        if not torch.is_tensor(value):
            continue

        clean_key = key[7:] if key.startswith("module.") else key
        cleaned[clean_key] = value.detach().cpu().clone()

    prefix_candidates = (
        "decoder.",
        "head.",
        "box_head.",
        "net.decoder.",
        "model.decoder.",
    )
    prefix_counts = {
        prefix: sum(key.startswith(prefix) for key in cleaned)
        for prefix in prefix_candidates
    }
    selected_prefix, selected_count = max(
        prefix_counts.items(),
        key=lambda item: item[1],
    )

    if selected_count > 0:
        return OrderedDict(
            (key[len(selected_prefix):], value)
            for key, value in cleaned.items()
            if key.startswith(selected_prefix)
        )

    if any(
        key.startswith(("conv1_ctr", "conv1_offset", "conv1_size"))
        for key in cleaned
    ):
        return cleaned

    sample_keys = list(cleaned.keys())[:20]
    raise ValueError(
        f"Cannot find a SUTrack tracking decoder in {checkpoint_path}. "
        f"Sample keys: {sample_keys}"
    )


def _center_layer_layout(
    state_dict: Mapping[str, torch.Tensor],
    branch: str,
    layer_index: int,
) -> Tuple[str, str, bool]:
    base = f"conv{layer_index}_{branch}"
    sequential_weight_key = f"{base}.0.weight"
    direct_weight_key = f"{base}.weight"

    if sequential_weight_key in state_dict:
        return base, sequential_weight_key, True

    if direct_weight_key in state_dict:
        return base, direct_weight_key, False

    raise KeyError(
        f"Cannot find convolution weight for branch={branch}, "
        f"layer={layer_index}; expected '{sequential_weight_key}' "
        f"or '{direct_weight_key}'."
    )


def _output_channel_parameter_keys(
    state_dict: Mapping[str, torch.Tensor],
    base: str,
    is_sequential: bool,
) -> list[str]:
    if is_sequential:
        candidates = (
            f"{base}.0.bias",
            f"{base}.1.weight",
            f"{base}.1.bias",
            f"{base}.1.running_mean",
            f"{base}.1.running_var",
        )
    else:
        candidates = (f"{base}.bias",)

    return [key for key in candidates if key in state_dict]


def _transport_input_channels(
    weight: torch.Tensor,
    previous_transport: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Express source-layer input channels in the target model's channel basis.

    ``previous_transport`` shape:
        [source_input_channels, target_input_channels]
    """
    weight_fp32 = weight.detach().float()

    if previous_transport is None:
        return weight_fp32

    transport = previous_transport.to(weight_fp32)

    if weight_fp32.ndim == 2:
        return weight_fp32 @ transport

    if weight_fp32.ndim < 3:
        raise ValueError(f"Unsupported weight shape: {tuple(weight_fp32.shape)}")

    out_channels, in_channels = weight_fp32.shape[:2]
    if in_channels != transport.shape[0]:
        raise ValueError(
            f"Input-channel mismatch: weight has {in_channels}, "
            f"transport has {transport.shape[0]} source channels."
        )

    spatial_shape = weight_fp32.shape[2:]
    flattened = weight_fp32.reshape(out_channels, in_channels, -1)
    flattened = flattened.permute(2, 0, 1)
    aligned = torch.matmul(flattened, transport).permute(1, 2, 0)

    return aligned.reshape(
        out_channels,
        transport.shape[1],
        *spatial_shape,
    )


def _transport_output_channels(
    tensor: torch.Tensor,
    current_transport: torch.Tensor,
) -> torch.Tensor:
    """
    Barycentrically map source output channels to the target channel basis.

    ``current_transport`` shape:
        [source_output_channels, target_output_channels]
    """
    tensor_fp32 = tensor.detach().float()

    if tensor_fp32.ndim == 0:
        raise ValueError("A scalar tensor cannot be channel-transported.")

    if tensor_fp32.shape[0] != current_transport.shape[0]:
        raise ValueError(
            f"Output-channel mismatch: tensor has {tensor_fp32.shape[0]}, "
            f"transport has {current_transport.shape[0]} source channels."
        )

    flattened = tensor_fp32.reshape(tensor_fp32.shape[0], -1)
    aligned = current_transport.transpose(0, 1).to(flattened) @ flattened

    return aligned.reshape(
        current_transport.shape[1],
        *tensor_fp32.shape[1:],
    )


def _build_neuron_descriptor(
    aligned_weight: torch.Tensor,
    original_state_dict: Mapping[str, torch.Tensor],
    base: str,
    is_sequential: bool,
    stats_weight: float,
) -> torch.Tensor:
    """
    Describe each output neuron mainly by incoming weights, augmented by
    convolution bias and BatchNorm statistics.
    """
    descriptor = aligned_weight.reshape(aligned_weight.shape[0], -1).float()
    descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-12)

    statistics = []
    for key in _output_channel_parameter_keys(
        original_state_dict,
        base,
        is_sequential,
    ):
        value = original_state_dict[key]

        if (
            not torch.is_floating_point(value)
            or value.ndim != 1
            or value.shape[0] != aligned_weight.shape[0]
        ):
            continue

        current = value.detach().float()

        if key.endswith("running_var"):
            current = torch.log(current.clamp_min(1e-8))

        current = current - current.mean()
        current = current / current.std(unbiased=False).clamp_min(1e-6)
        statistics.append(current.unsqueeze(1))

    if statistics:
        descriptor = torch.cat(
            (
                descriptor,
                stats_weight * torch.cat(statistics, dim=1),
            ),
            dim=1,
        )

    return F.normalize(descriptor, p=2, dim=1, eps=1e-12)


def _sinkhorn_transport(
    source_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    reg: float = 0.05,
    max_iter: int = 300,
    tol: float = 1e-6,
    identity_prior: float = 0.02,
) -> torch.Tensor:
    """
    Compute entropy-regularized optimal transport in log space.

    The returned matrix is column-normalized so that each target neuron is a
    barycentric combination of source neurons.
    """
    if reg <= 0:
        raise ValueError(f"reg must be positive, got {reg}")

    source_descriptor = F.normalize(
        source_descriptor.detach().float(),
        p=2,
        dim=1,
        eps=1e-12,
    )
    target_descriptor = F.normalize(
        target_descriptor.detach().float(),
        p=2,
        dim=1,
        eps=1e-12,
    )

    cost = torch.cdist(
        source_descriptor,
        target_descriptor,
        p=2,
    ).square()

    positive_cost = cost[cost > 0]
    if positive_cost.numel() > 0:
        cost = cost / positive_cost.median().clamp_min(1e-12)

    # The checkpoints are from the same task and usually the same training
    # trajectory. A small diagonal prior discourages gratuitous permutations,
    # while still allowing evidence-driven channel movement.
    if identity_prior > 0 and cost.shape[0] == cost.shape[1]:
        identity = torch.eye(
            cost.shape[0],
            device=cost.device,
            dtype=cost.dtype,
        )
        cost = cost + identity_prior * (1.0 - identity)

    source_count, target_count = cost.shape
    log_source_mass = torch.full(
        (source_count,),
        -math.log(source_count),
        device=cost.device,
        dtype=cost.dtype,
    )
    log_target_mass = torch.full(
        (target_count,),
        -math.log(target_count),
        device=cost.device,
        dtype=cost.dtype,
    )

    log_kernel = -cost / reg
    log_u = torch.zeros_like(log_source_mass)
    log_v = torch.zeros_like(log_target_mass)

    for _ in range(max_iter):
        previous_log_v = log_v

        log_u = log_source_mass - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0),
            dim=1,
        )
        log_v = log_target_mass - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1),
            dim=0,
        )

        if torch.max(torch.abs(log_v - previous_log_v)).item() < tol:
            break

    transport = torch.exp(
        log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
    )

    if not torch.isfinite(transport).all():
        raise FloatingPointError(
            "Sinkhorn produced NaN/Inf. Increase ot_reg, for example "
            "from 0.05 to 0.1."
        )

    # Barycentric correction used for neuron alignment:
    # each target column sums to one.
    transport = transport / transport.sum(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-12)

    return transport


def _weighted_merge(
    target_tensor: torch.Tensor,
    aligned_source_tensor: torch.Tensor,
    target_weight: float,
) -> torch.Tensor:
    if target_tensor.shape != aligned_source_tensor.shape:
        raise ValueError(
            f"Cannot merge shapes {tuple(target_tensor.shape)} and "
            f"{tuple(aligned_source_tensor.shape)}."
        )

    merged = (
        target_weight * target_tensor.detach().float()
        + (1.0 - target_weight) * aligned_source_tensor.detach().float()
    )
    return merged.to(dtype=target_tensor.dtype)


def merge_sutrack_center_heads_ot(
    source_head: Mapping[str, torch.Tensor],
    target_head: Mapping[str, torch.Tensor],
    target_weight: float = 0.5,
    ot_reg: float = 0.05,
    ot_max_iter: int = 300,
    ot_tol: float = 1e-6,
    identity_prior: float = 0.02,
    stats_weight: float = 0.1,
) -> OrderedDict:
    """
    Align ``source_head`` to ``target_head`` branch by branch, then merge.

    Recommended usage:
      * source_head: head_24ep.pth or head_48ep.pth
      * target_head: head.pth

    The target checkpoint defines the final channel coordinate system.
    """

    merged = OrderedDict(
        (key, value.detach().clone())
        for key, value in target_head.items()
    )
    processed_keys: set[str] = set()
    found_branch = False

    with torch.no_grad():
        for branch in CENTER_HEAD_BRANCHES:
            first_layer_candidates = (
                f"conv1_{branch}.0.weight",
                f"conv1_{branch}.weight",
            )
            if not any(key in target_head for key in first_layer_candidates):
                continue

            found_branch = True
            previous_transport = None

            for layer_index in range(1, 6):
                base, target_weight_key, is_sequential = _center_layer_layout(
                    target_head,
                    branch,
                    layer_index,
                )
                source_base, source_weight_key, source_is_sequential = (
                    _center_layer_layout(
                        source_head,
                        branch,
                        layer_index,
                    )
                )

                if base != source_base or is_sequential != source_is_sequential:
                    raise ValueError(
                        f"Head structures differ at {base}: "
                        f"source sequential={source_is_sequential}, "
                        f"target sequential={is_sequential}."
                    )

                source_layer_weight = source_head[source_weight_key]
                target_layer_weight = target_head[target_weight_key]

                if source_layer_weight.shape != target_layer_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {base}: "
                        f"source={tuple(source_layer_weight.shape)}, "
                        f"target={tuple(target_layer_weight.shape)}."
                    )

                source_input_aligned = _transport_input_channels(
                    source_layer_weight,
                    previous_transport,
                )

                if layer_index < 5:
                    source_descriptor = _build_neuron_descriptor(
                        source_input_aligned,
                        source_head,
                        source_base,
                        source_is_sequential,
                        stats_weight,
                    )
                    target_descriptor = _build_neuron_descriptor(
                        target_layer_weight,
                        target_head,
                        base,
                        is_sequential,
                        stats_weight,
                    )

                    current_transport = _sinkhorn_transport(
                        source_descriptor,
                        target_descriptor,
                        reg=ot_reg,
                        max_iter=ot_max_iter,
                        tol=ot_tol,
                        identity_prior=identity_prior,
                    )

                    aligned_source_weight = _transport_output_channels(
                        source_input_aligned,
                        current_transport,
                    )
                    merged[target_weight_key] = _weighted_merge(
                        target_layer_weight,
                        aligned_source_weight,
                        target_weight,
                    )
                    processed_keys.add(target_weight_key)

                    target_output_keys = _output_channel_parameter_keys(
                        target_head,
                        base,
                        is_sequential,
                    )
                    source_output_keys = _output_channel_parameter_keys(
                        source_head,
                        source_base,
                        source_is_sequential,
                    )

                    if set(target_output_keys) != set(source_output_keys):
                        raise KeyError(
                            f"Output parameter keys differ for {base}: "
                            f"source={source_output_keys}, "
                            f"target={target_output_keys}."
                        )

                    for parameter_key in target_output_keys:
                        source_parameter = source_head[parameter_key]
                        target_parameter = target_head[parameter_key]

                        if source_parameter.shape != target_parameter.shape:
                            raise ValueError(
                                f"Shape mismatch for {parameter_key}: "
                                f"source={tuple(source_parameter.shape)}, "
                                f"target={tuple(target_parameter.shape)}."
                            )

                        aligned_source_parameter = _transport_output_channels(
                            source_parameter,
                            current_transport,
                        )
                        merged[parameter_key] = _weighted_merge(
                            target_parameter,
                            aligned_source_parameter,
                            target_weight,
                        )
                        processed_keys.add(parameter_key)

                    previous_transport = current_transport

                else:
                    # Final channels have fixed semantics:
                    # ctr=objectness, offset=(x,y), size=(w,h).
                    # Only align their inputs; never permute their outputs.
                    merged[target_weight_key] = _weighted_merge(
                        target_layer_weight,
                        source_input_aligned,
                        target_weight,
                    )
                    processed_keys.add(target_weight_key)

                    target_bias_key = f"{base}.bias"
                    source_bias_key = f"{source_base}.bias"

                    if target_bias_key in target_head:
                        if source_bias_key not in source_head:
                            raise KeyError(
                                f"Missing source bias: {source_bias_key}"
                            )

                        merged[target_bias_key] = _weighted_merge(
                            target_head[target_bias_key],
                            source_head[source_bias_key],
                            target_weight,
                        )
                        processed_keys.add(target_bias_key)

        if not found_branch:
            raise ValueError(
                "No CENTER-head branch was found. Expected parameters such as "
                "conv1_ctr, conv1_offset and conv1_size."
            )

        # Preserve integer buffers such as num_batches_tracked from head.pth.
        # Merge any extra floating-point tensors not covered above directly.
        for key, target_value in target_head.items():
            if key in processed_keys or key not in source_head:
                continue

            source_value = source_head[key]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"Unhandled shape mismatch for {key}: "
                    f"source={tuple(source_value.shape)}, "
                    f"target={tuple(target_value.shape)}."
                )

            if torch.is_floating_point(target_value):
                merged[key] = _weighted_merge(
                    target_value,
                    source_value,
                    target_weight,
                )

    return merged


def merge_and_save_task_heads(
    model_home: str,
    datasets: Iterable[str],
    save_dir: str,
    *,
    target_weight: float = 0.5,
    ot_reg: float = 0.05,
    ot_max_iter: int = 300,
    ot_tol: float = 1e-6,
    identity_prior: float = 0.02,
    stats_weight: float = 0.1,
) -> Dict[str, OrderedDict]:
    """
    Merge and save one task-specific head per dataset.
    
    """
    os.makedirs(save_dir, exist_ok=True)
    merged_heads: Dict[str, OrderedDict] = {}

    for dataset in datasets:
        if dataset not in HEAD_EARLY_CHECKPOINT:
            raise KeyError(
                f"No early head checkpoint rule for dataset '{dataset}'."
            )

        early_checkpoint_name = HEAD_EARLY_CHECKPOINT[dataset]
        source_path = os.path.join(model_home, dataset, early_checkpoint_name)
        target_path = os.path.join(model_home, dataset, "head.pth")

        source_head = extract_tracking_head_state_dict(source_path)
        target_head = extract_tracking_head_state_dict(target_path)

        merged_head = merge_sutrack_center_heads_ot(
            source_head=source_head,
            target_head=target_head,
            target_weight=target_weight,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_tol=ot_tol,
            identity_prior=identity_prior,
            stats_weight=stats_weight,
        )

        # In a full SUTrack model, the tracking decoder is named "decoder".
        full_model_head = _add_prefix_to_state_dict(
            merged_head,
            "decoder.",
        )
        merged_heads[dataset] = full_model_head

        output_path = os.path.join(
            save_dir,
            f"{dataset}_head_ot.pth",
        )
        torch.save(
            {
                "net": full_model_head,
                "dataset": dataset,
            },
            output_path,
        )

        print(f"Saved OT-merged head to: {output_path}")

    return merged_heads

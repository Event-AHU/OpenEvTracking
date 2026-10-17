import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_weight_file(weight_path):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    print(f"Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported weight type in {weight_path}: {type(checkpoint)}")

    for field in ("net", "model", "state_dict", "module"):
        if field in checkpoint and isinstance(checkpoint[field], Mapping):
            print(f"  Using field: {field}")
            checkpoint = checkpoint[field]
            break

    state_dict = OrderedDict()

    for key, value in checkpoint.items():
        if not isinstance(key, str):
            print(f"  [Warning] Skip non-string key: {key}")
            continue

        if not torch.is_tensor(value):
            print(f"  [Warning] Skip non-tensor item: {key}")
            continue

        state_dict[key] = value

    return state_dict


def load_sparse_expert_override(expert_path):
    """
    Load the sparse expert override checkpoint.

    """
    if not os.path.isfile(expert_path):
        raise FileNotFoundError(f"Expert file not found: {expert_path}")

    print(f"Loading sparse expert from: {expert_path}")

    checkpoint = torch.load(expert_path, map_location="cpu")

    dataset = checkpoint.get("dataset", "unknown")

    print(f"  Expert dataset: {dataset}")

    return checkpoint


def remove_prefix(key, prefix):
    """
    Remove the specified prefix to avoid adding it twice.
    
    """
    if key.startswith(prefix):
        return key[len(prefix):]

    return key


def build_normalized_key_map(state_dict, prefix="encoder.body."):
    """
    Map backbone keys to forms without the encoder.body. prefix.
    
    """
    normalized_to_actual = {}

    for actual_key in state_dict:
        normalized_key = remove_prefix(actual_key, prefix)

        if normalized_key in normalized_to_actual:
            previous_key = normalized_to_actual[normalized_key]

            raise KeyError(
                "Duplicate normalized backbone key detected:\n"
                f"  normalized key: {normalized_key}\n"
                f"  first key: {previous_key}\n"
                f"  second key: {actual_key}"
            )

        normalized_to_actual[normalized_key] = actual_key

    return normalized_to_actual


def compose_unified_and_sparse_override(
    unified_state_dict,
    expert_checkpoint,
    prefix="encoder.body.",
    verify_unique_indices=True,
):
    """Apply the sparse expert override to the unified backbone."""
    if not isinstance(unified_state_dict, Mapping):
        raise TypeError("unified_state_dict must be a mapping.")

    if not isinstance(expert_checkpoint, Mapping):
        raise TypeError("expert_checkpoint must be a mapping.")

    parameters = expert_checkpoint.get("parameters")

    composed_state_dict = OrderedDict((key, value.detach().clone()) for key, value in unified_state_dict.items())

    normalized_key_map = build_normalized_key_map(composed_state_dict, prefix=prefix)

    with torch.no_grad():
        for expert_key, sparse_item in parameters.items():
            if not isinstance(expert_key, str):
                raise TypeError(f"Expert key must be a string, received {type(expert_key)}")

            if not isinstance(sparse_item, Mapping):
                raise TypeError(f"Sparse item for '{expert_key}' must be a mapping.")

            normalized_expert_key = remove_prefix(expert_key, prefix)

            if normalized_expert_key not in normalized_key_map:
                raise KeyError(
                    f"Expert parameter '{expert_key}' does not exist in unified backbone.\n"
                    f"Normalized key: {normalized_expert_key}"
                )

            backbone_key = normalized_key_map[normalized_expert_key]
            base_tensor = composed_state_dict[backbone_key]

            if not torch.is_tensor(base_tensor):
                raise TypeError(f"Backbone value for '{backbone_key}' is not a tensor.")

            for required_field in ("shape", "indices", "values"):
                if required_field not in sparse_item:
                    raise KeyError(f"Sparse expert item '{expert_key}' is missing '{required_field}'.")

            expected_shape = tuple(sparse_item["shape"])

            if tuple(base_tensor.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for '{expert_key}':\n"
                    f"  backbone: {tuple(base_tensor.shape)}\n"
                    f"  expert: {expected_shape}"
                )

            indices = sparse_item["indices"]
            values = sparse_item["values"]

            if not torch.is_tensor(indices):
                raise TypeError(f"Expert indices for '{expert_key}' must be a tensor.")

            if not torch.is_tensor(values):
                raise TypeError(f"Expert values for '{expert_key}' must be a tensor.")

            indices = indices.reshape(-1).to(dtype=torch.long)
            values = values.reshape(-1)

            if indices.numel() != values.numel():
                raise ValueError(
                    f"Indices/value count mismatch for '{expert_key}': "
                    f"indices={indices.numel()}, values={values.numel()}"
                )

            if indices.numel() == 0:
                continue

            min_index = indices.min().item()
            max_index = indices.max().item()

            if min_index < 0:
                raise IndexError(f"Negative expert index for '{expert_key}': {min_index}")

            if max_index >= base_tensor.numel():
                raise IndexError(
                    f"Expert index out of range for '{expert_key}': "
                    f"max_index={max_index}, numel={base_tensor.numel()}"
                )

            if verify_unique_indices:
                unique_count = torch.unique(indices).numel()

                if unique_count != indices.numel():
                    raise ValueError(
                        f"Duplicate indices detected for '{expert_key}': "
                        f"indices={indices.numel()}, unique={unique_count}"
                    )

            if values.dtype != base_tensor.dtype:
                raise TypeError(
                    f"Dtype mismatch for '{expert_key}':\n"
                    f"  backbone dtype: {base_tensor.dtype}\n"
                    f"  expert dtype: {values.dtype}\n"
                    "For exact reconstruction, the expert values must use the same dtype as the unified backbone."
                )

            patched_tensor = base_tensor.detach().clone().contiguous()
            flat_tensor = patched_tensor.view(-1)
            flat_tensor.index_copy_(0, indices.to(device=flat_tensor.device), values.to(device=flat_tensor.device))
            composed_state_dict[backbone_key] = patched_tensor

    return composed_state_dict


def add_parameter(
    merged_state_dict,
    source_key,
    value,
    prefix,
    source_name,
):
    """
    Add a parameter to the complete state_dict and check for key conflicts.
    """
    clean_key = remove_prefix(source_key, prefix)
    merged_key = prefix + clean_key

    if merged_key in merged_state_dict:
        raise KeyError(
            f"Duplicate key detected: {merged_key}\n"
            f"Source: {source_name}\n"
            f"Original key: {source_key}"
        )

    merged_state_dict[merged_key] = value


def merge_sutrack_weights(
    backbone_path,
    head_path,
    task_path,
    specific_path,
    output_path,
    expert_path: Optional[str] = None,
    checkpoint_field="net",
):

    # ---------------------------------------------------------
    # 1. Load the unified backbone
    # ---------------------------------------------------------
    backbone_state_dict = load_weight_file(backbone_path)

    # ---------------------------------------------------------
    # 2. Optionally apply the sparse expert override
    # ---------------------------------------------------------
    if expert_path is not None:
        expert_checkpoint = load_sparse_expert_override(expert_path)

        backbone_state_dict = compose_unified_and_sparse_override(
            unified_state_dict=backbone_state_dict,
            expert_checkpoint=expert_checkpoint,
            prefix="encoder.body.",
        )

    else:
        print("No expert selected. Using pure unified backbone.")

    # ---------------------------------------------------------
    # 3. Load the remaining model components
    # ---------------------------------------------------------
    head_state_dict = load_weight_file(head_path)
    task_state_dict = load_weight_file(task_path)
    specific_state_dict = load_weight_file(specific_path)

    merged_state_dict = OrderedDict()

    # ---------------------------------------------------------
    # 4. Backbone
    # ---------------------------------------------------------
    for key, value in backbone_state_dict.items():
        add_parameter(
            merged_state_dict=merged_state_dict,
            source_key=key,
            value=value,
            prefix="encoder.body.",
            source_name="backbone",
        )

    # ---------------------------------------------------------
    # 5. Specific
    # ---------------------------------------------------------
    for key, value in specific_state_dict.items():
        add_parameter(
            merged_state_dict=merged_state_dict,
            source_key=key,
            value=value,
            prefix="encoder.body.",
            source_name="specific",
        )

    # ---------------------------------------------------------
    # 6. Decoder / Head
    # ---------------------------------------------------------
    for key, value in head_state_dict.items():
        add_parameter(
            merged_state_dict=merged_state_dict,
            source_key=key,
            value=value,
            prefix="decoder.",
            source_name="head",
        )

    # ---------------------------------------------------------
    # 7. Task decoder
    # ---------------------------------------------------------
    for key, value in task_state_dict.items():
        add_parameter(
            merged_state_dict=merged_state_dict,
            source_key=key,
            value=value,
            prefix="task_decoder.",
            source_name="task",
        )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    output_checkpoint = {checkpoint_field: merged_state_dict}

    torch.save(output_checkpoint, output_path)
    
    print("Save checkpoint to:", output_path)


if __name__ == "__main__":
    backbone_weight_dir = os.path.join(PROJECT_ROOT, "weight_root")
    other_weight_dir = "stage2/checkpoints/train/sutrack/sutrack_b224_rgbe"

    backbone_path = os.path.join(backbone_weight_dir, "unified_backbone/unified_backbone.pth")
    head_path = os.path.join(backbone_weight_dir, "merged_heads/sutrack_b224_rgbe_head_ot.pth")
    task_path = os.path.join(other_weight_dir, "task.pth")
    specific_path = os.path.join(other_weight_dir, "specific.pth")

    expert_path = os.path.join(backbone_weight_dir, "experts", "sutrack_b224_rgbe_expert.pth")

    output_path = os.path.join(PROJECT_ROOT, "pretrained/merged_rgbe_ckp.pth.tar")

    merge_sutrack_weights(
        backbone_path=backbone_path,
        head_path=head_path,
        task_path=task_path,
        specific_path=specific_path,
        expert_path=expert_path,
        output_path=output_path,
        checkpoint_field="net",
    )

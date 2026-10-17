import os
import torch
from collections import OrderedDict


def extract_backbone_and_head(
    ckpt_path,
    save_dir=None,
    backbone_save_name="backbone.pth",
    head_save_name="head.pth",
    task_save_name="task.pth",
    specific_save_name="specific.pth",
    
):
    if save_dir is None:
        save_dir = os.path.dirname(ckpt_path)

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if isinstance(ckpt, dict):
        if "net" in ckpt:
            state_dict = ckpt["net"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "module" in ckpt:
            state_dict = ckpt["module"]
        else:
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    backbone_state_dict = OrderedDict()
    head_state_dict = OrderedDict()
    task_state_dict = OrderedDict() 
    specific_state_dict = OrderedDict() 
    
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("blocks."):
            backbone_state_dict[clean_k] = v
        
        elif clean_k.startswith("encoder.body."):
            new_k = clean_k[len("encoder.body."):] 
            if 'block' not in clean_k or 'cls_token' in clean_k:
                specific_state_dict[new_k] = v
            else:
                backbone_state_dict[new_k] = v
                
        if clean_k.startswith("decoder."):
            new_k = clean_k[len("decoder."):]
            head_state_dict[new_k] = v

        if clean_k.startswith("task_decoder."):
            new_k = clean_k[len("task_decoder."):]
            task_state_dict[new_k] = v

    backbone_save_path = os.path.join(save_dir, backbone_save_name)
    head_save_path = os.path.join(save_dir, head_save_name)
    task_save_path = os.path.join(save_dir, task_save_name)
    specific_save_path = os.path.join(save_dir, specific_save_name)
    

    torch.save(backbone_state_dict, backbone_save_path)
    torch.save(head_state_dict, head_save_path)
    torch.save(task_state_dict, task_save_path)
    torch.save(specific_state_dict, specific_save_path)
    

    print(f"Backbone params: {len(backbone_state_dict)}")
    print(f"Head params: {len(head_state_dict)}")
    print(f"Task params: {len(task_state_dict)}")
    print(f"Specific params: {len(specific_state_dict)}")

    print(f"Saved backbone to: {backbone_save_path}")
    print(f"Saved head to: {head_save_path}")
    print(f"Saved task to: {task_save_path}")
    print(f"Saved specific to: {specific_save_path}")
    

    if len(backbone_state_dict) == 0:
        print("[Warning] No backbone parameters found. Please check key names in checkpoint.")

    if len(head_state_dict) == 0:
        print("[Warning] No head parameters found. Please check key names in checkpoint.")

    if len(task_state_dict) == 0:
        print("[Warning] No task parameters found. Please check key names in checkpoint.")

    if len(specific_state_dict) == 0:
        print("[Warning] No specific parameters found. Please check key names in checkpoint.")
               
    return backbone_state_dict, head_state_dict, task_state_dict, specific_state_dict


if __name__ == "__main__":
    ckpt_path = ".../train/sutrack/sutrack_b224_rgbe/SUTRACK_ep0030.pth.tar"
    save_dir = ".../train/sutrack/sutrack_b224_rgbe"

    extract_backbone_and_head(
        ckpt_path=ckpt_path,
        save_dir=save_dir,
        backbone_save_name="backbone.pth",
        head_save_name="head.pth",
        task_save_name="task.pth",
        specific_save_name="specific.pth",
    )

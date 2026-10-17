# UniMergeTrack

UniMergeTrack is a three-stage pipeline for unified **RGB-T, RGB-E, and RGB-D tracking**: train task-specific trackers, merge model parameters, and align each task with a frozen unified backbone.

<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/OpenEvTracking/blob/main/UniMergeTrack/figures/framework.jpg" />
</p>


## Stage 1: Train Task-Specific Trackers

Train three trackers independently:

```bash
cd stage1/RGBT && bash train_rgbt.sh
cd ../RGBE && bash train_rgbe.sh
cd ../RGBD && bash train_rgbd.sh
```

Save the **epoch-24** and **epoch-30** checkpoints for all three tasks.

## Stage 2: Extract and Merge

For each task (`rgbt`, `rgbe`, `rgbd`):

1. Edit `ckpt_path` and `save_dir` in `stage2/extract_params.py`, then run `cd stage2 && python extract_params.py` on the **epoch-30** checkpoint to obtain `backbone.pth`, `head.pth`, `task.pth`, and `specific.pth`.
2. Extract the **epoch-24** tracking head as `head_24ep.pth`.
3. Store all five files under `stage2/checkpoints/train/sutrack/sutrack_b224_<task>/`.

Also extract the pretrained backbone from `RELO_ep0090.pth.tar` to `stage2/pretrained/backbone.pth`.

Run the merge from the repository root:

```bash
cd stage2
bash merge.sh
```

Key outputs under `stage2/checkpoints/` are:

* `unified_backbone/unified_backbone.pth`
* `experts/sutrack_b224_<task>_expert.pth`
* `merged_heads/sutrack_b224_<task>_head_ot.pth`

## Stage 3: Assemble and Fine-Tune

Copy the Stage 2 **unified backbone**, **task experts**, and **merged heads** into the matching directories under `stage3/weight_root/`.

From the repository root, assemble the three task checkpoints:

```bash
python stage3/assemble_params_rgbt.py
python stage3/assemble_params_rgbe.py
python stage3/assemble_params_rgbd.py
```

The assembled checkpoints are saved to `stage3/pretrained/merged_<task>_ckp.pth.tar`.

Then fine-tune each task:

```bash
cd stage3
bash train_rgbt.sh
bash train_rgbe.sh
bash train_rgbd.sh
```

**Note:** The unified backbone remains **frozen** during Stage 3; task-specific components provide adaptation for each modality.

## Experiment Results

<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/OpenEvTracking/blob/main/UniMergeTrack/figures/Experiments.png" />
</p>


<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/OpenEvTracking/blob/main/UniMergeTrack/figures/Attribute.png" />
</p>

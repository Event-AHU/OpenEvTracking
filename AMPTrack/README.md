# 🔎 AMPTrack
**Decoupling Amplitude and Phase Attention in Frequency Domain for RGB-Event based Visual Object Tracking** 

Shiao Wang, Xiao Wang, Haonan Zhao, Jiarui Xu, Bo Jiang, Lin Zhu, Xin Zhao, Yonghong Tian, Jin Tang

[[Paper](https://arxiv.org/abs/2309.14611)]
[[Code](https://github.com/Event-AHU/OpenEvTracking/edit/main/AMPTrack/)]

<div align="center">

<img src="https://github.com/Event-AHU/OpenEvTracking/blob/main/AMPTrack/figures/framework.jpg" width="888">
  



# :dart: Abstract 
Existing RGB–Event visual object tracking approaches primarily rely on conventional feature-level fusion, failing to fully exploit the unique advantages of event cameras. In particular, the high dynamic range and motion-sensitive nature of event cameras are often overlooked, while low-information regions are processed uniformly, leading to unnecessary computational overhead for the backbone network. To address these issues, we propose a novel tracking framework that performs early fusion in the frequency domain, enabling effective aggregation of high-frequency information from the event modality. Specifically, RGB and event modalities are transformed from the spatial domain to the frequency domain via the Fast Fourier Transform, with their amplitude and phase components decoupled. High-frequency event information is selectively fused into RGB modality through amplitude and phase attention, enhancing feature representation while substantially reducing backbone computation. In addition, a motion-guided spatial sparsification module leverages the motion-sensitive nature of event cameras to capture the relationship between target motion cues and spatial probability distribution, filtering out low-information regions and enhancing target-relevant features. Finally, a sparse set of target-relevant features is fed into the backbone network for learning, and the tracking head predicts the final target position. Extensive experiments on three widely used RGB–Event tracking benchmark datasets, including FE108, FELT, and COESOT, demonstrate the high performance and efficiency of our method.


# :collision: Update Log 

* :fire: [2025.02.17] Based on HDETrack (CVPR2024), we have further expanded HDETrackV2, the paper and code are all released
  [[Paper](https://arxiv.org/pdf/2502.05574.pdf)]
  

* :fire: [2024.03.12] A New Long-term RGB-Event based Visual Object Tracking Benchmark Dataset (termed **FELT**) is available at
  [[Paper](https://arxiv.org/pdf/2403.05839.pdf)] 
  [[Code](https://github.com/Event-AHU/FELT_SOT_Benchmark)] 
  [[DemoVideo](https://youtu.be/6zxiBHTqOhE?si=6ARRGFdBLSxyp3G8)]


* :fire: [2024.02.28] Our code, visualizations and other experimental results have been updated.
* :fire: [2024.02.27] Our work is accepted by CVPR-2024!
* :fire: [2023.12.04] EventVOT_eval_toolkit, from [EventVOT_eval_toolki (Passcode：wsad)](https://pan.baidu.com/s/1rDsLIsNLxN6Gh9u-EdElyA?pwd=wsad)
* :fire: [2023.09.26] arXiv paper, dataset, pre-trained models, and benchmark results are all released [[arXiv](https://arxiv.org/abs/2309.14611)]




# :video_camera: Demo Video
A demo video [Youtube](https://youtu.be/FcwH7tkSXK0?si=GHOG7rfw4-GFd9dz) can be found by clicking the image below: 
<p align="center">
  <a href="https://youtu.be/FcwH7tkSXK0">
    <img src="https://github.com/Event-AHU/EventVOT_Benchmark/blob/main/figures/EventVOT_youtube.png" alt="DemoVideo" width="800"/>
  </a>
</p> 

<p align="center">
  <img src="./figures/EventVOT_samples.jpg" alt="EventVOT_samples" width="800"/>
</p>


<p align="center">
  <img src="./figures/gif.gif" alt="EventVOT_gif" width="800"/>
</p>



# :hammer: Environment 

**A distillation framework for Event Stream-based Visual Object Tracking.**

[[HDETrack_S_ep0050.pth](https://pan.baidu.com/s/1GigDXtkSd9oE04dUM3W6Nw?pwd=wsad)] Passcode：wsad

[[HDETrackV2_ep0050.pth](https://pan.baidu.com/s/1vOhcw87HpgPoSprsiL6Hew?pwd=wsad)] Passcode：wsad

[[Raw Results](https://pan.baidu.com/s/1vkC8fNisBqmIPjXzWvveWA?pwd=wsad)] Passcode：wsad


<p align="center">
  <img width="85%" src="./figures/HDETrack.jpg" alt="Framework"/>
</p>

Install env
```
conda create -n hdetrack python=3.8
conda activate hdetrack
bash install.sh
```

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

Then, put the tracking datasets EventVOT in `./data`. 

Download pre-trained [MAE ViT-Base weights](https://pan.baidu.com/s/1M1_CPXgH3PHr7MwXP-G5VQ?pwd=wsad) and put it under `$/pretrained_models`

Download teacher pre-trained [CEUTrack_ep0050.pth](https://pan.baidu.com/s/1Z6jA6bnoY8sBSbRsxaEo4w?pwd=wsad) and put it under `$/pretrained_models`

Download the trained model weights from [[HDETrack_S_ep0050.pth](https://pan.baidu.com/s/1GigDXtkSd9oE04dUM3W6Nw?pwd=wsad)] and put it under `$/output/checkpoints/train/hdetrack/hdetrack_eventvot` for test directly.

You can also access [Weight files in Dropbox](https://www.dropbox.com/scl/fo/8novqy1dg8enbjlocxb1p/AEFy8K2d0TkbNcdF1xtcvfQ?rlkey=kf5py912x2cmk6txc35xyasv6&st=d14w1433&dl=0) to download these weight files.

## Train & Test
```
# train
python tracking/train.py --script hdetrack --config hdetrack_eventvot --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0

# test
python tracking/test.py hdetrack hdetrack_eventvot --dataset eventvot --threads 1 --num_gpus 1
```


### Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX 3090 GPU.


# :dvd: EventVOT Dataset 


* **Event Image version** (train.zip 28.16GB, val.zip 703M, test.zip 9.94GB)

:floppy_disk: **Baidu Netdisk**: link：https://pan.baidu.com/s/1NLSnczJ8gnHqF-69bE7Ldg?pwd=wsad code：wsad


* **Complete version** (Event Image + Raw Event data, train.zip 180.7GB, val.zip 4.34GB, test.zip 64.88GB)
  
:floppy_disk: **Baidu Netdisk**: link：https://pan.baidu.com/s/1ZTX7O5gWlAdpKmd4R9VhYA?pwd=wsad code：wsad
  
:floppy_disk: **Dropbox**: https://www.dropbox.com/scl/fo/fv2e3i0ytrjt14ylz81dx/h?rlkey=6c2wk2z7phmbiwqpfhhe29i5p&dl=0

* If you want to download the dataset directly on the Ubuntu terminal using a script, please try this:
```
wget -O EventVOT_dataset.zip https://www.dropbox.com/scl/fo/fv2e3i0ytrjt14ylz81dx/h?rlkey=6c2wk2z7phmbiwqpfhhe29i5p"&"dl=1
```

The directory should have the below format:
```Shell
├── EventVOT dataset
    ├── Training Subset (841 videos, 180.7GB)
        ├── recording_2022-10-10_17-28-38
            ├── img
            ├── recording_2022-10-10_17-28-38.csv
            ├── groundtruth.txt
            ├── absent.txt
        ├── ... 
    ├── Testing Subset (282 videos, 64.88GB)
        ├── recording_2022-10-10_17-28-24
            ├── img
            ├── recording_2022-10-10_17-28-24.csv
            ├── groundtruth.txt
            ├── absent.txt
        ├── ...
    ├── validating Subset (18 videos, 4.34GB)
        ├── recording_2022-10-10_17-31-07
            ├── img
            ├── recording_2022-10-10_17-31-07.csv
            ├── groundtruth.txt
            ├── absent.txt
        ├── ... 
```
Normally, we only need the "img" and "..._voxel" files from the EventVOT dataset for training. During testing, we only input "img" for inference. As shown in the following figure,
<p align="center">
  <img src="./figures/EventVOT_dataset.png" alt="EventVOT_files" width="600"/>
</p>

Note: Our EventVOT dataset is an unimodal Event Dataset, if you need a multimodal RGB-E dataset, please refer to [[COESOT](https://github.com/Event-AHU/COESOT/tree/main)]， [[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)], or [[FELT](https://github.com/Event-AHU/FELT_SOT_Benchmark)].

# :triangular_ruler: Evaluation Toolkit

1. Download the EventVOT_eval_toolkit from [EventVOT_eval_toolki (Passcode：wsad)](https://pan.baidu.com/s/1rDsLIsNLxN6Gh9u-EdElyA?pwd=wsad), and open it with Matlab (over Matlab R2020).
2. add your tracking results and [baseline results (Passcode：wsad)](https://pan.baidu.com/s/1xScOxwW_y2lzoXrYtJX-RA?pwd=wsad)  in `$/eventvot_tracking_results/` and modify the name in `$/utils/config_tracker.m`
3. run `Evaluate_EventVOT_benchmark_SP_PR_only.m` for the overall performance evaluation, including SR, PR, NPR.
4. run `plot_BOC.m` for BOC score evaluation and figure plot.
5. run `plot_radar.m` for attributes radar figrue plot.
6.  run `Evaluate_EventVOT_benchmark_attributes.m` for attributes analysis and figure saved in `$/res_fig/`. 
<p align="center">
  <img width=50%" src="./figures/BOC.png" alt="Radar"/><img width="50%" src="./figures/attributes.png" alt="Radar"/>
</p>

# :chart_with_upwards_trend: Benchmark Results
The overall performance evaluation, including SR, PR, NPR.

<p align="left">
  <img width="100%" src="./figures/SRPRNPR.png" alt="SRPRNPR"/>
</p>


# :cupid: Acknowledgement 
* Thanks for the  [CEUTrack](https://github.com/Event-AHU/COESOT), [OSTrack](https://github.com/botaoye/OSTrack), [PyTracking](https://github.com/visionml/pytracking) and [ViT](https://github.com/rwightman/pytorch-image-models) library for a quickly implement.

# :newspaper: Citation 
```bibtex
@inproceedings{wang2024event,
  title={Event stream-based visual object tracking: A high-resolution benchmark dataset and a novel baseline},
  author={Wang, Xiao and Wang, Shiao and Tang, Chuanming and Zhu, Lin and Jiang, Bo and Tian, Yonghong and Tang, Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19248--19257},
  year={2024}
}
```


## Star History

<a href="https://star-history.com/#Event-AHU/EventVOT_Benchmark&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Event-AHU/EventVOT_Benchmark&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Event-AHU/EventVOT_Benchmark&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Event-AHU/EventVOT_Benchmark&type=Date" />
 </picture>
</a>
































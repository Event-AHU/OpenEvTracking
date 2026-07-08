# 🔎 APRTrack
**Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking** [[Paper](https://arxiv.org/abs/2606.26455)]

Xiao Wang, Xufeng Lou, Zikang Yan, Lan Chen*, Si-bao Chen, Yaowei Wang, Yonghong Tian, Jin Tang

# :dart: Abstract 
RGB-Event tracking improves localization robustness by fusing RGB appearance textures and dense temporal motion cues from event sensors. While this multi-modal scheme broadens tracking applicability, real-world scenes suffer diverse structured signal degradations that hinder traditional multi-modal fusion. In harsh environments, either modality can lose reliability drastically, and targets frequently appear incomplete due to occlusion, edge truncation and foreground clutter.To tackle the above challenges, we present a hierarchical perturbation and retrieval framework tailored for RGB-Event tracking with robustness against partial target missing and modal degradation, termed APRTrack. To mimic real-world signal corruption, APRTrack constructs structured degradation via two adversarial perturbation branches at the modality and spatial levels, which separately simulate full-modal failure and localized target region absence. A hierarchical routing mechanism is designed to disentangle the training pipelines of the two perturbation types, effectively eliminating feature collapse induced by superimposed degradation constraints. Furthermore, we devise Footprint-guided Channel-calibrated Hopfield Retrieval (FCHR) for reliable historical information compensation. This module evaluates retrieval confidence based on association footprints between queries and memory banks, and calibrates the retrieval metric space prior to Hopfield matching, realizing controllable historical feature compensation bounded to target regions. Extensive experiments on FE108, COESOT, VisEvent, and FELT datasets demonstrate the effectiveness of our proposed strategies for the RGB-Event visual object tracking.

### Framework 

<p align="center">
  <img width="90%" src="./figures/framework.jpg">
</p>

An overview of the proposed APRTrack framework for missing-robust RGB-Event tracking. APRTrack first maps RGB and Event template-search inputs into token representations, then applies adversarial hierarchical perturbation and footprint-guided channel-calibrated Hopfield retrieval before the Transformer backbone to model structured degradation and introduce controlled historical compensation. The fused search representation is finally fed into the tracking head for target localization.


# :collision: Update Log 



# :hammer: Environment 

Install environment
```
conda create -n aprtrack python=3.10
conda activate aprtrack
bash install.sh
```

### Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

<!--* Download pretrained model [[]()] and put it under `$/pretrained_models` for training.-->

<!--* Download the trained model weight from [[]()] and put it under `$/output/checkpoints/train/aprtrack/coesot` for testing directly.-->


# :checkered_flag: Train & Test

```
# train
python tracking/train.py --script aprtrack --config coesot --save_dir ./output --mode single --use_wandb 0

# test
python tracking/test.py --tracker_name aprtrack --tracker_param coesot --dataset_name coesot --threads 0 --num_gpus 1
```

# :cupid: Acknowledgement 
[[CEUTrack](https://github.com/Event-AHU/COESOT)] 
[[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)] 
[[FE108](https://github.com/Jee-King/ICCV2021_Event_Frame_Tracking)] 
[[FELT](https://github.com/Event-AHU/FELT_SOT_Benchmark)] 
[[Awesome_Modern_Hopfield_Networks](https://github.com/Event-AHU/Awesome_Modern_Hopfield_Networks)] 
[[OSTrack](https://github.com/botaoye/OSTrack)] 
[[SUTrack](https://github.com/chenxin-dlut/SUTrack)] 

# :newspaper: Citation 
```bibtex
@article{wang2026active,
  title={Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking},
  author={Wang, Xiao and Lou, Xufeng and Yan, Zikang and Chen, Lan and Chen, Sibao and Wang, Yaowei and Tian, Yonghong and Tang, Jin},
  journal={arXiv preprint arXiv:2606.26455},
  year={2026}
}
```






# lib/train/base_functions.py

import torch
from torch.utils.data.distributed import DistributedSampler

# datasets related
from lib.train.dataset import Coesot, Fe108, VisEvent
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process

from lib.train.trainers.base_trainer import WarmupSchedulerWrapper

def update_settings(settings, cfg):
    settings.print_interval    = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {
        'template': cfg.DATA.TEMPLATE.FACTOR,
        'search'  : cfg.DATA.SEARCH.FACTOR
    }
    settings.output_sz = {
        'template': cfg.DATA.TEMPLATE.SIZE,
        'search'  : cfg.DATA.SEARCH.SIZE
    }
    settings.center_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
        'search'  : cfg.DATA.SEARCH.CENTER_JITTER
    }
    settings.scale_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
        'search'  : cfg.DATA.SEARCH.SCALE_JITTER
    }
    settings.grad_clip_norm  = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats     = None
    settings.batchsize       = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type  = cfg.TRAIN.SCHEDULER.TYPE
    settings.backbone_stride = cfg.MODEL.BACKBONE.STRIDE

    settings.warmup_epoch   = getattr(cfg.TRAIN.SCHEDULER, 'WARMUP_EPOCH',   0)
    settings.warmup_lr_init = getattr(cfg.TRAIN.SCHEDULER, 'WARMUP_LR_INIT', 1e-6)


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in [
            "LASOT", "GOT10K_vottrain", "GOT10K_votval",
            "GOT10K_train_full", "GOT10K_official_val",
            "COCO17", "VID", "TRACKINGNET",
            "COESOT", "COESOT_VAL",
            "FE108", "FE108_VAL",
            "VisEvent", "VisEvent_VAL"
        ]
        if name == "LASOT":
            if settings.use_lmdb:
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir,
                                            split='train',
                                            image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir,
                                       split='train',
                                       image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir,
                                             split='vottrain',
                                             image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir,
                                        split='vottrain',
                                        image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir,
                                             split='train_full',
                                             image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir,
                                        split='train_full',
                                        image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir,
                                             split='votval',
                                             image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir,
                                        split='votval',
                                        image_loader=image_loader))
        if name == "GOT10K_official_val":
            datasets.append(Got10k(settings.env.got10k_val_dir,
                                    split=None,
                                    image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir,
                                                version="2017",
                                                image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir,
                                           version="2017",
                                           image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir,
                                                  image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir,
                                             image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir,
                                                  image_loader=image_loader))
            else:
                datasets.append(TrackingNet(settings.env.trackingnet_dir,
                                             image_loader=image_loader))
        if name == "COESOT":
            datasets.append(Coesot(settings.env.coesot_dir,
                                    split='train',
                                    image_loader=image_loader))
        if name == "COESOT_VAL":
            datasets.append(Coesot(settings.env.coesot_val_dir,
                                    split='val',
                                    image_loader=image_loader))
        if name == "FE108":
            datasets.append(Fe108(settings.env.fe108_dir,
                                   split='train',
                                   image_loader=image_loader))
        if name == "FE108_VAL":
            datasets.append(Fe108(settings.env.fe108_val_dir,
                                   split='val',
                                   image_loader=image_loader))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir,
                                      split='train',
                                      image_loader=image_loader))
        if name == "VisEvent_VAL":
            datasets.append(VisEvent(settings.env.visevent_val_dir,
                                      split='val',
                                      image_loader=image_loader))
    return datasets

def build_dataloaders(cfg, settings):
    transform_joint = tfm.Transform(
        tfm.ToGrayscale(probability=0.05)
    )
    transform_train = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )
    transform_val = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )
    EVENT_MEAN = getattr(cfg.DATA, 'EVENT_MEAN', [0.485, 0.456, 0.406])
    EVENT_STD  = getattr(cfg.DATA, 'EVENT_STD',  [0.229, 0.224, 0.225])
    transform_train_event = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=EVENT_MEAN, std=EVENT_STD)
    )
    transform_val_event = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=EVENT_MEAN, std=EVENT_STD)
    )
    output_sz          = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(
        search_area_factor   = search_area_factor,
        output_sz            = output_sz,
        center_jitter_factor = settings.center_jitter_factor,
        scale_jitter_factor  = settings.scale_jitter_factor,
        mode                 = 'sequence',
        transform            = transform_train,
        template_event_transform = transform_train_event,
        search_event_transform   = transform_train_event,
        joint_transform      = transform_joint,
        settings             = settings
    )
    data_processing_val = processing.STARKProcessing(
        search_area_factor   = search_area_factor,
        output_sz            = output_sz,
        center_jitter_factor = settings.center_jitter_factor,
        scale_jitter_factor  = settings.scale_jitter_factor,
        mode                 = 'sequence',
        transform            = transform_val,
        template_event_transform = transform_val_event,
        search_event_transform   = transform_val_event,
        joint_transform      = transform_joint,
        settings             = settings
    )
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search   = getattr(cfg.DATA.SEARCH,   "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls    = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)

    dataset_train = sampler.TrackingSampler(
        datasets          = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME,
                                           settings, opencv_loader),
        p_datasets        = cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch = cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap           = cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames = settings.num_search,
        num_template_frames = settings.num_template,
        processing        = data_processing_train,
        frame_sample_mode = sampler_mode,
        train_cls         = train_cls
    )
    train_sampler = (DistributedSampler(dataset_train)
                     if settings.local_rank != -1 else None)
    shuffle = False if settings.local_rank != -1 else True
    loader_train = LTRLoader(
        'train', dataset_train, training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True,
        stack_dim=1, sampler=train_sampler
    )

    dataset_val = sampler.TrackingSampler(
        datasets          = names2datasets(cfg.DATA.VAL.DATASETS_NAME,
                                           settings, opencv_loader),
        p_datasets        = cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch = cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap           = cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames = settings.num_search,
        num_template_frames = settings.num_template,
        processing        = data_processing_val,
        frame_sample_mode = sampler_mode,
        train_cls         = train_cls
    )
    val_sampler = (DistributedSampler(dataset_val)
                   if settings.local_rank != -1 else None)
    loader_val = LTRLoader(
        'val', dataset_val, training=False,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True,
        stack_dim=1, sampler=val_sampler,
        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL
    )
    return loader_train, loader_val

def _build_warmup_wrapper(optimizer, lr_scheduler, cfg):
    """
    读取 cfg.TRAIN.SCHEDULER.WARMUP_EPOCH：
      - 若 > 0：用 WarmupSchedulerWrapper 包装 lr_scheduler 后返回
      - 若 = 0：直接返回原 lr_scheduler

    args:
        optimizer    : 与 lr_scheduler 绑定的同一个 optimizer 实例
        lr_scheduler : 已构建好的 StepLR / MultiStepLR 等
        cfg          : 全局配置

    returns:
        lr_scheduler
    """
    warmup_epoch   = int(getattr(cfg.TRAIN.SCHEDULER, 'WARMUP_EPOCH',   0))
    warmup_lr_init = float(getattr(cfg.TRAIN.SCHEDULER, 'WARMUP_LR_INIT', 1e-6))

    if warmup_epoch > 0:
        lr_scheduler = WarmupSchedulerWrapper(
            base_scheduler = lr_scheduler,
            optimizer      = optimizer,
            warmup_epoch   = warmup_epoch,
            warmup_lr_init = warmup_lr_init,
        )
        if is_main_process():
            print(
                f'[Scheduler] WarmupSchedulerWrapper enabled | '
                f'warmup_epoch={warmup_epoch} | '
                f'warmup_lr_init={warmup_lr_init:.2e} | '
                f'target_lr={cfg.TRAIN.LR:.2e}'
            )
    else:
        if is_main_process():
            print('[Scheduler] warmup_epoch=0, using base scheduler directly.')

    return lr_scheduler

def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    if train_cls:
        print("Only training classification head.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters()
                        if "cls" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters()
                           if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in net.named_parameters()
                           if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported Optimizer")

    #  构建 base scheduler 
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = cfg.TRAIN.LR_DROP_EPOCH,
            gamma     = cfg.TRAIN.SCHEDULER.DECAY_RATE,
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA
        )
    else:
        raise ValueError("Unsupported scheduler")

    lr_scheduler = _build_warmup_wrapper(optimizer, lr_scheduler, cfg)

    return optimizer, lr_scheduler

def get_optimizer_scheduler_sor_ft(net, cfg):
    backbone_names     = set()
    gabor_names        = set()    
    sor_frontend_names = set()
    box_head_names     = set()
    other_names        = set()
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_names.add(n)
        elif 'kernel_gen' in n:        
            gabor_names.add(n)
        elif 'sor_frontend' in n:
            sor_frontend_names.add(n)
        elif 'box_head' in n:
            box_head_names.add(n)
        else:
            other_names.add(n)
    base_lr = cfg.TRAIN.LR
    _GABOR_LR_SCALE = 0.1            
    param_dicts = [
        # group0: sor_frontend
        {
            "params": [p for n, p in net.named_parameters()
                       if n in sor_frontend_names],
            "lr": base_lr,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
        },
        # group1: box_head
        {
            "params": [p for n, p in net.named_parameters()
                       if n in box_head_names],
            "lr": base_lr,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
        },
        # group2: backbone
        {
            "params": [p for n, p in net.named_parameters()
                       if n in backbone_names],
            "lr": base_lr * cfg.TRAIN.BACKBONE_MULTIPLIER,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
        },
        # group3: 其他
        {
            "params": [p for n, p in net.named_parameters()
                       if n in other_names],
            "lr": base_lr,
            "weight_decay": 0.0,
        },
        # group4: Gabor kernel_gen — 保守 LR
        {
            "params": [p for n, p in net.named_parameters()
                       if n in gabor_names],
            "lr": base_lr * _GABOR_LR_SCALE,
            "weight_decay": 0.0,   
        },
    ]

    if is_main_process():
        for i, (names, label) in enumerate([
            (sor_frontend_names, 'sor_frontend'),
            (box_head_names,     'box_head'),
            (backbone_names,     'backbone'),
            (other_names,        'other'),
        ]):
            param_count = sum(
                p.numel() for n, p in net.named_parameters()
                if n in names
            )
            print(f'[OptimizerGroup{i}] {label}: '
                  f'{len(names)} tensors, {param_count:,} params, '
                  f'lr={param_dicts[i]["lr"]:.2e}')

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=base_lr,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported Optimizer")

    #  构建 base scheduler 
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.TRAIN.LR_DROP_EPOCH,
            gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA
        )
    else:
        raise ValueError("Unsupported scheduler")

    lr_scheduler = _build_warmup_wrapper(optimizer, lr_scheduler, cfg)

    return optimizer, lr_scheduler
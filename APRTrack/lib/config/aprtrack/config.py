from easydict import EasyDict as edict
import yaml

"""
Add default config for aprtrack.
"""
cfg = edict()

       
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_hivit_base_1600ep.pth"
cfg.MODEL.PRETRAIN_PATH = ""

cfg.MODEL.APG = edict()
cfg.MODEL.APG.ENABLE = True
cfg.MODEL.APG.WARMUP_EPOCHS = 10
cfg.MODEL.APG.MODAL_ENABLE = True
cfg.MODEL.APG.SPATIAL_ENABLE = True
cfg.MODEL.APG.MODAL_BALANCE_WEIGHT = 0.1
cfg.MODEL.APG.MODAL_TARGET_PROBS = [0.2, 0.4, 0.4]
cfg.MODEL.APG.SPATIAL_MASK_RATIO_MIN = 0.15
cfg.MODEL.APG.SPATIAL_MASK_RATIO_MAX = 0.30
cfg.MODEL.APG.SPATIAL_TEMPERATURE = 1.0
cfg.MODEL.APG.SPATIAL_ANCHOR_OVERLAP_THRESHOLD = 0.45
cfg.MODEL.APG.SPATIAL_ANCHOR_PENALTY_WEIGHT = 0.5
cfg.MODEL.APG.ROUTE_PROBS = [0.4, 0.3, 0.3]
cfg.MODEL.APG.ROUTE_COMPLETION = [False, False, True]

cfg.MODEL.COMPLETION = edict()
cfg.MODEL.COMPLETION.ENABLE = True
cfg.MODEL.COMPLETION.MEMORY_SIZE = 5
cfg.MODEL.COMPLETION.NUM_HEADS = 4
cfg.MODEL.COMPLETION.DROPOUT = 0.1
cfg.MODEL.COMPLETION.GATE_INIT_VALUE = -1.0

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

                
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "hivit_base"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.ATTN_DROP_RATE = 0.0

            
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.HEAD.INPUT_TYPE = "add"

       
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 50
cfg.TRAIN.LR_DROP_EPOCH = 40
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FOCAL_WEIGHT = 1.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 5
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20                                     
cfg.TRAIN.CE_WARM_EPOCH = 80                                       
cfg.TRAIN.DROP_PATH_RATE = 0.1                                   

                 
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.SCHEDULER.MILESTONES = [50, 80]
cfg.TRAIN.SCHEDULER.GAMMA = 0.1

      
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"                    
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
            
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 30000
          
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
             
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 3
               
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 2
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

      
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 50
cfg.TEST.UPDATE_INTERVAL = 25
cfg.TEST.UPDATE_THRESHOLD = 0.70


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename, 'r', encoding='utf-8') as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)

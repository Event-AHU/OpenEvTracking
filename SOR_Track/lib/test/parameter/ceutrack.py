# lib/test/parameter/ceutrack.py

from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ceutrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch: int = None):
    """
    args:
        yaml_name: config 文件名（不含.yaml），如 'ceutrack_visevent'
        epoch:     覆盖 yaml 中的 TEST.EPOCH，为 None 时使用 yaml 值
    """
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir

    yaml_file = os.path.join(prj_dir, 'experiments/ceutrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size   = cfg.TEST.TEMPLATE_SIZE
    params.search_factor   = cfg.TEST.SEARCH_FACTOR
    params.search_size     = cfg.TEST.SEARCH_SIZE
    params.parameter_name  = yaml_name   

    # epoch 优先级：命令行 > yaml
    test_epoch = epoch if epoch is not None else cfg.TEST.EPOCH
    print(f"[parameters] Using epoch={test_epoch} "
          f"({'from --runid' if epoch is not None else 'from yaml TEST.EPOCH'})")

    # checkpoint 路径
    params.checkpoint = os.path.join(
        save_dir,
        "output/checkpoints/train/ceutrack/%s/CEUTrack_ep%04d.pth.tar" % (yaml_name, test_epoch)
    )
    print(f"[parameters] checkpoint: {params.checkpoint}")

    # 验证 checkpoint 存在
    if not os.path.isfile(params.checkpoint):
        raise FileNotFoundError(
            f"[parameters] Checkpoint NOT found: {params.checkpoint}\n"
            f"Available checkpoints in dir:\n" +
            "\n".join(
                sorted(os.listdir(os.path.dirname(params.checkpoint)))
                if os.path.isdir(os.path.dirname(params.checkpoint)) else ["<dir not found>"]
            )
        )

    params.save_all_boxes = False
    return params
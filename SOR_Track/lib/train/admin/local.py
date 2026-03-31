class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/zhangfan/CEUTrack_COESOT_rgbe/pretrained_networks'
        self.coesot_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/COESOT/train'
        self.coesot_val_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/COESOT/test'
        self.fe108_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/FE108/train'
        self.fe108_val_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/FE108/test'
        self.visevent_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/visevent/train'
        self.visevent_val_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/visevent/test'
